# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Optional libuipc adapter for tetrahedral deformables."""

from __future__ import annotations

import math
import tempfile
from importlib.metadata import version
from pathlib import Path
from typing import ClassVar

import numpy as np
import warp as wp

from ...geometry import GeoType, ParticleFlags, ShapeFlags
from ...sim import BodyFlags, Contacts, Control, Model, ModelFlags, State
from ..solver import SolverBase


class SolverUIPC(SolverBase):
    """Simulate tetrahedral solids with the optional libuipc CUDA backend.

    libuipc provides the IPC dynamics used by IsaacIPC. This adapter supports
    tetrahedra, pinned particles, external particle forces, isolated Newton
    worlds, and plane, box, and triangle mesh obstacles. Bodies must be
    kinematic: their input poses are held fixed during each step. Body and
    joint dynamics, cloth, springs, damping, and muscle activation are not
    supported. See :doc:`/solvers/uipc` for material and contact conventions.

    The first :meth:`step` initializes the backend and fixes its timestep.
    Each step imports the input state's positions and velocities, so state
    swaps, state edits, and resets do not depend on a hidden backend trajectory.
    Transfers currently pass through host memory. CUDA graph capture and
    differentiation are not supported. Input configurations must be free of
    intersections and inverted tetrahedra, including after prescribed motion.

    IsaacIPC's GMCP contact model and Isaac Sim/Fabric bridge are outside the
    scope of this adapter.

    .. experimental::
    """

    # pyuipc retains native CUDA caches after engine destruction. Switching
    # devices in one process can cause an illegal memory access in 0.0.29.
    _backend_device: ClassVar[int | None] = None

    def __init__(
        self,
        model: Model,
        *,
        contact_distance: float = 1.0e-3,
        contact_stiffness: float = 1.0e9,
        friction: float = 0.5,
        max_iterations: int = 1024,
        velocity_tolerance: float = 1.0e-3,
        workspace: str | None = None,
    ) -> None:
        """Create a libuipc solver without initializing its CUDA world.

        Args:
            model: Newton model to simulate, allocated on a CUDA device.
            contact_distance: IPC barrier activation distance [m].
            contact_stiffness: Requested uniform IPC contact stiffness [Pa],
                subject to the backend's adaptive stiffness bounds.
            friction: Uniform Coulomb friction coefficient, dimensionless.
            max_iterations: Maximum nonlinear iterations per step.
            velocity_tolerance: Nonlinear convergence tolerance [m/s].
            workspace: Directory for backend diagnostics. If None, use a
                temporary directory owned by this solver.
        """
        super().__init__(model)
        for name, value in (
            ("contact_distance", contact_distance),
            ("contact_stiffness", contact_stiffness),
            ("velocity_tolerance", velocity_tolerance),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(friction) or friction < 0.0:
            raise ValueError("friction must be finite and nonnegative")
        if isinstance(max_iterations, bool) or not isinstance(max_iterations, int) or max_iterations < 1:
            raise ValueError("max_iterations must be a positive integer")
        self._validate_model()

        # Keep the optional native extension out of Newton's public imports.
        try:
            import uipc  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError("SolverUIPC requires pyuipc; install it with `uv pip install 'newton[uipc]'`.") from exc
        if version("pyuipc") != "0.0.29":
            raise ImportError("SolverUIPC currently requires pyuipc==0.0.29; install `newton[uipc]`.")
        self._uipc = uipc
        self._contact_distance = contact_distance
        self._contact_stiffness = contact_stiffness
        self._friction = friction
        self._max_iterations = max_iterations
        self._velocity_tolerance = velocity_tolerance
        self._temporary_workspace = tempfile.TemporaryDirectory(prefix="newton-uipc-") if workspace is None else None
        self._workspace = Path(self._temporary_workspace.name if workspace is None else workspace)
        self._workspace.mkdir(parents=True, exist_ok=True)
        self._world = None
        self._engine = None
        self._scene = None
        self._accessor = None
        self._records = []
        self._dt = None
        self._closed = False

    def _validate_model(self) -> None:
        model = self.model
        if not model.device.is_cuda:
            raise ValueError("SolverUIPC requires a CUDA model")
        if model.requires_grad:
            raise ValueError("SolverUIPC does not support differentiation")
        if model.tet_count == 0:
            raise ValueError("SolverUIPC requires a tetrahedral deformable model")
        for name in ("spring_count", "attachment_particle_body_count", "tendon_count", "muscle_count"):
            if getattr(model, name):
                raise ValueError(f"SolverUIPC does not support {name.removesuffix('_count')}")
        if model.body_count and np.any(model.body_flags.numpy() != int(BodyFlags.KINEMATIC)):
            raise ValueError("SolverUIPC only supports kinematic bodies; rigid body dynamics require another solver")
        if model.tri_count and np.any(model.tri_materials.numpy() != 0.0):
            raise ValueError("SolverUIPC requires zero surface triangle material coefficients; cloth is unsupported")
        if model.edge_count and np.any(model.edge_bending_properties.numpy() != 0.0):
            raise ValueError("SolverUIPC does not support bending edges")
        materials = model.tet_materials.numpy()
        if not np.all(np.isfinite(materials)) or np.any(materials[:, :2] <= 0.0):
            raise ValueError("SolverUIPC requires finite positive tetrahedral Lamé parameters")
        if np.any(materials[:, 2] != 0.0):
            raise ValueError("SolverUIPC does not support tetrahedral damping (k_damp must be zero)")
        if np.any(model.tet_activations.numpy() != 0.0):
            raise ValueError("SolverUIPC does not support tetrahedral activation")
        indices = model.tet_indices.numpy().reshape(-1, 4)
        if len(np.unique(indices)) != model.particle_count:
            raise ValueError("SolverUIPC requires every particle to belong to a tetrahedron")
        worlds = model.particle_world.numpy()
        if np.any(worlds[indices] != worlds[indices[:, :1]]):
            raise ValueError("SolverUIPC does not support tetrahedra spanning multiple worlds")
        positions = model.particle_q.numpy().astype(np.float64)
        edges = (positions[indices[:, 1:]] - positions[indices[:, :1]]).transpose(0, 2, 1)
        if not np.all(np.isfinite(positions)) or np.any(np.linalg.det(edges) <= 0.0):
            raise ValueError("SolverUIPC requires finite, positively oriented rest tetrahedra")
        if not np.allclose(np.linalg.inv(edges), model.tet_poses.numpy(), rtol=1.0e-4, atol=1.0e-6):
            raise ValueError("SolverUIPC requires tet_poses to match the rest geometry in model.particle_q")
        mass, inv_mass = model.particle_mass.numpy(), model.particle_inv_mass.numpy()
        if not np.all(np.isfinite(mass)) or np.any(mass < 0.0) or not np.all(np.isfinite(inv_mass)):
            raise ValueError("SolverUIPC requires finite nonnegative particle masses")
        if np.any(inv_mass < 0.0) or np.any((inv_mass > 0.0) & (mass <= 0.0)):
            raise ValueError("SolverUIPC requires consistent particle masses and inverse masses")
        flags = model.particle_flags.numpy()
        if np.any((flags & int(ParticleFlags.PROXY)) != 0):
            raise ValueError("SolverUIPC does not support coupled-solver proxy particles")
        if not np.all(np.isfinite(model.gravity.numpy())):
            raise ValueError("SolverUIPC requires finite gravity")
        if model.shape_count:
            enabled = (model.shape_flags.numpy() & int(ShapeFlags.COLLIDE_PARTICLES)) != 0
            kinds = model.shape_type.numpy()
            if np.any(enabled & ~np.isin(kinds, [GeoType.PLANE, GeoType.BOX, GeoType.MESH])):
                raise ValueError("SolverUIPC supports only plane, box, and triangle mesh collision shapes")
            planes = enabled & (kinds == GeoType.PLANE)
            if np.any(model.shape_body.numpy()[planes] != -1):
                raise ValueError("SolverUIPC planes must be static (body=-1)")
            if np.any(model.shape_scale.numpy()[planes, :2] != 0.0):
                raise ValueError("SolverUIPC supports infinite planes only; use a mesh for a finite plane")
            if np.any(model.shape_margin.numpy()[enabled] != 0.0):
                raise ValueError("SolverUIPC does not support shape margins; use contact_distance")

    def _initialize(self, dt: float, state_q, state_qd, state_f, body_q) -> None:
        if SolverUIPC._backend_device is not None and SolverUIPC._backend_device != self.model.device.ordinal:
            raise RuntimeError("pyuipc 0.0.29 supports one CUDA device per process; use a separate process per GPU")
        SolverUIPC._backend_device = self.model.device.ordinal
        uipc, model = self._uipc, self.model
        geometry, constitution = uipc.geometry, uipc.constitution
        config = uipc.Scene.default_config()
        config["dt"] = dt
        config["gravity"] = [[0.0], [0.0], [0.0]]
        config["contact"]["d_hat"] = self._contact_distance
        config["newton"]["max_iter"] = self._max_iterations
        config["newton"]["velocity_tol"] = self._velocity_tolerance
        config["newton"]["semi_implicit"]["enable"] = 0
        engine_config = uipc.Engine.default_config()
        engine_config["gpu"]["device"] = model.device.ordinal
        engine_config["extras"]["gui"]["enable"] = False
        engine = uipc.Engine("cuda", str(self._workspace), engine_config)
        scene = uipc.Scene(config)
        world = uipc.World(engine)
        scene.contact_tabular().default_model(self._friction, self._contact_stiffness)
        obstacle_element = scene.contact_tabular().create("prescribed_obstacles")
        scene.contact_tabular().insert(obstacle_element, obstacle_element, 0.0, self._contact_stiffness, enable=False)
        subscenes = {-1: scene.subscene_tabular().default_element()}
        for world_id in range(model.world_count):
            element = scene.subscene_tabular().create(f"world_{world_id}")
            scene.subscene_tabular().insert(subscenes[-1], element, True)
            subscenes[world_id] = element

        positions = model.particle_q.numpy().astype(np.float64)
        indices = model.tet_indices.numpy().reshape(-1, 4)
        worlds = model.particle_world.numpy()
        mass = model.particle_mass.numpy().astype(np.float64)
        fixed = (model.particle_inv_mass.numpy() == 0.0) | (
            (model.particle_flags.numpy() & int(ParticleFlags.ACTIVE)) == 0
        )
        gravity = model.gravity.numpy().astype(np.float64)
        materials = model.tet_materials.numpy().astype(np.float64)
        records = []
        for world_id in np.unique(worlds):
            particle_ids = np.flatnonzero(worlds == world_id)
            tet_ids = np.flatnonzero(worlds[indices[:, 0]] == world_id)
            local = np.full(model.particle_count, -1, dtype=np.int32)
            local[particle_ids] = np.arange(len(particle_ids), dtype=np.int32)
            mesh = geometry.tetmesh(positions[particle_ids], local[indices[tet_ids]])
            geometry.label_surface(mesh)
            geometry.label_triangle_orient(mesh)
            constitution.StableNeoHookean().apply_to(mesh, constitution.ElasticModuli.lame(1.0, 1.0))
            # The wheel's apply_to uses a calibration for a different energy.
            # SNK1 needs mu=mu_L and lambda=lambda_L+mu_L to match physical Lamé inputs.
            uipc.view(mesh.tetrahedra().find("mu"))[:] = materials[tet_ids, 0]
            uipc.view(mesh.tetrahedra().find("lambda"))[:] = materials[tet_ids, 1] + materials[tet_ids, 0]
            uipc.view(mesh.vertices().create("mass", 1.0))[:] = np.where(
                mass[particle_ids] > 0.0, mass[particle_ids], 1.0
            )
            uipc.view(mesh.vertices().find(uipc.builtin.is_fixed))[:] = fixed[particle_ids]
            mesh.vertices().create(uipc.builtin.gravity, gravity[world_id])
            constitution.FiniteElementExternalForce().apply_to(mesh, np.zeros(3))
            subscenes[int(world_id)].apply_to(mesh)
            obj = scene.objects().create(f"particles_{world_id}")
            slot, _ = obj.geometries().create(mesh)
            forces = np.zeros((len(particle_ids), 3), dtype=np.float64)

            def animate(info, forces=forces):
                # The backend clears these attributes before invoking animators.
                geo = info.geo_slots()[0].geometry()
                uipc.view(geo.vertices().find("external_force/is_constrained"))[:] = np.any(forces != 0.0, axis=1)
                uipc.view(geo.vertices().find("external_force")).reshape(-1, 3)[:] = forces

            scene.animator().insert(obj, animate)
            records.append({"slot": slot, "particles": particle_ids, "forces": forces, "fixed": fixed[particle_ids]})

        shape_flags = model.shape_flags.numpy() if model.shape_count else []
        for shape_id, flags in enumerate(shape_flags):
            if flags & ShapeFlags.COLLIDE_PARTICLES:
                self._add_obstacle(scene, subscenes, obstacle_element, records, shape_id)

        # create() stores separate live/rest geometries. Initialize only the
        # live geometry from State before native collision sanity checking.
        for record in records:
            mesh = record["slot"].geometry()
            record["q"] = uipc.view(mesh.positions()).reshape(-1, 3)
            record["qd"] = uipc.view(mesh.vertices().find(uipc.builtin.velocity)).reshape(-1, 3)
        self._write_state(records, state_q, state_qd, state_f, body_q)
        world.init(scene)
        if not world.is_valid():
            raise RuntimeError(f"libuipc initialization failed; see diagnostics in {self._workspace}")
        accessor = world.features().find(uipc.core.FiniteElementStateAccessorFeature)
        if accessor is None:
            raise RuntimeError("libuipc did not provide its finite element state accessor")
        for record in records:
            mesh = record["slot"].geometry()
            offset = int(uipc.view(mesh.meta().find(uipc.builtin.backend_fem_vertex_offset))[0])
            state_geo = accessor.create_geometry(offset, mesh.vertices().size())
            state_geo.vertices().create(uipc.builtin.position, np.zeros(3))
            state_geo.vertices().create(uipc.builtin.velocity, np.zeros(3))
            record["state"] = state_geo
            record["q"] = uipc.view(state_geo.vertices().find(uipc.builtin.position)).reshape(-1, 3)
            record["qd"] = uipc.view(state_geo.vertices().find(uipc.builtin.velocity)).reshape(-1, 3)

        self._engine, self._scene, self._world = engine, scene, world
        self._accessor, self._records, self._dt = accessor, records, dt

    @staticmethod
    def _write_state(records, positions, velocities, forces, body_q) -> None:
        for record in records:
            ids = record["particles"]
            if ids is not None:
                record["q"][:] = positions[ids]
                record["qd"][:] = velocities[ids]
                record["qd"][record["fixed"]] = 0.0
                record["forces"][:] = forces[ids]
            else:
                points = record["points"]
                if record["body"] >= 0:
                    xform = wp.transform(*body_q[record["body"]])
                    points = np.array([wp.transform_point(xform, wp.vec3(*p)) for p in points])
                record["q"][:] = points
                record["qd"][:] = 0.0

    def _add_obstacle(self, scene, subscenes, obstacle_element, records, shape_id: int) -> None:
        uipc, model = self._uipc, self.model
        kind = model.shape_type.numpy()[shape_id]
        xform = wp.transform(*model.shape_transform.numpy()[shape_id])
        scale = model.shape_scale.numpy()[shape_id]
        world_id = int(model.shape_world.numpy()[shape_id])
        obj = scene.objects().create(f"shape_{shape_id}")
        if kind == GeoType.PLANE:
            normal = np.array(wp.transform_vector(xform, wp.vec3(0.0, 0.0, 1.0)), dtype=np.float64)
            height = float(np.dot(normal, np.array(wp.transform_get_translation(xform))))
            mesh = uipc.geometry.ground(height, normal)
            subscenes[world_id].apply_to(mesh)
            obstacle_element.apply_to(mesh)
            obj.geometries().create(mesh)
            return
        if kind == GeoType.BOX:
            points = (
                np.array(
                    [
                        [-1, -1, -1],
                        [1, -1, -1],
                        [1, 1, -1],
                        [-1, 1, -1],
                        [-1, -1, 1],
                        [1, -1, 1],
                        [1, 1, 1],
                        [-1, 1, 1],
                    ],
                    dtype=np.float64,
                )
                * scale
            )
            triangles = np.array(
                [
                    [0, 2, 1],
                    [0, 3, 2],
                    [4, 5, 6],
                    [4, 6, 7],
                    [0, 1, 5],
                    [0, 5, 4],
                    [1, 2, 6],
                    [1, 6, 5],
                    [2, 3, 7],
                    [2, 7, 6],
                    [3, 0, 4],
                    [3, 4, 7],
                ],
                dtype=np.int32,
            )
        else:
            source = model.shape_source[shape_id]
            points = np.asarray(source.vertices, dtype=np.float64) * scale
            triangles = np.asarray(source.indices, dtype=np.int32).reshape(-1, 3)
        points = np.array([wp.transform_point(xform, wp.vec3(*p)) for p in points], dtype=np.float64)
        body = int(model.shape_body.numpy()[shape_id])
        rest_points = points.copy()
        if body >= 0:
            body_xform = wp.transform(*model.body_q.numpy()[body])
            rest_points = np.array([wp.transform_point(body_xform, wp.vec3(*p)) for p in points], dtype=np.float64)
        mesh = uipc.geometry.trimesh(rest_points, triangles)
        uipc.geometry.label_surface(mesh)
        uipc.constitution.Empty().apply_to(mesh, thickness=0.0)
        uipc.view(mesh.vertices().find(uipc.builtin.is_fixed))[:] = 1
        mesh.vertices().create("mass", 1.0)
        subscenes[world_id].apply_to(mesh)
        obstacle_element.apply_to(mesh)
        slot, _ = obj.geometries().create(mesh)
        records.append({"slot": slot, "particles": None, "points": points, "body": body})

    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control | None,
        contacts: Contacts | None,
        dt: float,
        *,
        observables: SolverBase.Observables | None = None,
    ) -> None:
        """Advance the deformables, preserving prescribed body state.

        Args:
            state_in: Input positions [m], velocities [m/s], and forces [N].
            state_out: Output state. May alias ``state_in``.
            control: Optional control; nonzero tetrahedral activations are unsupported.
            contacts: Must be None; libuipc performs its own collision detection.
            dt: Positive timestep [s], fixed after the first step.
            observables: Optional empty observable container; force observables are unsupported.
        """
        # libuipc uses the current CUDA context even when gpu.device is supplied.
        with wp.ScopedDevice(self.model.device):
            self._step(state_in, state_out, control, contacts, dt, observables)

    def _step(self, state_in, state_out, control, contacts, dt, observables):
        if self._closed:
            raise RuntimeError("SolverUIPC is closed")
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        if self._dt is not None and dt != self._dt:
            raise ValueError("SolverUIPC requires a fixed dt; construct another solver to change the timestep")
        if contacts is not None:
            raise ValueError("SolverUIPC performs its own collision detection; pass contacts=None")
        self.validate_observables(observables, contacts)
        if wp.get_stream(self.model.device).is_capturing:
            raise RuntimeError("SolverUIPC cannot run inside CUDA graph capture")
        activations = self.model.tet_activations if control is None else control.tet_activations
        if activations is not None and np.any(activations.numpy() != 0.0):
            raise ValueError("SolverUIPC does not support tetrahedral activation")
        positions = state_in.particle_q.numpy()
        velocities = state_in.particle_qd.numpy()
        forces = state_in.particle_f.numpy()
        if not all(np.all(np.isfinite(a)) for a in (positions, velocities, forces)):
            raise ValueError("SolverUIPC requires finite particle positions, velocities, and forces")
        indices = self.model.tet_indices.numpy().reshape(-1, 4)
        tet_q = positions[indices].astype(np.float64)
        if np.any(np.linalg.det((tet_q[:, 1:] - tet_q[:, :1]).transpose(0, 2, 1)) <= 0.0):
            raise ValueError("SolverUIPC requires non-inverted input tetrahedra")
        body_q = state_in.body_q.numpy() if self.model.body_count else None
        if body_q is not None and not np.all(np.isfinite(body_q)):
            raise ValueError("SolverUIPC requires finite prescribed body poses")
        if self._world is None:
            self._initialize(dt, positions, velocities, forces, body_q)
        self._write_state(self._records, positions, velocities, forces, body_q)
        for record in self._records:
            self._accessor.copy_from(record["state"])
        self._world.advance()
        if not self._world.is_valid():
            raise RuntimeError(f"libuipc step failed; see diagnostics in {self._workspace}")
        stats = self._engine.frame_stats()
        if not stats.get("completed") or not stats.get("converged"):
            raise RuntimeError(f"libuipc failed to converge: {stats}")
        self._world.retrieve()
        for record in self._records:
            ids = record["particles"]
            if ids is not None:
                self._accessor.copy_to(record["state"])
                positions[ids], velocities[ids] = record["q"], record["qd"]
        if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(velocities)):
            raise RuntimeError("libuipc returned nonfinite particle state")
        state_out.particle_q.assign(positions)
        state_out.particle_qd.assign(velocities)
        if self.model.body_count:
            wp.copy(state_out.body_q, state_in.body_q)
            wp.copy(state_out.body_qd, state_in.body_qd)
        if self.model.joint_count:
            wp.copy(state_out.joint_q, state_in.joint_q)
            wp.copy(state_out.joint_qd, state_in.joint_qd)

    def notify_model_changed(self, flags: ModelFlags | int) -> None:
        """Rebuild cached backend data on the next step after a model edit.

        Args:
            flags: Nonzero model change flags. Rebuilds all cached properties,
                including masses, materials, gravity, and collision geometry.
        """
        if flags:
            self._validate_model()
            self._release_backend()

    def _release_backend(self) -> None:
        with wp.ScopedDevice(self.model.device):
            self._records = []
            self._accessor = None
            self._world = None
            self._scene = None
            self._engine = None

    def close(self) -> None:
        """Release the native CUDA world and any owned temporary workspace.

        Safe to call more than once. A closed solver cannot be stepped again.
        User-supplied workspace directories are retained for diagnostics.
        """
        self._release_backend()
        if self._temporary_workspace is not None:
            self._temporary_workspace.cleanup()
        self._closed = True

    def __del__(self):
        if getattr(self, "_engine", None) is not None or getattr(self, "_temporary_workspace", None) is not None:
            self.close()
