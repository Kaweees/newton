.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

libuipc / IsaacIPC
==================

:class:`~newton.solvers.SolverUIPC` connects Newton's model and state interface
to `libuipc <https://github.com/spiriMirror/libuipc>`__, the GPU IPC physics
backend used by `IsaacIPC <https://arxiv.org/abs/2605.24339>`__.
It advances tetrahedral deformables with implicit integration, barrier contact,
continuous collision detection, and Coulomb friction.

.. experimental::

    ``SolverUIPC`` is experimental. The integration uses the published
    ``pyuipc==0.0.29`` backend. It does not implement articulation constraints
    or the Isaac Sim/Fabric bridge. The paper's GMCP normal-contact model is
    available separately in Warp; see :doc:`/concepts/gmcp`. It is not injected
    into the native libuipc solve.

Install and run
---------------

The optional backend requires an NVIDIA CUDA device, Python 3.11--3.14, and
Linux or Windows on x86-64. The Linux wheel requires GLIBC 2.34 or newer.
Install into a Newton checkout with:

.. code-block:: console

    uv sync --extra uipc

For a packaged installation, use ``pip install "newton[uipc]"``. The extra is
excluded on platforms without wheels; constructing the solver still requires
a working ``uipc`` installation and a CUDA model. The pinned PyPI wheel
declares Apache-2.0 and includes its own LICENSE and NOTICE. Newton does not
vendor the native library. The upstream source tree has a different license
file under ``python/``; the packaging license discrepancy needs upstream
clarification before substituting or redistributing a source build.

This complete example drops a small soft cube onto a plane:

.. code-block:: python

    import warp as wp
    import newton

    builder = newton.ModelBuilder()
    builder.add_soft_grid(
        pos=wp.vec3(0.0, 0.0, 0.3),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        dim_x=1, dim_y=1, dim_z=1,
        cell_x=0.1, cell_y=0.1, cell_z=0.1,
        density=1000.0,
        k_mu=1.0e4, k_lambda=2.0e4, k_damp=0.0,
        particle_radius=0.0,
    )
    builder.add_ground_plane()
    model = builder.finalize(device="cuda:0")
    solver = newton.solvers.SolverUIPC(model, contact_distance=0.001)
    state_in, state_out = model.state(), model.state()

    try:
        for _ in range(60):
            state_in.clear_forces()
            solver.step(state_in, state_out, control=None, contacts=None, dt=1.0 / 60.0)
            state_in, state_out = state_out, state_in
    finally:
        solver.close()

Pass ``contacts=None``: libuipc owns collision detection and its barrier
constraints. Newton contact buffers cannot be converted into IPC constraints.
The first step initializes the native world and fixes the timestep. Changing
``dt`` requires a new solver.

Supported model data
--------------------

Every particle must belong to a tetrahedron. The adapter preserves each
particle's mass and external force, the tetrahedral rest geometry, and each
world's gravity. Particles with zero inverse mass or without
:attr:`~newton.ParticleFlags.ACTIVE` are fixed. Each tetrahedron must belong to
one world. Distinct local worlds do not collide with each other; global
geometry interacts with all of them.

The native stable Neo-Hookean energy uses ``mu = k_mu`` and
``lambda = k_lambda + k_mu`` as its stored coefficients. This matches the
physical Lamé parameters at small strain. Its finite-strain energy differs
from Newton solvers that use the logarithmically stabilized Smith energy.
Both Lamé inputs must be positive, and ``k_damp`` must be zero.
Surface triangles and edges may be present for visualization with zero
material coefficients; collision surfaces are extracted from the tetrahedra.

Collision obstacles support infinite static planes, boxes, and triangle
meshes with ``COLLIDE_PARTICLES`` enabled. Boxes use their exact triangulated
surface. Mesh scale, local transform, and body pose are composed before
contact detection. Body-attached obstacles require kinematic bodies. Their
poses are held fixed for the entire step; body and joint arrays pass through
unchanged. Updating a body pose between steps is a prescribed repositioning,
so it must leave a nonintersecting configuration. There is no swept contact
test for that repositioning and no contact reaction on the prescribed body.

Contact distance, stiffness, and friction are uniform solver parameters.
The backend may clamp the requested stiffness to its adaptive bounds.
Per-shape Newton contact materials, contact gaps, and particle radii do not
parameterize this surface barrier. Nonzero shape margins are rejected.
Cloth, springs, particle-body attachments, dynamic rigid bodies, active
tetrahedral contraction, and deformable damping are unsupported.

State, rendering, and USD
-------------------------

Each step reads ``state_in.particle_q``, ``particle_qd``, and ``particle_f`` and
writes positions and velocities to ``state_out``. It supports both separate
and aliased states. Reset the desired state arrays before the next step;
``reset()`` only validates the optional world mask because no trajectory is
retained independently of the input state. After changing model parameters,
call ``notify_model_changed(ModelFlags.MODEL_PROPERTIES)`` to rebuild cached
backend data on the next step. The geometry's rest positions must continue
to match ``tet_poses``.

Newton viewers can display the simulated surface directly. Existing
:class:`~newton.geometry.DeformableVisualMesh` bindings and
:class:`~newton.geometry.DeformableVisuals` also work with the output particle
positions, preserving the visual mesh's topology, UVs, and texture. This does
not add the paper's nearest-surface mapper with normal offsets.

USD support reuses :meth:`~newton.ModelBuilder.add_usd`: imported
``UsdGeom.TetMesh`` geometry with ``PhysicsVolumeDeformableSimAPI`` and
``PhysicsVolumeDeformableMaterialAPI`` produces the same tetrahedral data as
builder-authored models. The supported mass, density, and elastic material
fields need no additional schema. Solver selection, convergence settings,
and backend workspace are runtime options, so a new USD schema is not
applicable. Unsupported imported features are subject to the same checks as
builder-authored features.

Execution limits
----------------

The adapter synchronizes through host memory to exchange float32 Newton
states with float64 native states. CUDA graph capture, differentiation,
solver observables, and a run-to-run determinism guarantee are not supported.
The pinned backend retains native CUDA caches: use one GPU per process and
separate processes for other GPUs. The adapter rejects switching devices
before invoking the native backend.

Start with nonintersecting geometry and positive tetrahedral volumes.
Teleporting a state into contact violates IPC's feasible-start assumption.
Native initialization errors, failed convergence, and nonfinite results raise
exceptions without publishing a new Newton output state. Set ``workspace``
to retain backend diagnostics in a chosen directory. Native libuipc logging
is controlled by ``uipc.Logger`` and is independent of Newton logging.
