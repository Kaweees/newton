# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Numerical integration and collision tests for the optional libuipc backend."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.tests._usd_deformable_test_utils import (
    _apply_deformable_body_api,
    _author_unit_tet,
    _bind_deformable_material,
    _deformable_stage,
)
from newton.tests.unittest_utils import add_function_test, get_selected_cuda_test_devices, requires


def _add_tet(builder, *, height=1.0, masses=(1.0, 2.0, 3.0, 4.0)):
    start = builder.particle_count
    for p, mass in zip(((0, 0, height), (0.1, 0, height), (0, 0.1, height), (0, 0, height + 0.1)), masses, strict=True):
        builder.add_particle(wp.vec3(*p), wp.vec3(0.0), mass, radius=0.0)
    builder.add_tetrahedron(start, start + 1, start + 2, start + 3, k_mu=1.0e4, k_lambda=2.0e4)


def test_state_and_forces(test, device):
    """Preserve unequal masses, forces, input state, resets, and in-place stepping."""
    builder = newton.ModelBuilder()
    _add_tet(builder)
    model = builder.finalize(device=device)
    gravity = np.array([0.4, -0.5, -1.0], dtype=np.float32)
    model.set_gravity(gravity)
    solver = newton.solvers.SolverUIPC(model)
    state, out = model.state(), model.state()
    dt = 0.01
    initial = state.particle_q.numpy() + np.array([0.25, 0.5, 0.75], dtype=np.float32)
    state.particle_q.assign(initial)
    velocity = np.tile([0.1, 0.2, 0.3], (4, 1)).astype(np.float32)
    state.particle_qd.assign(velocity)
    acceleration = np.array([1.0, 2.0, 3.0])
    state.particle_f.assign(model.particle_mass.numpy()[:, None] * acceleration)
    expected_v = velocity + dt * (gravity + acceleration)
    solver.step(state, out, None, None, dt)
    np.testing.assert_allclose(out.particle_qd.numpy(), expected_v, atol=3.0e-6)
    np.testing.assert_allclose(out.particle_q.numpy(), initial + dt * expected_v, atol=2.0e-7)
    np.testing.assert_array_equal(state.particle_q.numpy(), initial)
    np.testing.assert_array_equal(state.particle_qd.numpy(), velocity)

    # Reusing the original input must not advance the backend's previous output.
    solver.reset(state)
    solver.step(state, state, None, None, dt)
    np.testing.assert_allclose(state.particle_q.numpy(), out.particle_q.numpy(), atol=1.0e-7)
    state.clear_forces()
    q0, v0 = state.particle_q.numpy(), state.particle_qd.numpy()
    solver.step(state, out, None, None, dt)
    np.testing.assert_allclose(out.particle_qd.numpy(), v0 + gravity * dt, atol=4.0e-6)
    np.testing.assert_allclose(out.particle_q.numpy(), q0 + dt * (v0 + gravity * dt), atol=2.0e-7)

    model.gravity.zero_()
    solver.notify_model_changed(newton.ModelFlags.MODEL_PROPERTIES)
    solver.step(state, out, None, None, dt)
    np.testing.assert_allclose(out.particle_qd.numpy(), v0, atol=4.0e-6)
    with test.assertRaisesRegex(ValueError, "fixed dt"):
        solver.step(state, out, None, None, dt * 2)
    for other in wp.get_cuda_devices():
        if other != device:
            foreign = builder.finalize(device=other)
            foreign_solver = newton.solvers.SolverUIPC(foreign)
            with test.assertRaisesRegex(RuntimeError, "separate process"):
                foreign_solver.step(foreign.state(), foreign.state(), None, None, dt)
            foreign_solver.close()
            break


def test_pinned_elasticity(test, device):
    """Recover compressed tetrahedron volume while retaining a pinned vertex."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    _add_tet(builder, masses=(0.0, 0.1, 0.1, 0.1))
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverUIPC(model)
    state, out = model.state(), model.state()
    q = state.particle_q.numpy()
    q[3, 2] -= 0.04
    state.particle_q.assign(q)
    initial_volume = np.linalg.det((q[1:] - q[0]).T)
    for _ in range(8):
        solver.step(state, out, None, None, 0.01)
        state, out = out, state
    final = state.particle_q.numpy()
    np.testing.assert_array_equal(final[0], q[0])
    test.assertGreater(np.linalg.det((final[1:] - final[0]).T), initial_volume * 1.1)
    # Prescribed mounts must transmit motion through elasticity without
    # rigidly translating the whole solid.
    moved = final.copy()
    moved[0, 0] += 0.005
    state.particle_q.assign(moved)
    state.particle_qd.zero_()
    solver.step(state, out, None, None, 0.01)
    after = out.particle_q.numpy()
    np.testing.assert_array_equal(after[0], moved[0])
    test.assertGreater(np.linalg.norm(after[1:] - final[1:]), 1e-5)
    test.assertGreater(np.linalg.norm((after - final)[1:] - (after - final)[0]), 1e-4)


def test_worlds_and_ccd(test, device):
    """Prevent tunneling and isolate coincident worlds with local and global planes."""
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    for world in range(2):
        builder.begin_world()
        _add_tet(builder, height=0.25, masses=(0.01,) * 4)
        if world == 0:
            builder.add_ground_plane(height=0.2)
        builder.end_world()
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverUIPC(model)
    state, out = model.state(), model.state()
    state.particle_qd.assign(np.tile([0.0, 0.0, -10.0], (8, 1)))
    ids = model.tet_indices.numpy().reshape(-1, 4)
    worlds = model.particle_world.numpy()
    for _ in range(12):
        solver.step(state, out, None, None, 0.01)
        state, out = out, state
        q = state.particle_q.numpy()
        test.assertGreaterEqual(float(q[worlds == 0, 2].min()), 0.2 - 1.0e-6)
        test.assertGreaterEqual(float(q[:, 2].min()), -1.0e-6)
        test.assertTrue(np.all(np.linalg.det((q[ids[:, 1:]] - q[ids[:, :1]]).transpose(0, 2, 1)) > 0.0))
    test.assertLess(float(q[worlds == 1, 2].min()), 0.15)


def test_mesh_obstacles(test, device):
    """Collide with transformed box and mesh obstacles and honor prescribed body poses."""
    builder = newton.ModelBuilder()
    _add_tet(builder, height=0.35, masses=(0.01,) * 4)
    # The model pose intersects the tet; the first input state moves the
    # obstacle clear. Native sanity checking must use that input pose.
    body = builder.add_body(is_kinematic=True, xform=wp.transform((0.0, 0.0, 0.25), wp.quat_identity()))
    builder.add_shape_box(body, hx=0.5, hy=0.5, hz=0.05, xform=wp.transform((0.0, 0.0, 0.1), wp.quat_identity()))
    # A separate mesh exercises geometry conversion and static transform composition.
    mesh = newton.Mesh(
        vertices=np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], dtype=np.float32),
        indices=np.array([0, 1, 2, 0, 2, 3], dtype=np.int32),
    )
    builder.add_shape_mesh(-1, mesh=mesh, xform=wp.transform((0, 0, -0.2), wp.quat_identity()))
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverUIPC(model)
    state, out = model.state(), model.state()
    body_q = state.body_q.numpy()
    body_q[body, 2] = 0.1
    state.body_q.assign(body_q)
    state.particle_qd.assign(np.tile([0.0, 0.0, -4.0], (4, 1)))
    for _ in range(8):
        solver.step(state, out, None, None, 0.01)
        state, out = out, state
        test.assertGreaterEqual(float(state.particle_q.numpy()[:, 2].min()), 0.25 - 1.0e-6)
        np.testing.assert_array_equal(state.body_q.numpy(), body_q)


def test_invalid_inputs(test, device):
    """Reject unsupported material behavior and invalid step arguments before advancing."""
    builder = newton.ModelBuilder()
    _add_tet(builder)
    model = builder.finalize(device=device)
    materials = model.tet_materials.numpy()
    materials[:, 2] = 1.0
    model.tet_materials.assign(materials)
    with test.assertRaisesRegex(ValueError, "damping"):
        newton.solvers.SolverUIPC(model)
    materials[:, 2] = 0.0
    model.tet_materials.assign(materials)
    solver = newton.solvers.SolverUIPC(model)
    state, out = model.state(), model.state()
    for dt in (0.0, -0.01, float("nan"), float("inf")):
        with test.subTest(dt=dt), test.assertRaisesRegex(ValueError, "dt"):
            solver.step(state, out, None, None, dt)
    with test.assertRaisesRegex(ValueError, "contacts=None"):
        solver.step(state, out, None, newton.CollisionPipeline(model).contacts(), 0.01)
    control = model.control()
    control.tet_activations.fill_(0.1)
    with test.assertRaisesRegex(ValueError, "activation"):
        solver.step(state, out, control, None, 0.01)
    q = state.particle_q.numpy()
    q[[0, 1]] = q[[1, 0]]
    state.particle_q.assign(q)
    with test.assertRaisesRegex(ValueError, "non-inverted"):
        solver.step(state, out, None, None, 0.01)
    solver.close()
    with test.assertRaisesRegex(RuntimeError, "closed"):
        solver.step(state, out, None, None, 0.01)
    q = model.particle_q.numpy()
    q[3, 2] -= 0.04
    state.particle_q.assign(q)
    solver = newton.solvers.SolverUIPC(model, max_iterations=1)
    before = out.particle_q.numpy()
    with test.assertRaisesRegex(RuntimeError, "converge"):
        solver.step(state, out, None, None, 0.01)
    np.testing.assert_array_equal(out.particle_q.numpy(), before)
    solver.close()


@requires("pxr.Usd")
def test_usd_volume(test, device):
    """Simulate equivalent USD and builder-authored tetrahedral materials and masses."""
    stage = _deformable_stage()
    tet = _author_unit_tet(stage, "/Soft", sim_api=True)
    _apply_deformable_body_api(tet.GetPrim())
    _bind_deformable_material(stage, tet.GetPrim(), "/Material", youngsModulus=1.0e4, poissonsRatio=0.3, density=100.0)
    imported = newton.ModelBuilder()
    imported.add_usd(stage)
    model_usd = imported.finalize(device=device)
    builder = newton.ModelBuilder()
    for q, mass in zip(model_usd.particle_q.numpy(), model_usd.particle_mass.numpy(), strict=True):
        builder.add_particle(wp.vec3(*q), wp.vec3(0.0), float(mass), radius=0.0)
    mu = 1.0e4 / (2.0 * 1.3)
    lam = 1.0e4 * 0.3 / (1.3 * 0.4)
    builder.add_tetrahedron(0, 1, 2, 3, k_mu=mu, k_lambda=lam)
    model_builder = builder.finalize(device=device)
    results = []
    for model in (model_usd, model_builder):
        model.set_gravity((0.0, 0.0, 0.0))
        state, out = model.state(), model.state()
        q = state.particle_q.numpy()
        q[3, 2] -= 0.1
        state.particle_q.assign(q)
        solver = newton.solvers.SolverUIPC(model)
        solver.step(state, out, None, None, 0.01)
        results.append(out.particle_q.numpy())
        solver.close()
    np.testing.assert_allclose(results[0], results[1], atol=1.0e-6)


@requires("uipc")
class TestSolverUIPC(unittest.TestCase):
    pass


# Native libuipc caches require one GPU per process. Exercise other GPUs by
# launching this suite in separate processes with CUDA_VISIBLE_DEVICES set.
devices = get_selected_cuda_test_devices()[:1]
for name, function in (
    ("test_state_and_forces", test_state_and_forces),
    ("test_pinned_elasticity", test_pinned_elasticity),
    ("test_worlds_and_ccd", test_worlds_and_ccd),
    ("test_mesh_obstacles", test_mesh_obstacles),
    ("test_invalid_inputs", test_invalid_inputs),
    ("test_usd_volume", test_usd_volume),
):
    # libuipc emits native diagnostics outside Newton's logging system.
    add_function_test(TestSolverUIPC, name, function, devices=devices, check_output=False)


if __name__ == "__main__":
    unittest.main(verbosity=2)
