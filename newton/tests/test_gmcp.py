# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Verify GMCP quadrature and lagged derivatives independently of a time integrator."""

import unittest

import numpy as np
import warp as wp

from newton.geometry import ContactModelGMCP
from newton.tests.unittest_utils import add_function_test, get_test_devices


def _array(x, device):
    return wp.array(np.asarray(x, dtype=np.float32), dtype=wp.vec3, device=device)


def _barrier(gap, support):
    d = gap - support
    return -d * d * np.log(gap / support), -2 * d * np.log(gap / support) - d * d / gap


def test_quadrature(test, device):
    """Integrate known overlap areas, clipped lengths, and point measures."""
    reference = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [-1, -1, 0.2],
            [3, -1, 0.2],
            [-1, 3, 0.2],
            [-1, 0.25, 0.2],
            [2, 0.25, 0.2],
            [0.2, 0.3, 0.2],
        ],
        dtype=np.float32,
    )
    x = reference.copy()
    x[3:, 2] = 0.05
    positions = _array(x, device)
    contact = ContactModelGMCP(
        _array(reference, device),
        np.array([[0, 1, 2]]),
        master_triangles=np.array([[3, 4, 5]]),
        face_pairs=np.array([[0, 0]]),
        master_edges=np.array([[6, 7]]),
        edge_pairs=np.array([[0, 0]]),
        master_points=np.array([8]),
        point_pairs=np.array([[0, 0]]),
        stiffness_face=2.0,
        stiffness_edge=3.0,
        stiffness_point=5.0,
        mollifier_face=0.001,
        mollifier_edge=0.001,
        support_max=0.1,
    )
    contact.update(positions)
    energy = contact.evaluate(positions)
    barrier, derivative = _barrier(0.05, 0.1)
    coefficient = 2 * 0.5 + 3 * 0.75 + 5
    test.assertAlmostEqual(energy, coefficient * barrier, places=7)
    force = contact.forces.numpy()
    np.testing.assert_allclose(force.sum(axis=0), 0, atol=2e-7)
    test.assertAlmostEqual(force[3:, 2].sum(), -coefficient * derivative, places=6)
    test.assertAlmostEqual(contact.sample_force.numpy().sum(), -coefficient * derivative, places=6)
    measures = contact.sample_measure.numpy()
    np.testing.assert_allclose([measures[:12].sum(), measures[12:14].sum(), measures[14]], [0.5, 0.75, 1.0], atol=2e-7)
    np.testing.assert_allclose(contact.sample_gap.numpy()[measures > 0], 0.05, atol=1e-7)
    # The reference gap controls support, so the reference state has no prestress.
    test.assertEqual(contact.evaluate(_array(reference, device)), 0)
    # Moving the features out of the slave projection discards stale samples.
    x[3:, 0] += 10
    contact.update(_array(x, device))
    test.assertEqual(contact.evaluate(_array(x, device)), 0)


def test_hexagon_and_mollifiers(test, device):
    """Exercise six-vertex clipping and Hermite partition weights."""
    reference = np.array(
        [[0, 0, 0], [2, 0, 0], [1, 2, 0], [0, 1.5, 0.2], [2, 1.5, 0.2], [1, -0.5, 0.2]],
        dtype=np.float32,
    )
    x = reference.copy()
    x[3:, 2] = 0.05
    contact = ContactModelGMCP(
        _array(reference, device),
        np.array([[0, 1, 2]]),
        master_triangles=np.array([[3, 4, 5]]),
        face_pairs=np.array([[0, 0]]),
        stiffness_face=1,
        mollifier_face=0.0001,
        support_max=0.1,
    )
    contact.update(_array(x, device))
    barrier, _ = _barrier(0.05, 0.1)
    # Intersection is a hexagon: area 2 minus a top area 1/8 and two areas 9/32.
    test.assertAlmostEqual(contact.evaluate(_array(x, device)) / barrier, 1.3125, places=5)
    test.assertEqual(np.count_nonzero(contact.sample_force.numpy()), 12)
    # Identical triangles with quadrature min barycentric coordinate 1/6.
    reference[3:, :2] = reference[:3, :2]
    x[3:, :2] = x[:3, :2]
    contact = ContactModelGMCP(
        _array(reference, device),
        np.array([[0, 1, 2]]),
        master_triangles=np.array([[3, 4, 5]]),
        face_pairs=np.array([[0, 0]]),
        stiffness_face=1,
        mollifier_face=1 / 3,
        support_max=0.1,
    )
    contact.update(_array(x, device))
    test.assertAlmostEqual(contact.evaluate(_array(x, device)), 2 * 0.5 * barrier, places=7)
    # Edge quadrature at eta=(1 +/- 1/sqrt(3))/2 exercises both Hermite ramps.
    reference = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0.1, 0.1, 0.2], [0.6, 0.1, 0.2]])
    x = reference.copy()
    x[3:, 2] = 0.05
    contact = ContactModelGMCP(
        _array(reference, device),
        np.array([[0, 1, 2]]),
        master_edges=np.array([[3, 4]]),
        edge_pairs=np.array([[0, 0]]),
        stiffness_edge=1,
        mollifier_edge=1,
        support_max=0.1,
    )
    contact.update(_array(x, device))
    eta = (1 - 1 / np.sqrt(3)) / 2

    def h(t):
        return t * t * (3 - 2 * t)

    test.assertAlmostEqual(contact.evaluate(_array(x, device)), 0.5 * h(eta) * h(1 - eta) * barrier, places=8)


def test_derivatives_and_step_bound(test, device):
    """Check frozen-sample energy gradients, Hessian products, and feasible steps."""
    reference = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0.1, 0.1, 0.2], [0.7, 0.1, 0.2], [0.1, 0.7, 0.2]],
        dtype=np.float32,
    )
    x = reference.copy()
    x[3:, 2] = [0.03, 0.05, 0.07]
    positions = _array(x, device)
    contact = ContactModelGMCP(
        _array(reference, device),
        np.array([[0, 1, 2]]),
        master_triangles=np.array([[3, 4, 5]]),
        face_pairs=np.array([[0, 0]]),
        master_edges=np.array([[3, 4]]),
        edge_pairs=np.array([[0, 0]]),
        master_points=np.array([5]),
        point_pairs=np.array([[0, 0]]),
        stiffness_face=2,
        stiffness_edge=3,
        stiffness_point=1,
        support_max=0.1,
    )
    contact.update(positions)
    contact.evaluate(positions)
    force = contact.forces.numpy().copy()
    direction = np.random.default_rng(21).normal(size=x.shape).astype(np.float32) * 0.1
    hessian = contact.apply_hessian(_array(direction, device)).numpy()
    step = 1e-4
    plus = contact.evaluate(_array(x + step * direction, device))
    force_plus = contact.forces.numpy().copy()
    minus = contact.evaluate(_array(x - step * direction, device))
    force_minus = contact.forces.numpy().copy()
    test.assertAlmostEqual((plus - minus) / (2 * step), -np.sum(force * direction), delta=2e-5)
    np.testing.assert_allclose(hessian, -(force_plus - force_minus) / (2 * step), rtol=0.003, atol=0.0004)
    test.assertGreaterEqual(np.sum(direction * hessian), -1e-8)
    np.testing.assert_allclose(contact.apply_hessian(_array(np.ones_like(x), device)).numpy(), 0, atol=1e-6)
    displacement = np.zeros_like(x)
    displacement[3:, 2] = -1
    alpha = contact.step_size(positions, _array(displacement, device))
    test.assertGreater(alpha, 0)
    test.assertLess(alpha, 0.1)
    test.assertTrue(np.isfinite(contact.evaluate(_array(x + alpha * displacement, device))))
    with test.assertRaisesRegex(ValueError, "infeasible"):
        contact.evaluate(_array(x + 1.2 * alpha * displacement, device))
    displacement *= -1
    test.assertEqual(contact.step_size(positions, _array(displacement, device)), 1)
    external = _array(np.ones_like(x), device)
    contact.evaluate(positions, forces=external)
    np.testing.assert_allclose(external.numpy(), 1 + force, atol=1e-6)


def test_invalid_geometry(test, device):
    """Reject infeasible reference/current states without publishing external forces."""
    reference = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0.2, 0.2, 0.01]], dtype=np.float32)
    contact = ContactModelGMCP(
        _array(reference, device),
        np.array([[0, 1, 2]]),
        master_points=np.array([3]),
        point_pairs=np.array([[0, 0]]),
        support_max=1.0,
    )
    x = reference.copy()
    x[3, 2] = 0.008
    contact.update(_array(x, device))
    test.assertAlmostEqual(contact.evaluate(_array(x, device)), _barrier(0.008, 0.009)[0], places=10)
    external = _array(np.ones_like(x), device)
    x[3, 2] = 0
    with test.assertRaisesRegex(ValueError, "infeasible"):
        contact.evaluate(_array(x, device), forces=external)
    np.testing.assert_array_equal(external.numpy(), 1)
    with test.assertRaisesRegex(RuntimeError, "evaluate"):
        contact.apply_hessian(external)
    x[1] = x[0]
    with test.assertRaisesRegex(ValueError, "nondegenerate"):
        contact.update(_array(x, device))
    with test.assertRaisesRegex(RuntimeError, "update"):
        contact.evaluate(_array(reference, device))
    x = reference.copy()
    x[0, 0] = np.nan
    with test.assertRaisesRegex(ValueError, "finite"):
        contact.update(_array(x, device))
    reference[3, 2] = -0.01
    contact = ContactModelGMCP(
        _array(reference, device),
        np.array([[0, 1, 2]]),
        master_points=np.array([3]),
        point_pairs=np.array([[0, 0]]),
    )
    with test.assertRaisesRegex(ValueError, "reference gaps"):
        contact.update(_array(reference, device))
    with test.assertRaisesRegex(ValueError, "indices"):
        ContactModelGMCP(_array(reference, device), np.array([[0, 1, 4]]))


class TestGMCP(unittest.TestCase):
    pass


for test in (test_quadrature, test_hexagon_and_mollifiers, test_derivatives_and_step_bound, test_invalid_geometry):
    add_function_test(TestGMCP, test.__name__, test, devices=get_test_devices())


if __name__ == "__main__":
    unittest.main(verbosity=2)
