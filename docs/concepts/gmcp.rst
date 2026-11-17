.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

Geometric Mortar Contact Potential
==================================

:class:`~newton.geometry.ContactModelGMCP` implements Section 4 of
`IsaacIPC <https://arxiv.org/abs/2605.24339>`__ in Warp, on CPU and CUDA.
It constructs contact quadrature between oriented slave triangles and master
faces, edges, and points, and evaluates normal barrier forces. This runtime
component can add forces directly to a Newton particle state.

.. experimental::

    ``ContactModelGMCP`` is experimental. It supplies contact quadrature and
    lagged derivatives, not a complete IPC integrator. Applications supply
    candidate pairs, feature ownership, integration, and collision checks.
    There is no friction model, rigid-body force mapping, or automatic solver coupling.

Discretization
--------------

All indices refer to one shared vertex array. Slave triangle winding sets the
outward normal; master features must lie in front of that surface. Candidate
pairs associate a slave triangle with a master feature. Their projected overlap
is clipped in slave barycentric space:

* Face pairs use Sutherland--Hodgman clipping, fan triangulation, and a
  three-point degree-two quadrature on each fan triangle (up to twelve samples).
* Edge pairs use interval clipping and two-point Gauss--Legendre quadrature.
* Point pairs contribute one sample when their projection is inside the slave.

Face weights have units of area, edge weights of length, and point weights are
dimensionless. A zero projected measure contributes nothing. Nonfinite vertices,
degenerate slave faces, nonpositive reference gaps, and infeasible current gaps
raise errors. Candidate pairs must not be duplicated. A point on a shared slave
boundary needs a single owner; so does an edge lying along that boundary.
Feature extraction, ownership, world isolation, and broad phase are caller-owned.

For a sample, the signed gap is

.. math::

    g = \mathbf{n}_s \cdot (\mathbf{x}_m - \mathbf{x}_s).

The mixed-dimensional energy is the paper's Equation 10:

.. math::

    \Psi = \sum_{q=0}^{2}\sum_k \kappa_q w_k \gamma_k B(g_k,\epsilon_k).

The barrier is :math:`B(g,\epsilon)=-(g-\epsilon)^2\log(g/\epsilon)` for
:math:`0<g<\epsilon`, and zero for :math:`g\geq\epsilon`, following
`Incremental Potential Contact <https://ipc-sim.github.io/file/IPC-paper-350ppi.pdf>`__.
The stiffness units are N/m³ for faces, N/m² for edges, and N/m for points.
Face contributions divided by quadrature area give normal pressure in Pa;
line and point contributions cannot be interpreted as pressure without a
chosen spatial reconstruction.

The cubic Hermite ramp is :math:`H(t)=3t^2-2t^3` on [0,1], clamped outside.
Face weights apply it to the minimum master barycentric coordinate divided by
``mollifier_face``. Edge weights multiply ramps at both ends of the master
edge using ``mollifier_edge``. Point weights are one (Equations 13--16).

Reference geometry and derivatives
----------------------------------

Pass a penetration-free reference configuration at construction. It is copied
and remains fixed. At each quadrature update, the reference gap is evaluated
using the sample's barycentric coordinates and the reference slave normal.
The support is :math:`\epsilon=\min(0.9g_{ref},\epsilon_{max})` (Equation 17).
This is an explicit reference-lifecycle interpretation of the paper; resetting
the reference requires a new contact model. Replacing the reference gap with
the current gap on every evaluation would suppress all contact forces.

``update(positions)`` rebuilds samples. Until the next update, barycentric
coordinates, normals, weights, mollifiers, and supports are frozen.
``evaluate(positions)`` returns energy and overwrites the model's ``forces``
and ``sample_force`` buffers. ``sample_gap`` and ``sample_measure`` expose the
evaluated gaps and integration measures; divide face sample forces by nonzero
face measures to recover pressure in Pa. Its optional ``forces`` argument accumulates the
result into an external array after successful feasibility validation.

The gradient is Equation 11 with these lagged samples. ``apply_hessian``
applies Equation 12's positive-semidefinite approximation
:math:`\sum\kappa w\gamma B''\nabla g\nabla g^T`, without assembling a dense
matrix. Clipping and changes in sample geometry are not differentiated.
These are derivatives of the frozen-sample energy, not of the entire
resampling operation. Warp autodiff and CUDA graph capture are unsupported;
the host methods synchronize to report invalid geometry.

``step_size`` returns the minimum of one and Equation 18's fraction-to-boundary
bound, preserving 10% of each frozen positive gap. This bound covers the current
linearized samples only. It does not detect new contacts or guarantee feasibility
after normals rotate or samples are rebuilt. A general IPC solver must also
handle those events, tetrahedral inversion, nonlinear convergence, and CCD.

Particle-state usage
--------------------

.. code-block:: python

    import numpy as np
    import warp as wp
    from newton.geometry import ContactModelGMCP

    reference = wp.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0.2, 0.2, 0.002]],
        dtype=wp.vec3,
    )
    contact = ContactModelGMCP(
        reference,
        np.array([[0, 1, 2]]),
        master_points=np.array([3]),
        point_pairs=np.array([[0, 0]]),
        stiffness_point=100.0,
        support_max=0.001,
    )
    positions = wp.clone(reference)
    positions.assign([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0.2, 0.2, 0.0005]])
    contact.update(positions)
    energy = contact.evaluate(positions)
    # For a matching Newton particle array:
    # state.clear_forces()
    # contact.evaluate(state.particle_q, forces=state.particle_f)
    # solver.step(...)
    # Validate/limit the proposed step before accepting it.

USD schema disposition: not applicable. This is a runtime array operation;
geometry may come from existing Newton USD mesh/deformable importers. It adds
no model material attribute or importer-specific authoring schema.

Validation
----------

``newton.tests.test_gmcp`` checks mixed-dimensional quadrature against known
measures, six-vertex clipping, Hermite weights, support adaptation, force
conservation, finite-difference gradients and Hessian products, translation
null modes, feasible steps, and invalid states on CPU and CUDA.
These are local mathematical tests, not reproductions of the paper's Hertz or
full finite-element contact-patch benchmarks.
