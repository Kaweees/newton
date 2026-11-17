# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Mixed-dimensional, lagged GMCP contact quadrature (IsaacIPC, Section 4)."""

import numpy as np
import warp as wp

_Polygon = wp.types.matrix(shape=(8, 3), dtype=wp.float32)


@wp.struct
class _Sample:
    slave: wp.vec3i
    master: wp.vec3i
    beta_slave: wp.vec3
    beta_master: wp.vec3
    normal: wp.vec3
    measure: float
    coefficient: float
    support: float


@wp.func
def _interpolate(x: wp.array[wp.vec3], ids: wp.vec3i, beta: wp.vec3):
    return beta[0] * x[ids[0]] + beta[1] * x[ids[1]] + beta[2] * x[ids[2]]


@wp.func
def _normal(x: wp.array[wp.vec3], ids: wp.vec3i):
    return wp.normalize(wp.cross(x[ids[1]] - x[ids[0]], x[ids[2]] - x[ids[0]]))


@wp.func
def _barycentric(p: wp.vec3, a: wp.vec3, b: wp.vec3, c: wp.vec3):
    u = b - a
    v = c - a
    d = p - a
    n = wp.cross(u, v)
    det = wp.length_sq(n)
    y = wp.dot(wp.cross(d, v), n) / det
    z = wp.dot(wp.cross(u, d), n) / det
    return wp.vec3(1.0 - y - z, y, z)


@wp.func
def _hermite(x: float, width: float):
    t = wp.clamp(x / width, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


@wp.func
def _emit(
    reference: wp.array[wp.vec3],
    slave: wp.vec3i,
    master: wp.vec3i,
    bs: wp.vec3,
    bm: wp.vec3,
    normal: wp.vec3,
    measure: float,
    gamma: float,
    stiffness: float,
    support_max: float,
    samples: wp.array[_Sample],
    index: int,
    error: wp.array[int],
):
    if measure <= 0.0 or gamma <= 0.0 or stiffness == 0.0:
        return
    reference_gap = wp.dot(
        _normal(reference, slave), _interpolate(reference, master, bm) - _interpolate(reference, slave, bs)
    )
    if reference_gap <= 0.0 or not wp.isfinite(reference_gap):
        wp.atomic_max(error, 0, 1)
        return
    s = _Sample()
    s.slave = slave
    s.master = master
    s.beta_slave = bs
    s.beta_master = bm
    s.normal = normal
    s.measure = measure
    s.coefficient = stiffness * measure * gamma
    s.support = wp.min(0.9 * reference_gap, support_max)
    samples[index] = s


@wp.kernel
def _check_positions(x: wp.array[wp.vec3], error: wp.array[int]):
    i = wp.tid()
    if not wp.isfinite(x[i]):
        wp.atomic_max(error, 0, 2)


@wp.kernel
def _sample_faces(
    x: wp.array[wp.vec3],
    reference: wp.array[wp.vec3],
    slave_faces: wp.array[wp.vec3i],
    master_faces: wp.array[wp.vec3i],
    pairs: wp.array[wp.vec2i],
    stiffness: float,
    width: float,
    support_max: float,
    samples: wp.array[_Sample],
    error: wp.array[int],
):
    pair = wp.tid()
    slave = slave_faces[pairs[pair][0]]
    master = master_faces[pairs[pair][1]]
    a = x[slave[0]]
    b = x[slave[1]]
    c = x[slave[2]]
    cross = wp.cross(b - a, c - a)
    if wp.length_sq(cross) <= 1.0e-24:
        wp.atomic_max(error, 0, 3)
        return
    n = wp.normalize(cross)
    # Columns map master barycentric coordinates to projected slave coordinates.
    transform = wp.matrix_from_cols(
        _barycentric(x[master[0]], a, b, c),
        _barycentric(x[master[1]], a, b, c),
        _barycentric(x[master[2]], a, b, c),
    )
    polygon = _Polygon()
    polygon[0] = wp.vec3(1.0, 0.0, 0.0)
    polygon[1] = wp.vec3(0.0, 1.0, 0.0)
    polygon[2] = wp.vec3(0.0, 0.0, 1.0)
    count = int(3)
    # Sutherland-Hodgman clipping in barycentric space. Intersection of two
    # triangles has at most six vertices; eight slots also cover intermediates.
    for axis in range(3):
        clipped = _Polygon()
        next_count = int(0)
        for j in range(count):
            p = polygon[(j + count - 1) % count]
            q = polygon[j]
            dp = (transform * p)[axis]
            dq = (transform * q)[axis]
            # A vertex exactly on the boundary is emitted by the inside branch.
            # Emitting it again as an intersection can overflow the six-vertex bound.
            if (dp < 0.0 and dq > 0.0) or (dp > 0.0 and dq < 0.0):
                clipped[next_count] = p + (dp / (dp - dq)) * (q - p)
                next_count += 1
            if dq >= 0.0:
                clipped[next_count] = q
                next_count += 1
        polygon = clipped
        count = next_count
    for j in range(1, count - 1):
        p = polygon[0]
        q = polygon[j]
        r = polygon[j + 1]
        u = _interpolate(x, master, q) - _interpolate(x, master, p)
        v = _interpolate(x, master, r) - _interpolate(x, master, p)
        measure = wp.abs(wp.dot(wp.cross(u, v), n)) / 6.0
        for k in range(3):
            bm = (p + q + r) / 6.0
            if k == 0:
                bm += 0.5 * p
            elif k == 1:
                bm += 0.5 * q
            else:
                bm += 0.5 * r
            gamma = _hermite(wp.min(bm[0], wp.min(bm[1], bm[2])), width)
            _emit(
                reference,
                slave,
                master,
                transform * bm,
                bm,
                n,
                measure,
                gamma,
                stiffness,
                support_max,
                samples,
                pair * 12 + (j - 1) * 3 + k,
                error,
            )


@wp.kernel
def _sample_edges(
    x: wp.array[wp.vec3],
    reference: wp.array[wp.vec3],
    slave_faces: wp.array[wp.vec3i],
    master_edges: wp.array[wp.vec2i],
    pairs: wp.array[wp.vec2i],
    stiffness: float,
    width: float,
    support_max: float,
    offset: int,
    samples: wp.array[_Sample],
    error: wp.array[int],
):
    pair = wp.tid()
    slave = slave_faces[pairs[pair][0]]
    edge = master_edges[pairs[pair][1]]
    master = wp.vec3i(edge[0], edge[1], edge[1])
    a = x[slave[0]]
    b = x[slave[1]]
    c = x[slave[2]]
    cross = wp.cross(b - a, c - a)
    if wp.length_sq(cross) <= 1.0e-24:
        wp.atomic_max(error, 0, 3)
        return
    n = wp.normalize(cross)
    bs0 = _barycentric(x[edge[0]], a, b, c)
    bs1 = _barycentric(x[edge[1]], a, b, c)
    lo = float(0.0)
    hi = float(1.0)
    for axis in range(3):
        slope = bs1[axis] - bs0[axis]
        if slope > 0.0:
            lo = wp.max(lo, -bs0[axis] / slope)
        elif slope < 0.0:
            hi = wp.min(hi, -bs0[axis] / slope)
        elif bs0[axis] < 0.0:
            return
    if lo >= hi:
        return
    delta = x[edge[1]] - x[edge[0]]
    measure = 0.5 * (hi - lo) * wp.length(delta - wp.dot(delta, n) * n)
    for k in range(2):
        eta = 0.5 * (lo + hi) + (float(k) * 2.0 - 1.0) * (hi - lo) * 0.2886751345948129
        gamma = _hermite(eta, width) * _hermite(1.0 - eta, width)
        _emit(
            reference,
            slave,
            master,
            bs0 + eta * (bs1 - bs0),
            wp.vec3(1.0 - eta, eta, 0.0),
            n,
            measure,
            gamma,
            stiffness,
            support_max,
            samples,
            offset + pair * 2 + k,
            error,
        )


@wp.kernel
def _sample_points(
    x: wp.array[wp.vec3],
    reference: wp.array[wp.vec3],
    slave_faces: wp.array[wp.vec3i],
    master_points: wp.array[int],
    pairs: wp.array[wp.vec2i],
    stiffness: float,
    support_max: float,
    offset: int,
    samples: wp.array[_Sample],
    error: wp.array[int],
):
    pair = wp.tid()
    slave = slave_faces[pairs[pair][0]]
    point = master_points[pairs[pair][1]]
    a = x[slave[0]]
    b = x[slave[1]]
    c = x[slave[2]]
    cross = wp.cross(b - a, c - a)
    if wp.length_sq(cross) <= 1.0e-24:
        wp.atomic_max(error, 0, 3)
        return
    bs = _barycentric(x[point], a, b, c)
    if wp.min(bs[0], wp.min(bs[1], bs[2])) < 0.0:
        return
    _emit(
        reference,
        slave,
        wp.vec3i(point),
        bs,
        wp.vec3(1.0, 0.0, 0.0),
        wp.normalize(cross),
        1.0,
        1.0,
        stiffness,
        support_max,
        samples,
        offset + pair,
        error,
    )


@wp.kernel
def _evaluate(
    x: wp.array[wp.vec3],
    samples: wp.array[_Sample],
    energy: wp.array[float],
    forces: wp.array[wp.vec3],
    curvature: wp.array[float],
    sample_force: wp.array[float],
    sample_measure: wp.array[float],
    sample_gap: wp.array[float],
    error: wp.array[int],
):
    i = wp.tid()
    s = samples[i]
    curvature[i] = 0.0
    sample_force[i] = 0.0
    sample_measure[i] = s.measure
    sample_gap[i] = 0.0
    if s.coefficient == 0.0:
        return
    gap = wp.dot(s.normal, _interpolate(x, s.master, s.beta_master) - _interpolate(x, s.slave, s.beta_slave))
    sample_gap[i] = gap
    if gap <= 0.0 or not wp.isfinite(gap):
        wp.atomic_max(error, 0, 4)
        return
    if gap >= s.support:
        return
    # IPC barrier B = -(g-epsilon)^2 log(g/epsilon), with lagged sample data.
    d = gap - s.support
    log_ratio = wp.log(gap / s.support)
    first = -2.0 * d * log_ratio - d * d / gap
    second = -2.0 * log_ratio - 4.0 * d / gap + d * d / (gap * gap)
    wp.atomic_add(energy, 0, -s.coefficient * d * d * log_ratio)
    curvature[i] = s.coefficient * wp.max(0.0, second)
    sample_force[i] = -s.coefficient * first
    f = s.coefficient * first * s.normal
    for j in range(3):
        wp.atomic_add(forces, s.slave[j], s.beta_slave[j] * f)
        wp.atomic_sub(forces, s.master[j], s.beta_master[j] * f)


@wp.kernel
def _hessian_product(
    direction: wp.array[wp.vec3], samples: wp.array[_Sample], curvature: wp.array[float], out: wp.array[wp.vec3]
):
    i = wp.tid()
    s = samples[i]
    if curvature[i] == 0.0:
        return
    dg = wp.dot(
        s.normal, _interpolate(direction, s.master, s.beta_master) - _interpolate(direction, s.slave, s.beta_slave)
    )
    f = curvature[i] * dg * s.normal
    for j in range(3):
        wp.atomic_sub(out, s.slave[j], s.beta_slave[j] * f)
        wp.atomic_add(out, s.master[j], s.beta_master[j] * f)


@wp.kernel
def _step_size(
    x: wp.array[wp.vec3],
    direction: wp.array[wp.vec3],
    samples: wp.array[_Sample],
    alpha: wp.array[float],
    error: wp.array[int],
):
    s = samples[wp.tid()]
    if s.coefficient == 0.0:
        return
    gap = wp.dot(s.normal, _interpolate(x, s.master, s.beta_master) - _interpolate(x, s.slave, s.beta_slave))
    dg = wp.dot(
        s.normal, _interpolate(direction, s.master, s.beta_master) - _interpolate(direction, s.slave, s.beta_slave)
    )
    if gap <= 0.0 or not wp.isfinite(gap) or not wp.isfinite(dg):
        wp.atomic_max(error, 0, 4)
    elif dg < 0.0:
        wp.atomic_min(alpha, 0, 0.9 * gap / (-dg))


@wp.kernel
def _add_forces(source: wp.array[wp.vec3], target: wp.array[wp.vec3]):
    i = wp.tid()
    target[i] += source[i]


class ContactModelGMCP:
    """Geometric Mortar Contact Potential (GMCP) evaluated in Warp.

    .. experimental::
        This class implements IsaacIPC's normal contact quadrature, potential,
        lagged derivatives, and linearized gap bound. The API may change without
        a deprecation period. It is a contact component, not a complete IPC
        solver: broad phase, feature ownership, friction, nonlinear integration,
        and continuous collision detection are the caller's responsibility.

    All feature indices address the same vertex array, which can be Newton's
    particle positions. Slave triangles must face their master features. Supply
    candidate pairs explicitly; repeated pairs and overlapping slave triangles
    double-count their contributions. Boundary points/edges must have a unique
    slave owner. Call :meth:`update` to rebuild quadrature, then :meth:`evaluate`
    for forces and energy. Samples remain frozen until the next update, including
    barycentric coordinates, normals, measures, mollifiers, and barrier support.

    The reference configuration is copied at construction. Equation 17 is
    interpreted using those reference positions at each sample's barycentric
    coordinates. It must have positive signed gaps for every generated sample.
    Reference positions are never replaced by the current iterate implicitly.

    Args:
        reference_positions: Reference vertex positions [m], shape (N,), dtype vec3.
        slave_triangles: Oriented slave vertex indices, shape (S, 3).
        master_triangles: Master face vertex indices, shape (F, 3).
        face_pairs: (slave triangle, master triangle) indices, shape (P, 2).
        master_edges: Master edge vertex indices, shape (E, 2).
        edge_pairs: (slave triangle, master edge) indices, shape (Q, 2).
        master_points: Master vertex indices, shape (V,).
        point_pairs: (slave triangle, master point entry) indices, shape (R, 2).
        stiffness_face: Area-potential stiffness [N/m^3].
        stiffness_edge: Line-potential stiffness [N/m^2].
        stiffness_point: Point-potential stiffness [N/m].
        support_max: Upper bound on barrier support distance [m].
        mollifier_face: Face Hermite transition width in master barycentric coordinates.
        mollifier_edge: Edge Hermite transition width in master edge coordinates.

    Note:
        Methods synchronize to report invalid geometry. CUDA graph capture and
        differentiating through clipping with Warp autodiff are not supported.
        USD schema support is not applicable to this runtime array operation;
        callers can supply geometry loaded through Newton's existing importers.
    """

    def __init__(
        self,
        reference_positions: wp.array[wp.vec3],
        slave_triangles: np.ndarray,
        *,
        master_triangles: np.ndarray | None = None,
        face_pairs: np.ndarray | None = None,
        master_edges: np.ndarray | None = None,
        edge_pairs: np.ndarray | None = None,
        master_points: np.ndarray | None = None,
        point_pairs: np.ndarray | None = None,
        stiffness_face: float = 1.0e6,
        stiffness_edge: float = 1.0e3,
        stiffness_point: float = 1.0,
        support_max: float = 0.001,
        mollifier_face: float = 0.1,
        mollifier_edge: float = 0.1,
    ):
        if (
            not isinstance(reference_positions, wp.array)
            or reference_positions.dtype != wp.vec3
            or reference_positions.ndim != 1
        ):
            raise ValueError("reference_positions must be a one-dimensional Warp vec3 array")
        self.device = reference_positions.device
        self.vertex_count = len(reference_positions)
        ref = reference_positions.numpy()
        if not np.isfinite(ref).all():
            raise ValueError("reference_positions must be finite")
        for name, value in (
            ("stiffness_face", stiffness_face),
            ("stiffness_edge", stiffness_edge),
            ("stiffness_point", stiffness_point),
        ):
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not np.isfinite(support_max) or support_max <= 0.0:
            raise ValueError("support_max must be finite and positive")
        for name, value in (("mollifier_face", mollifier_face), ("mollifier_edge", mollifier_edge)):
            if not np.isfinite(value) or not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be in (0, 1]")

        def indices(data, width, bound, name):
            shape = (0,) if width == 1 else (0, width)
            raw = np.asarray(data) if data is not None else np.empty(shape, dtype=np.int32)
            if raw.ndim != len(shape) or (width > 1 and raw.shape[1] != width):
                raise ValueError(f"{name} has invalid shape")
            if raw.size and (not np.issubdtype(raw.dtype, np.integer) or np.any(raw < 0) or np.any(raw >= bound)):
                raise ValueError(f"{name} contains invalid indices")
            return raw.astype(np.int32)

        slave = indices(slave_triangles, 3, self.vertex_count, "slave_triangles")
        if len(slave) and np.any(
            np.linalg.norm(np.cross(ref[slave[:, 1]] - ref[slave[:, 0]], ref[slave[:, 2]] - ref[slave[:, 0]]), axis=1)
            <= 1.0e-12
        ):
            raise ValueError("reference slave triangles must be nondegenerate")
        self._slave = wp.array(slave, dtype=wp.vec3i, device=self.device)
        self._features = []
        self._pairs = []
        for data, pairs, width, name in (
            (master_triangles, face_pairs, 3, "face"),
            (master_edges, edge_pairs, 2, "edge"),
            (master_points, point_pairs, 1, "point"),
        ):
            feature = indices(data, width, self.vertex_count, f"master_{name}")
            pair = indices(pairs, 2, np.array([len(slave), len(feature)]), f"{name}_pairs")
            self._features.append(
                wp.array(feature, dtype={1: wp.int32, 2: wp.vec2i, 3: wp.vec3i}[width], device=self.device)
            )
            self._pairs.append(wp.array(pair, dtype=wp.vec2i, device=self.device))
        self._reference = wp.clone(reference_positions)
        self._stiffness = (float(stiffness_face), float(stiffness_edge), float(stiffness_point))
        self._widths = (float(mollifier_face), float(mollifier_edge))
        self._support_max = float(support_max)
        self.sample_capacity = 12 * len(self._pairs[0]) + 2 * len(self._pairs[1]) + len(self._pairs[2])
        """Maximum quadrature sample count; inactive slots have zero force and measure."""
        self._samples = wp.zeros(self.sample_capacity, dtype=_Sample, device=self.device)
        self.forces = wp.zeros(self.vertex_count, dtype=wp.vec3, device=self.device)
        """Contact forces [N] from the last successful evaluation, shape (N,)."""
        self.sample_force = wp.zeros(self.sample_capacity, dtype=float, device=self.device)
        """Integrated compressive force magnitude [N] per quadrature slot."""
        self.sample_measure = wp.zeros(self.sample_capacity, dtype=float, device=self.device)
        """Quadrature measure from the last evaluation: area [m²], length [m], or 1.

        Slots are packed as twelve per face pair, two per edge pair, then one
        per point pair. Unused slots have zero measure. Face pressure [Pa] is
        ``sample_force / sample_measure`` on nonzero face slots.
        """
        self.sample_gap = wp.zeros(self.sample_capacity, dtype=float, device=self.device)
        """Signed normal gap [m] at each evaluated sample; zero in unused slots."""
        self._curvature = wp.zeros(self.sample_capacity, dtype=float, device=self.device)
        self._energy = wp.zeros(1, dtype=float, device=self.device)
        self._alpha = wp.ones(1, dtype=float, device=self.device)
        self._error = wp.zeros(1, dtype=int, device=self.device)
        self._updated = False
        self._evaluated = False

    def _validate_array(self, array, name):
        if (
            not isinstance(array, wp.array)
            or array.dtype != wp.vec3
            or array.shape != (self.vertex_count,)
            or array.device != self.device
        ):
            raise ValueError(f"{name} must be a vec3 array of length {self.vertex_count} on {self.device}")

    def _check_error(self):
        code = int(self._error.numpy()[0])
        if code:
            messages = {
                1: "GMCP reference gaps must be positive for all generated samples",
                2: "GMCP vertex positions must be finite",
                3: "GMCP slave triangles must be nondegenerate",
                4: "GMCP current gaps must be finite and positive (infeasible contact)",
            }
            raise ValueError(messages[code])

    def update(self, positions: wp.array[wp.vec3]) -> None:
        """Rebuild and freeze quadrature at ``positions`` [m].

        Faces use a three-point degree-two rule on each clipped fan triangle;
        edges use two-point Gauss-Legendre quadrature. Degenerate projected master
        faces/edges contribute zero measure. Degenerate slave faces are errors.
        A failed update invalidates the previous quadrature and evaluation.
        """
        self._validate_array(positions, "positions")
        self._updated = self._evaluated = False
        self._samples.zero_()
        self._error.zero_()
        wp.launch(_check_positions, self.vertex_count, [positions, self._error], device=self.device)
        self._check_error()
        offset = 0
        for i, (kernel, capacity) in enumerate(((_sample_faces, 12), (_sample_edges, 2), (_sample_points, 1))):
            args = [positions, self._reference, self._slave, self._features[i], self._pairs[i], self._stiffness[i]]
            if i < 2:
                args.append(self._widths[i])
            args.append(self._support_max)
            if i > 0:
                args.append(offset)
            if len(self._pairs[i]):
                wp.launch(kernel, len(self._pairs[i]), [*args, self._samples, self._error], device=self.device)
            offset += capacity * len(self._pairs[i])
        self._check_error()
        self._updated = True

    def evaluate(self, positions: wp.array[wp.vec3], *, forces: wp.array[wp.vec3] | None = None) -> float:
        """Return contact energy [J] and compute forces and lagged PSD curvature.

        If ``forces`` is provided, add the result to that buffer (for example
        ``state.particle_f``) after validating feasibility. The owned
        :attr:`forces` buffer is overwritten on every call. Outputs from a failed
        evaluation must not be used; an external force buffer is left unchanged.
        """
        self._validate_array(positions, "positions")
        if forces is not None:
            self._validate_array(forces, "forces")
            if forces.ptr == self.forces.ptr or forces.ptr == positions.ptr:
                raise ValueError("external forces must not alias positions or the owned force buffer")
        if not self._updated:
            raise RuntimeError("Call update() before evaluating GMCP")
        self._evaluated = False
        self.forces.zero_()
        self._energy.zero_()
        self._error.zero_()
        wp.launch(_check_positions, self.vertex_count, [positions, self._error], device=self.device)
        wp.launch(
            _evaluate,
            self.sample_capacity,
            [
                positions,
                self._samples,
                self._energy,
                self.forces,
                self._curvature,
                self.sample_force,
                self.sample_measure,
                self.sample_gap,
                self._error,
            ],
            device=self.device,
        )
        self._check_error()
        if forces is not None:
            wp.launch(_add_forces, self.vertex_count, [self.forces, forces], device=self.device)
        self._evaluated = True
        return float(self._energy.numpy()[0])

    def apply_hessian(
        self, direction: wp.array[wp.vec3], *, output: wp.array[wp.vec3] | None = None
    ) -> wp.array[wp.vec3]:
        """Apply the last evaluated lagged PSD Hessian to a displacement [m].

        Return a vector [N]. If provided, ``output`` is overwritten and cannot
        alias ``direction``. Geometry derivatives excluded by the paper's
        approximate Hessian are deliberately not differentiated.
        """
        self._validate_array(direction, "direction")
        if not self._evaluated:
            raise RuntimeError("Call evaluate() before applying the GMCP Hessian")
        if output is None:
            output = wp.empty_like(direction)
        self._validate_array(output, "output")
        if output.ptr == direction.ptr:
            raise ValueError("Hessian output must not alias direction")
        output.zero_()
        wp.launch(
            _hessian_product,
            self.sample_capacity,
            [direction, self._samples, self._curvature, output],
            device=self.device,
        )
        return output

    def step_size(self, positions: wp.array[wp.vec3], direction: wp.array[wp.vec3]) -> float:
        """Return a step in [0, 1] preserving 10% of each frozen positive gap.

        ``direction`` is a displacement [m]. This is Equation 18's linearized
        bound, not CCD: new contacts and rotating/resampled normals require
        additional collision checks in the integrating solver.
        """
        self._validate_array(positions, "positions")
        self._validate_array(direction, "direction")
        if not self._updated:
            raise RuntimeError("Call update() before computing the GMCP step bound")
        self._alpha.fill_(1.0)
        self._error.zero_()
        wp.launch(
            _step_size,
            self.sample_capacity,
            [positions, direction, self._samples, self._alpha, self._error],
            device=self.device,
        )
        self._check_error()
        return float(self._alpha.numpy()[0])
