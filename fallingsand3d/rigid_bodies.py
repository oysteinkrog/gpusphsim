"""Rigid body / SDF object management for GPU SPH simulation.

Defines:
  - SDFObject numpy dtype and upload functions for constant memory
  - RigidBody numpy dtype and RigidBodyManager for Akinci boundary coupling

SDFObject: constant memory, read-only (kinematic collision objects).
RigidBody: global memory, read-write (dynamic bodies with force accumulators).
"""

from __future__ import annotations

import numpy as np
import cupy

# ---------------------------------------------------------------------------
# SDFObject dtype -- matches struct SDFObject in common.cuh (80 bytes)
#
# 5 float4s, each stored as 4 x float32:
#   pos_and_type:  xyz=center, w=__int_as_float(type)
#   size_and_r:    xyz=half_extents/radius/height, w=restitution
#   quat:          orientation quaternion (x,y,z,w), w is scalar part
#   velocity:      xyz=linear velocity, w=angular_speed
#   angular_axis:  xyz=rotation axis, w=friction
# ---------------------------------------------------------------------------

SDF_OBJECT_DTYPE = np.dtype([
    ("pos_and_type", np.float32, (4,)),
    ("size_and_r", np.float32, (4,)),
    ("quat", np.float32, (4,)),
    ("velocity", np.float32, (4,)),
    ("angular_axis", np.float32, (4,)),
], align=False)

# Type constants (must match common.cuh)
SDF_BOX = 0
SDF_SPHERE = 1
SDF_CYLINDER = 2
SDF_PLANE = 3

MAX_SDF_OBJECTS = 16


def _type_as_float(sdf_type: int) -> float:
    """Encode an integer SDF type as a float using bit reinterpretation."""
    return np.array([sdf_type], dtype=np.int32).view(np.float32)[0]


def make_sdf_object(
    sdf_type: int,
    position: tuple[float, float, float] = (0.0, 0.0, 0.0),
    size: tuple[float, float, float] = (0.1, 0.1, 0.1),
    rotation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
    velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
    restitution: float = 0.5,
    friction: float = 0.3,
    angular_speed: float = 0.0,
    angular_axis: tuple[float, float, float] = (0.0, 1.0, 0.0),
) -> np.ndarray:
    """Create a single SDFObject as a numpy structured array element."""
    obj = np.zeros(1, dtype=SDF_OBJECT_DTYPE)
    obj[0]["pos_and_type"] = (*position, _type_as_float(sdf_type))
    obj[0]["size_and_r"] = (*size, restitution)
    obj[0]["quat"] = rotation  # (x, y, z, w) where w is scalar
    obj[0]["velocity"] = (*velocity, angular_speed)
    obj[0]["angular_axis"] = (*angular_axis, friction)
    return obj[0]


def upload_sdf_objects(objects: np.ndarray | None, count: int) -> None:
    """Upload SDF objects to constant memory in all relevant CUDA modules.

    Parameters
    ----------
    objects : np.ndarray or None
        Array of SDF_OBJECT_DTYPE with up to MAX_SDF_OBJECTS entries.
        If None or count==0, just sets c_num_sdf_objects to 0.
    count : int
        Number of active SDF objects (0..16).
    """
    import integrate
    import pbf_solver
    import dfsph_solver

    modules = [
        integrate.get_module(),
        pbf_solver.get_module(),
        dfsph_solver.get_module(),
    ]

    count = min(count, MAX_SDF_OBJECTS)
    count_arr = np.array([count], dtype=np.int32)

    for module in modules:
        _upload_sdf_to_module(module, objects, count, count_arr)


def _upload_sdf_to_module(
    module: object, objects: np.ndarray | None, count: int, count_arr: np.ndarray
) -> None:
    """Upload SDF data to a single CUDA module's constant memory.

    Silently skips if the module doesn't reference c_sdf_objects
    (symbol optimized away by CUDA compiler when not used yet).
    """
    try:
        d_count = module.get_global("c_num_sdf_objects")  # type: ignore[union-attr]
    except Exception:
        return  # Symbol not present in this module yet

    cupy.cuda.runtime.memcpy(
        int(d_count), count_arr.ctypes.data, count_arr.nbytes, 1
    )

    if count > 0 and objects is not None:
        d_objs = module.get_global("c_sdf_objects")  # type: ignore[union-attr]
        nbytes = objects[:count].nbytes
        cupy.cuda.runtime.memcpy(
            int(d_objs), objects[:count].ctypes.data, nbytes, 1
        )


# ---------------------------------------------------------------------------
# RigidBody dtype -- matches struct RigidBody in common.cuh (96 bytes)
#
# 6 float4s, each stored as 4 x float32:
#   position:     xyz=COM, w=inv_mass
#   rotation:     quaternion (x,y,z,w), w is scalar part
#   lin_vel:      xyz=velocity, w=restitution
#   ang_vel:      xyz=angular velocity, w=friction
#   half_extents: xyz=size, w=int_as_float(sdf_type)
#   inertia_inv:  xyz=1/I_diagonal, w=int_as_float(is_kinematic)
# ---------------------------------------------------------------------------

RIGID_BODY_DTYPE = np.dtype([
    ("position",     np.float32, (4,)),
    ("rotation",     np.float32, (4,)),
    ("lin_vel",      np.float32, (4,)),
    ("ang_vel",      np.float32, (4,)),
    ("half_extents", np.float32, (4,)),
    ("inertia_inv",  np.float32, (4,)),
], align=False)

assert RIGID_BODY_DTYPE.itemsize == 96, (
    f"RigidBody size mismatch: {RIGID_BODY_DTYPE.itemsize} != 96"
)

MAX_RIGID_BODIES = 8


def sample_box_surface(half_extents: tuple, spacing: float) -> tuple:
    """Sample boundary particles on all 6 faces of a box.

    Returns
    -------
    (positions, normals) : (np.ndarray (N,3), np.ndarray (N,3)) float32
        Local-space positions and outward face normals.
    """
    hx, hy, hz = half_extents
    particles = []
    normals = []

    # +X/-X faces: grid in (y, z)
    ys = np.arange(-hy + spacing * 0.5, hy, spacing, dtype=np.float32)
    zs = np.arange(-hz + spacing * 0.5, hz, spacing, dtype=np.float32)
    if len(ys) == 0:
        ys = np.array([0.0], dtype=np.float32)
    if len(zs) == 0:
        zs = np.array([0.0], dtype=np.float32)
    yg, zg = np.meshgrid(ys, zs, indexing='ij')
    yf, zf = yg.ravel(), zg.ravel()
    n = len(yf)
    for sign in [1.0, -1.0]:
        face = np.column_stack([np.full(n, sign * hx, dtype=np.float32), yf, zf])
        nrm = np.zeros((n, 3), dtype=np.float32)
        nrm[:, 0] = sign
        particles.append(face)
        normals.append(nrm)

    # +Y/-Y faces: grid in (x, z)
    xs = np.arange(-hx + spacing * 0.5, hx, spacing, dtype=np.float32)
    if len(xs) == 0:
        xs = np.array([0.0], dtype=np.float32)
    xg, zg = np.meshgrid(xs, zs, indexing='ij')
    xf, zf = xg.ravel(), zg.ravel()
    n = len(xf)
    for sign in [1.0, -1.0]:
        face = np.column_stack([xf, np.full(n, sign * hy, dtype=np.float32), zf])
        nrm = np.zeros((n, 3), dtype=np.float32)
        nrm[:, 1] = sign
        particles.append(face)
        normals.append(nrm)

    # +Z/-Z faces: grid in (x, y)
    xg, yg = np.meshgrid(xs, ys, indexing='ij')
    xf, yf = xg.ravel(), yg.ravel()
    n = len(xf)
    for sign in [1.0, -1.0]:
        face = np.column_stack([xf, yf, np.full(n, sign * hz, dtype=np.float32)])
        nrm = np.zeros((n, 3), dtype=np.float32)
        nrm[:, 2] = sign
        particles.append(face)
        normals.append(nrm)

    return (np.concatenate(particles, axis=0).astype(np.float32),
            np.concatenate(normals, axis=0).astype(np.float32))


def sample_sphere_surface(radius: float, spacing: float) -> tuple:
    """Sample boundary particles on a sphere surface using Fibonacci spiral.

    Returns
    -------
    (positions, normals) : (np.ndarray (N,3), np.ndarray (N,3)) float32
        Local-space positions and outward radial normals.
    """
    import math
    surface_area = 4.0 * math.pi * radius * radius
    N = max(int(math.ceil(surface_area / (spacing * spacing))), 1)

    golden_ratio = (1.0 + math.sqrt(5.0)) / 2.0
    indices = np.arange(N, dtype=np.float64)
    theta = np.arccos(1.0 - 2.0 * (indices + 0.5) / N)
    phi = 2.0 * math.pi * indices / golden_ratio

    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)

    positions = np.column_stack([x, y, z]).astype(np.float32)
    # Normals are radial (same direction as position, normalized)
    normals = positions / radius

    return positions, normals


_psi_module = None


def _ensure_ptx_if_needed() -> None:
    """Force PTX compilation when GPU arch exceeds NVRTC's max sm target."""
    from cupy.cuda import compiler as _compiler
    from cupy.cuda import device as _device

    gpu_cc = _device.Device().compute_capability
    nvrtc_max = _compiler._get_max_compute_capability()
    if int(gpu_cc) > int(nvrtc_max):
        _compiler._use_ptx = True
        if hasattr(_compiler._get_arch_for_options_for_nvrtc, "_cache"):
            _compiler._get_arch_for_options_for_nvrtc._cache = {}
        if hasattr(_compiler._get_arch, "_cache"):
            _compiler._get_arch._cache = {}


def _get_psi_module():
    """Compile or return cached RawModule for boundary volume computation."""
    global _psi_module
    if _psi_module is not None:
        return _psi_module

    _ensure_ptx_if_needed()

    _psi_module = cupy.RawModule(code=r"""
    extern "C" __global__
    void K_ComputeBoundaryVolumes(
        const float* __restrict__ positions,
        int N,
        float h_sq,
        float poly6_coeff,
        float rho0,
        float* __restrict__ psi_out
    ) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= N) return;

        float xi = positions[i * 3 + 0];
        float yi = positions[i * 3 + 1];
        float zi = positions[i * 3 + 2];

        float sum_W = 0.0f;
        for (int j = 0; j < N; j++) {
            float dx = xi - positions[j * 3 + 0];
            float dy = yi - positions[j * 3 + 1];
            float dz = zi - positions[j * 3 + 2];
            float rlen_sq = dx * dx + dy * dy + dz * dz;
            if (rlen_sq < h_sq) {
                float diff = h_sq - rlen_sq;
                sum_W += poly6_coeff * diff * diff * diff;
            }
        }

        psi_out[i] = rho0 / fmaxf(sum_W, 1e-10f);
    }
    """, options=("--std=c++11",))
    return _psi_module


def compute_boundary_volumes(
    positions: np.ndarray,
    smoothing_length: float,
    rho0: float,
) -> cupy.ndarray:
    """Compute Akinci boundary volumes (psi_b) for sampled particles.

    Uses W_poly6 with squared arguments. Runs O(N^2) brute-force which
    is fine for ~600-1200 boundary particles per body.

    Parameters
    ----------
    positions : np.ndarray (N, 3) float32
        Local-space boundary particle positions.
    smoothing_length : float
        SPH smoothing length h.
    rho0 : float
        Rest density of the fluid.

    Returns
    -------
    cupy.ndarray (N,) float32
        Boundary volumes psi_b.
    """
    import math
    N = len(positions)
    h = smoothing_length
    h_sq = h * h
    poly6_coeff = 315.0 / (64.0 * math.pi * h**9)

    d_positions = cupy.asarray(positions.ravel(), dtype=cupy.float32)
    d_psi = cupy.zeros(N, dtype=cupy.float32)

    module = _get_psi_module()
    kernel = module.get_function("K_ComputeBoundaryVolumes")
    block = 256
    grid = (N + block - 1) // block
    kernel((grid,), (block,), (d_positions, np.int32(N), np.float32(h_sq),
                               np.float32(poly6_coeff), np.float32(rho0), d_psi))

    return d_psi


class RigidBodyManager:
    """Manages rigid bodies and their GPU-side state.

    Rigid body data lives in global GPU memory (not __constant__) because
    the rigid body integration kernel writes updated state each substep.
    Only c_num_rigid_bodies (the count) goes to constant memory.

    Force/torque accumulators are global memory float arrays sized for
    MAX_RIGID_BODIES * 4 floats (xyz + padding). Zero-cleared each substep.

    Boundary particle data is stored as a combined CuPy array with columns:
    (x_local, y_local, z_local, psi, body_id, nx, ny, nz)
    """

    def __init__(self, max_bodies: int = MAX_RIGID_BODIES) -> None:
        self.max_bodies = max_bodies
        self._bodies_cpu = np.zeros(max_bodies, dtype=RIGID_BODY_DTYPE)
        self._num_bodies = 0
        self._boundary_data_list = []  # per-body boundary data before combining
        self.num_boundary_particles = 0

        # GPU arrays
        self.d_rigid_bodies = cupy.zeros(max_bodies, dtype=RIGID_BODY_DTYPE)
        self.rigid_forces = cupy.zeros((max_bodies, 4), dtype=cupy.float32)
        self.rigid_torques = cupy.zeros((max_bodies, 4), dtype=cupy.float32)

        # Combined boundary particle data (set after all bodies added)
        # Columns: x_local, y_local, z_local, psi, body_id, nx, ny, nz
        self.boundary_data = None  # cupy (N, 8) float32

    @property
    def num_bodies(self) -> int:
        return self._num_bodies

    def add_rigid_body_with_particles(
        self,
        position: tuple,
        half_extents: tuple,
        mass: float,
        sdf_type: int = SDF_BOX,
        spacing: float = 0.02,
        smoothing_length: float = 0.04,
        rho0: float = 2500.0,
        restitution: float = 0.5,
        friction: float = 0.3,
        is_kinematic: bool = False,
        velocity: tuple = (0.0, 0.0, 0.0),
    ) -> int:
        """Add a rigid body with boundary particle sampling and psi_b computation.

        Convenience method that calls add_rigid_body, samples surface particles,
        computes Akinci boundary volumes, and stores boundary data.
        """
        idx = self.add_rigid_body(
            position, half_extents, mass, sdf_type,
            restitution, friction, is_kinematic, velocity
        )

        # Sample surface
        if sdf_type == SDF_SPHERE:
            positions, normals = sample_sphere_surface(half_extents[0], spacing)
        else:
            positions, normals = sample_box_surface(half_extents, spacing)

        # Compute boundary volumes
        psi = compute_boundary_volumes(positions, smoothing_length, rho0)
        psi_cpu = psi.get()

        # Build boundary data: (x, y, z, psi, body_id, nx, ny, nz)
        N = len(positions)
        data = np.zeros((N, 8), dtype=np.float32)
        data[:, 0:3] = positions
        data[:, 3] = psi_cpu
        data[:, 4] = float(idx)
        data[:, 5:8] = normals

        self._boundary_data_list.append(data)
        return idx

    def finalize_boundary_data(self) -> None:
        """Combine all per-body boundary data into a single GPU array."""
        if not self._boundary_data_list:
            self.boundary_data = None
            self.num_boundary_particles = 0
            return
        combined = np.concatenate(self._boundary_data_list, axis=0)
        self.boundary_data = cupy.asarray(combined)
        self.num_boundary_particles = len(combined)

    def add_rigid_body(
        self,
        position: tuple,
        half_extents: tuple,
        mass: float,
        sdf_type: int = SDF_BOX,
        restitution: float = 0.5,
        friction: float = 0.3,
        is_kinematic: bool = False,
        velocity: tuple = (0.0, 0.0, 0.0),
    ) -> int:
        """Add a rigid body and return its index.

        Automatically computes the inertia tensor for box or sphere primitives.

        Returns
        -------
        int
            Index of the new rigid body.
        """
        if self._num_bodies >= self.max_bodies:
            raise RuntimeError(f"Max rigid bodies ({self.max_bodies}) exceeded")

        idx = self._num_bodies
        body = self._bodies_cpu[idx]

        inv_mass = 0.0 if is_kinematic else 1.0 / mass
        body["position"][:] = [position[0], position[1], position[2], inv_mass]

        # Identity quaternion
        body["rotation"][:] = [0.0, 0.0, 0.0, 1.0]

        body["lin_vel"][:] = [velocity[0], velocity[1], velocity[2], restitution]
        body["ang_vel"][:] = [0.0, 0.0, 0.0, friction]

        hx, hy, hz = half_extents
        body["half_extents"][:] = [hx, hy, hz, _type_as_float(sdf_type)]

        # Compute inertia inverse
        if is_kinematic:
            ix_inv = iy_inv = iz_inv = 0.0
        elif sdf_type == SDF_SPHERE:
            r = hx
            I = 0.4 * mass * r * r  # I = 2/5 * m * r^2
            ix_inv = iy_inv = iz_inv = 1.0 / I
        else:  # SDF_BOX
            # Half-extent box: I_xx = m*(hy^2 + hz^2) / 3, etc.
            # (Derivation: integrate x^2 over [-hx,hx] => hx^3/3 * 2 = 2hx^3/3;
            #  full tensor I_xx = m*(hy^2+hz^2)/3 for a box with half-extents.)
            # Divisor is 3, NOT 12 (which would apply to full-extent side lengths).
            ix_inv = 3.0 / (mass * (hy * hy + hz * hz))
            iy_inv = 3.0 / (mass * (hx * hx + hz * hz))
            iz_inv = 3.0 / (mass * (hx * hx + hy * hy))

        body["inertia_inv"][:] = [ix_inv, iy_inv, iz_inv,
                                  _type_as_float(int(is_kinematic))]

        self._num_bodies += 1
        return idx

    def upload(self, modules: list) -> None:
        """Upload rigid body array to GPU global memory and count to constant memory.

        Parameters
        ----------
        modules : list
            CuPy RawModule objects to upload c_num_rigid_bodies count to.
        """
        # Copy body data to device global memory
        if self._num_bodies > 0:
            self.d_rigid_bodies[:self._num_bodies] = cupy.asarray(
                self._bodies_cpu[:self._num_bodies]
            )

        # Upload count to constant memory in all modules
        h_count = np.array([self._num_bodies], dtype=np.int32)
        for module in modules:
            try:
                d_ptr = module.get_global("c_num_rigid_bodies")
                cupy.cuda.runtime.memcpy(
                    int(d_ptr), h_count.ctypes.data, 4, 1
                )
            except Exception:
                pass  # Module may not have this symbol yet

    def zero_accumulators(self) -> None:
        """Zero force and torque accumulators (call at start of each substep)."""
        self.rigid_forces.fill(0)
        self.rigid_torques.fill(0)

    def download_bodies(self) -> np.ndarray:
        """Download rigid body data from GPU to CPU (for debug/readback)."""
        return self.d_rigid_bodies[:self._num_bodies].get()

    def inject_boundary_particles(self, world) -> int:
        """Inject boundary particles into world arrays for hash/sort.

        Transforms local-space boundary positions to world-space using
        each rigid body's current position and rotation, then appends
        them to the world's particle arrays.

        Must be called each substep BEFORE the hash/sort step.

        Parameters
        ----------
        world : World
            The particle world to inject into.

        Returns
        -------
        int
            Total particle count (fluid + boundary).
        """
        from materials import MAT_RIGID, STATIC

        if self.boundary_data is None or self.num_boundary_particles == 0:
            return world._high_water

        n_fluid = world._high_water
        n_boundary = self.num_boundary_particles
        total = n_fluid + n_boundary

        if total > world.max_particles:
            # Silently skip if not enough room
            return n_fluid

        # boundary_data columns: x_local, y_local, z_local, psi, body_id, nx, ny, nz
        bd = self.boundary_data  # cupy (N, 8)

        # For each body, transform local positions to world space
        # Simple approach: iterate bodies on CPU, batch-transform on GPU
        # Since we have at most 8 bodies, this is fine
        for body_idx in range(self._num_bodies):
            body = self._bodies_cpu[body_idx]
            cx, cy, cz = body["position"][:3]
            qx, qy, qz, qw = body["rotation"]

            # Find particles for this body
            mask = bd[:, 4] == float(body_idx)
            local_pos = bd[mask, :3]  # (M, 3)
            psi = bd[mask, 3]
            M = len(psi)
            if M == 0:
                continue

            # Quaternion rotation on GPU (vectorized)
            # q * v * q^-1 using the formula from quat_rotate
            ux, uy, uz, s = float(qx), float(qy), float(qz), float(qw)
            vx, vy, vz = local_pos[:, 0], local_pos[:, 1], local_pos[:, 2]

            # dot(u, v)
            d = ux * vx + uy * vy + uz * vz
            # dot(u, u)
            uu = ux * ux + uy * uy + uz * uz
            # cross(u, v)
            crsx = uy * vz - uz * vy
            crsy = uz * vx - ux * vz
            crsz = ux * vy - uy * vx
            # a = s^2 - dot(u,u)
            a = s * s - uu

            wx = 2.0 * d * ux + a * vx + 2.0 * s * crsx + cx
            wy = 2.0 * d * uy + a * vy + 2.0 * s * crsy + cy
            wz = 2.0 * d * uz + a * vz + 2.0 * s * crsz + cz

            # Find the actual indices in the combined boundary array
            indices = cupy.flatnonzero(mask)
            start = n_fluid + int(indices[0])
            end = start + M

            sl = slice(start, end)
            world.position[sl, 0] = wx
            world.position[sl, 1] = wy
            world.position[sl, 2] = wz
            world.position[sl, 3] = 0.0

            world.velocity[sl] = 0.0
            world.veleval[sl] = 0.0
            world.sph_force[sl] = 0.0

            # Mass = psi_b (Akinci boundary volume)
            world.mass[sl] = psi

            # packed_info: MAT_RIGID + STATIC + body_id
            packed = (int(MAT_RIGID) & 0xFF) | ((int(STATIC) & 0x3) << 8)
            packed = packed | ((int(body_idx) & 0xFF) << 13)
            world.packed_info[sl] = cupy.uint32(packed)

            world.density[sl] = 2500.0
            world.temperature[sl] = 293.0
            world.health[sl] = 1.0
            world.lifetime[sl] = 0.0
            world.sleep_counter[sl] = 0

            # Color for rigid body particles
            world.color[sl, 0] = 0.6
            world.color[sl, 1] = 0.6
            world.color[sl, 2] = 0.65
            world.color[sl, 3] = 0.75  # STATIC alpha

        return total

    def reset(self) -> None:
        """Clear all rigid bodies and boundary data."""
        self._num_bodies = 0
        self._bodies_cpu[:] = 0
        self._boundary_data_list = []
        self.num_boundary_particles = 0
        self.boundary_data = None
        self.d_rigid_bodies.fill(0)
        self.rigid_forces.fill(0)
        self.rigid_torques.fill(0)


# ---------------------------------------------------------------------------
# SDFManager -- manages SDF collision objects (constant memory, Phase A)
# ---------------------------------------------------------------------------

class SDFManager:
    """Manages SDF collision objects stored in GPU constant memory.

    SDF objects are purely collision primitives -- no dynamic simulation.
    Supports kinematic motion via update_sdf_object() or add_kinematic_motion().
    Uses dirty flag for lazy upload.
    """

    def __init__(self) -> None:
        self._objects: list[dict] = []
        self._dirty = True
        self._motions: dict[int, dict] = {}  # obj_id -> motion config

    @property
    def count(self) -> int:
        return len(self._objects)

    def add_sdf_object(
        self,
        sdf_type: int,
        position: tuple = (0.0, 0.0, 0.0),
        size: tuple = (0.1, 0.1, 0.1),
        rotation: tuple = (0.0, 0.0, 0.0, 1.0),
        velocity: tuple = (0.0, 0.0, 0.0),
        restitution: float = 0.5,
        friction: float = 0.3,
        angular_speed: float = 0.0,
        angular_axis: tuple = (0.0, 1.0, 0.0),
    ) -> int:
        """Add an SDF object. Returns object ID (0-15)."""
        if len(self._objects) >= MAX_SDF_OBJECTS:
            raise RuntimeError(f"Max SDF objects ({MAX_SDF_OBJECTS}) exceeded")
        obj_id = len(self._objects)
        self._objects.append({
            "type": sdf_type,
            "position": list(position),
            "size": list(size),
            "rotation": list(rotation),
            "velocity": list(velocity),
            "restitution": restitution,
            "friction": friction,
            "angular_speed": angular_speed,
            "angular_axis": list(angular_axis),
        })
        self._dirty = True
        return obj_id

    def remove_sdf_object(self, obj_id: int) -> None:
        """Remove SDF object by ID and re-pack."""
        if 0 <= obj_id < len(self._objects):
            self._objects.pop(obj_id)
            self._dirty = True

    def update_sdf_object(self, obj_id: int, **kwargs) -> None:
        """Update fields of an existing SDF object."""
        if 0 <= obj_id < len(self._objects):
            obj = self._objects[obj_id]
            for key, val in kwargs.items():
                if key in obj:
                    obj[key] = list(val) if isinstance(val, (tuple, list)) else val
            self._dirty = True

    def get_sdf_objects(self) -> list[dict]:
        """Return list of SDF object dicts for UI display."""
        return [dict(d, id=i) for i, d in enumerate(self._objects)]

    def _build_array(self) -> np.ndarray:
        """Pack SDF objects into numpy array for GPU upload."""
        n = len(self._objects)
        arr = np.zeros(max(n, 1), dtype=SDF_OBJECT_DTYPE)
        for i, obj in enumerate(self._objects):
            arr[i] = make_sdf_object(
                sdf_type=obj["type"],
                position=tuple(obj["position"]),
                size=tuple(obj["size"]),
                rotation=tuple(obj["rotation"]),
                velocity=tuple(obj["velocity"]),
                restitution=obj["restitution"],
                friction=obj["friction"],
                angular_speed=obj["angular_speed"],
                angular_axis=tuple(obj["angular_axis"]),
            )
        return arr

    def upload_if_dirty(self) -> None:
        """Upload SDF objects to constant memory if changed."""
        if not self._dirty:
            return
        arr = self._build_array()
        upload_sdf_objects(arr, len(self._objects))
        self._dirty = False

    def force_upload(self) -> None:
        """Force re-upload regardless of dirty flag."""
        self._dirty = True
        self.upload_if_dirty()

    # --- Kinematic motion ---

    def add_kinematic_motion(self, obj_id: int, motion_type: str, params: dict) -> None:
        """Attach a kinematic motion to an SDF object.

        motion_type: "rotate_y", "oscillate_x", "oscillate_y", "oscillate_z"
        params for rotate_y: {"speed": float}  (rad/s)
        params for oscillate_*: {"amplitude": float, "frequency": float}
        """
        if 0 <= obj_id < len(self._objects):
            self._motions[obj_id] = {
                "type": motion_type,
                "params": dict(params),
                "center": list(self._objects[obj_id]["position"]),  # snapshot initial pos
            }

    def remove_kinematic_motion(self, obj_id: int) -> None:
        """Stop kinematic motion for the given object."""
        self._motions.pop(obj_id, None)

    def update_kinematics(self, t: float, dt: float) -> None:
        """Update all kinematic SDF objects for simulation time t.

        Computes position/rotation and their analytical velocity derivatives.
        Marks dirty so next upload_if_dirty() will push to GPU.
        """
        import math
        if not self._motions:
            return

        changed = False
        for obj_id, motion in self._motions.items():
            if obj_id >= len(self._objects):
                continue
            obj = self._objects[obj_id]
            mtype = motion["type"]
            p = motion["params"]

            if mtype == "rotate_y":
                speed = p["speed"]  # rad/s
                angle = speed * t
                half = angle * 0.5
                # quat = (0, sin(a/2), 0, cos(a/2))
                obj["rotation"] = [0.0, math.sin(half), 0.0, math.cos(half)]
                obj["angular_axis"] = [0.0, 1.0, 0.0]
                obj["angular_speed"] = speed
                obj["velocity"] = [0.0, 0.0, 0.0]
                changed = True

            elif mtype.startswith("oscillate_"):
                axis = mtype[-1]  # 'x', 'y', or 'z'
                amp = p["amplitude"]
                freq = p["frequency"]
                omega = 2.0 * math.pi * freq
                center = motion["center"]

                # position = center + amp * sin(omega * t) along axis
                pos = list(center)
                axis_idx = {"x": 0, "y": 1, "z": 2}[axis]
                pos[axis_idx] = center[axis_idx] + amp * math.sin(omega * t)
                obj["position"] = pos

                # velocity = derivative: amp * omega * cos(omega * t) along axis
                vel = [0.0, 0.0, 0.0]
                vel[axis_idx] = amp * omega * math.cos(omega * t)
                obj["velocity"] = vel
                obj["angular_speed"] = 0.0
                changed = True

        if changed:
            self._dirty = True

    def clear(self) -> None:
        """Remove all SDF objects and upload empty state."""
        self._objects.clear()
        self._motions.clear()
        self._dirty = True
        self.upload_if_dirty()
