"""Render and UI math tests that need no GL context and no GPU work (bd-r5-review-n28.4).

(a) OrbitCamera projection round-trips through _unproject_mouse_ray.
(b) _build_sdf_model_matrix: plane normal, box and sphere rotation; plane raycast.
(c) Simulation.copy_to_vbos returns the drawn particle count, rigid tail included.
(d) UI._on_mouse_button forwards a release and blocks a press under ImGui capture.
(e) OrbitCamera.zoom with a large delta stays positive and above min_distance.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from camera import OrbitCamera


def _normalize(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


def _quat_axis_angle(axis, deg):
    axis = _normalize(axis)
    half = np.radians(deg) / 2.0
    s = np.sin(half)
    return [axis[0] * s, axis[1] * s, axis[2] * s, np.cos(half)]


def _rotate(v, q):
    """Rotate v by unit quaternion q = (x, y, z, w)."""
    qv = np.array(q[:3], dtype=np.float64)
    w = q[3]
    v = np.asarray(v, dtype=np.float64)
    t = 2.0 * np.cross(qv, v)
    return v + w * t + np.cross(qv, t)


# ---------------------------------------------------------------------------
# (a) projection and unprojection round-trip
# ---------------------------------------------------------------------------

POSES = [
    dict(target=[0.0, 0.0, 0.0], distance=3.0, azimuth=0.0, elevation=30.0),
    dict(target=[0.5, 0.2, -0.3], distance=1.2, azimuth=135.0, elevation=-20.0),
    dict(target=[-1.0, 0.5, 2.0], distance=8.0, azimuth=-60.0, elevation=75.0),
]

POINTS = [
    [0.0, 0.0, 0.0],
    [0.3, -0.2, 0.1],
    [-0.4, 0.5, 0.25],
]


@pytest.mark.parametrize("pose", POSES)
def test_unproject_mouse_ray_inverts_projection(pose):
    ui = pytest.importorskip("ui")
    cam = OrbitCamera(**pose)
    w, h = 1280, 720
    cam.set_aspect(w, h)
    view = cam.view_matrix()
    proj = cam.projection_matrix()

    for offset in POINTS:
        p = np.asarray(pose["target"], dtype=np.float64) + offset
        clip = proj.astype(np.float64) @ view.astype(np.float64) @ np.append(p, 1.0)
        assert clip[3] > 0.0, "test point must be in front of the camera"
        ndc = clip[:3] / clip[3]
        mx = (ndc[0] + 1.0) * 0.5 * w
        my = (1.0 - ndc[1]) * 0.5 * h

        origin, direction = ui._unproject_mouse_ray(mx, my, w, h, view, proj)
        assert abs(np.linalg.norm(direction) - 1.0) < 1e-5
        rel = p - origin.astype(np.float64)
        assert np.dot(rel, direction) > 0.0, "point must lie ahead of the ray origin"
        miss = np.linalg.norm(np.cross(rel, direction.astype(np.float64)))
        assert miss < 1e-4, f"ray misses point {p} by {miss:.2e} for pose {pose}"


# ---------------------------------------------------------------------------
# (b) SDF model matrix
# ---------------------------------------------------------------------------

def _model_matrix(sdf_type, pos, rot, size):
    pytest.importorskip("OpenGL")
    renderer = pytest.importorskip("renderer")
    return renderer.Renderer._build_sdf_model_matrix(sdf_type, pos, rot, size).astype(np.float64)


@pytest.mark.parametrize(
    "size",
    [
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.7, 0.7, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 0.2, -1.0],
        [-0.3, -0.8, 0.5],
    ],
)
def test_sdf_plane_model_matrix_matches_kernel_normal(size):
    # The plane kernel uses normalize(size) and ignores rotation, so a non-identity
    # quaternion must not move the drawn normal (bd-r5-review-n28.33).
    rot = _quat_axis_angle([0.3, 1.0, -0.4], 70.0)
    renderer = pytest.importorskip("renderer")
    m = renderer._build_sdf_model_matrix(3, [0.1, 0.2, 0.3], rot, size).astype(np.float64)
    up = m @ np.array([0.0, 1.0, 0.0, 0.0])
    assert np.allclose(up[:3], _normalize(size), atol=1e-5)
    assert abs(up[3]) < 1e-12
    assert np.allclose(m[:3, 3], [0.1, 0.2, 0.3], atol=1e-6)
    # The static method on Renderer stays a thin wrapper.
    assert np.allclose(_model_matrix(3, [0.1, 0.2, 0.3], rot, size), m)


def test_raycast_hits_tilted_plane_at_expected_distance():
    ui = pytest.importorskip("ui")
    normal = _normalize([0.7, 0.7, 0.0])
    center = np.array([0.2, 0.1, 0.0])
    plane = {"id": 7, "type": ui.SDF_PLANE, "position": list(center),
             "rotation": [0.0, 0.0, 0.0, 1.0], "size": [0.7, 0.7, 0.0]}
    mgr = SimpleNamespace(get_sdf_objects=lambda: [plane])

    origin = np.array([2.0, 2.0, 0.5], dtype=np.float32)
    direction = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
    expected_t = float(np.dot(center - origin, normal) / np.dot(direction, normal))

    hit_id, hit_t = ui._raycast_sdf_objects(origin, direction, mgr)
    assert hit_id == 7
    assert hit_t == pytest.approx(expected_t, abs=1e-5)
    hit = origin + hit_t * direction
    assert float(np.dot(hit - center, normal)) == pytest.approx(0.0, abs=1e-5)


@pytest.mark.parametrize(
    "vals, expected",
    [
        ([0.0, 2.0, 0.0], [0.0, 1.0, 0.0]),
        ([0.7, 0.7, 0.0], list(_normalize([1.0, 1.0, 0.0]))),
        ([0.0, 0.0, 0.0], None),
        ([1e-7, 0.0, 0.0], None),
    ],
)
def test_plane_normal_slider_value_is_normalised(vals, expected):
    ui = pytest.importorskip("ui")
    n = ui._normalize_plane_normal(vals)
    if expected is None:
        assert n is None
    else:
        assert np.allclose(n, expected, atol=1e-7)
        assert np.linalg.norm(n) == pytest.approx(1.0, abs=1e-7)


@pytest.mark.parametrize("sdf_type", [0, 1])
def test_sdf_box_and_sphere_model_matrix_honour_rotation(sdf_type):
    pos = [0.4, -0.1, 0.7]
    rot = _quat_axis_angle([1.0, 2.0, 0.5], 40.0)
    size = [0.3, 0.2, 0.1]
    m = _model_matrix(sdf_type, pos, rot, size)

    assert np.allclose(m[:3, 3], pos, atol=1e-6)
    scale = size if sdf_type == 0 else [size[0]] * 3
    for axis in range(3):
        local = np.zeros(3)
        local[axis] = 1.0
        expected = _rotate(local * scale[axis], rot)
        assert np.allclose(m[:3, :3] @ local, expected, atol=1e-5)


# ---------------------------------------------------------------------------
# (c) VBO copy range
# ---------------------------------------------------------------------------

class _FakeGLBuffer:
    def __init__(self, n):
        self.host = np.zeros((n, 4), dtype=np.float32)
        self.nbytes = self.host.nbytes

    def device_pointer_as_cupy_array(self, shape, dtype):
        assert shape == self.host.shape and np.dtype(dtype) == self.host.dtype
        return self.host


def _fake_sim(high_water, rigid_tail, capacity=512):
    rng = np.random.default_rng(0)
    world = SimpleNamespace(
        _high_water=high_water,
        position=rng.random((capacity, 4), dtype=np.float32),
        color=rng.random((capacity, 4), dtype=np.float32),
        velocity=rng.random((capacity, 4), dtype=np.float32),
    )
    return SimpleNamespace(world=world, _rigid_tail_count=rigid_tail)


@pytest.mark.xfail(
    strict=True,
    reason="copy_to_vbos returns None and stops at _high_water. "
    "Fix bead bd-r5-review-n28.51 (8-rigid-draw) not landed yet. Remove this marker when it does.",
)
def test_copy_to_vbos_returns_drawn_count_with_rigid_tail():
    pytest.importorskip("cupy")
    simulation = pytest.importorskip("simulation")
    copy = simulation.Simulation.copy_to_vbos

    for tail in (0, 64):
        sim = _fake_sim(high_water=200, rigid_tail=tail)
        pos, col, vel = _FakeGLBuffer(512), _FakeGLBuffer(512), _FakeGLBuffer(512)
        n = copy(sim, pos, col, vel)
        assert n == 200 + tail
        assert np.array_equal(pos.host[:n], sim.world.position[:n])
        assert np.array_equal(col.host[:n], sim.world.color[:n])
        assert np.array_equal(vel.host[:n], sim.world.velocity[:n])


# ---------------------------------------------------------------------------
# (d) mouse release routing (full coverage lives in test_ui_mouse.py)
# ---------------------------------------------------------------------------

def test_mouse_release_forwarded_and_press_blocked_under_capture(monkeypatch):
    pytest.importorskip("imgui_bundle")
    glfw = pytest.importorskip("glfw")
    from test_ui_mouse import _make_ui

    u, calls = _make_ui(monkeypatch, capture=True)
    u._on_mouse_button(None, glfw.MOUSE_BUTTON_RIGHT, glfw.PRESS, 0)
    assert calls == []
    u._on_mouse_button(None, glfw.MOUSE_BUTTON_RIGHT, glfw.RELEASE, 0)
    assert calls == [(glfw.MOUSE_BUTTON_RIGHT, glfw.RELEASE)]


# ---------------------------------------------------------------------------
# (e) zoom clamp
# ---------------------------------------------------------------------------

def test_zoom_large_delta_stays_positive_and_off_limits():
    cam = OrbitCamera(distance=3.0)
    cam.zoom(50.0)
    assert cam.distance > 0.0
    assert cam.distance > cam.min_distance
    assert cam.distance >= 1.5 - 1e-9

    cam = OrbitCamera(distance=3.0)
    cam.zoom(-50.0)
    assert cam.distance < cam.max_distance
    assert cam.distance <= 6.0 + 1e-9

    # A normal scroll step keeps its old size.
    cam = OrbitCamera(distance=3.0)
    cam.zoom(1.0)
    assert abs(cam.distance - 3.0 * (1.0 - cam.zoom_sensitivity)) < 1e-9


# ---------------------------------------------------------------------------
# (f) Save / Load panel scans the saves directory once, not every frame
#     (bd-r5-review-n28.13)
# ---------------------------------------------------------------------------

class _StubImgui:
    """Minimal imgui stand-in for UI._draw_save_load_panel."""

    class Cond_:
        first_use_ever = 0

    class WindowFlags_:
        always_auto_resize = 0

    def __init__(self):
        self.pressed = set()
        self.input_text_args = []

    @staticmethod
    def ImVec2(x, y):
        return (x, y)

    def set_next_window_pos(self, *a):
        pass

    def set_next_window_size(self, *a):
        pass

    def begin(self, *a):
        return (True, True)

    def end(self):
        pass

    def input_text(self, *args):
        self.input_text_args.append(args)
        return False, args[1]

    def button(self, label, size=None):
        return label in self.pressed

    def text(self, *a):
        pass

    def separator(self):
        pass

    def combo(self, label, idx, items):
        return False, idx


def test_save_panel_scans_saves_once_then_on_refresh_and_save(monkeypatch):
    pytest.importorskip("imgui_bundle")
    ui = pytest.importorskip("ui")
    stub = _StubImgui()
    monkeypatch.setattr(ui, "imgui", stub)
    calls = []
    monkeypatch.setattr(ui, "list_saves", lambda: calls.append(1) or [])
    monkeypatch.setattr(ui, "save_scene", lambda w, s, c, name: f"/x/{name}.fs3d")

    u = ui.UI.__new__(ui.UI)
    u._save_filename = "scene"
    u._save_status = ""
    u._load_files = []
    u._load_files_scanned = False
    u._load_selected = 0
    u._camera_ref = object()

    for _ in range(100):
        u._draw_save_load_panel(None, None, {})
    assert len(calls) == 1

    stub.pressed = {"Refresh"}
    u._draw_save_load_panel(None, None, {})
    stub.pressed = set()
    for _ in range(10):
        u._draw_save_load_panel(None, None, {})
    assert len(calls) == 2

    stub.pressed = {"Save"}
    u._draw_save_load_panel(None, None, {})
    stub.pressed = set()
    for _ in range(10):
        u._draw_save_load_panel(None, None, {})
    assert len(calls) == 3

    # No flags argument: 64 would be InputTextFlags_.enter_returns_true.
    assert all(args == ("Name", "scene") for args in stub.input_text_args)
