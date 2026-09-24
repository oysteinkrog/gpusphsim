"""Mouse event routing in UI._on_mouse_button (bd-r5-review-n28.3).

ImGui's want_capture_mouse is computed at new_frame, so a press can reach
the camera while the matching release reads True. A release must always
reach the user callback, or orbit, pan and SDF drag stay stuck on.
No window, GL context or GPU is needed: the ImGui backend is stubbed.
"""

from types import SimpleNamespace

import glfw
import pytest

import ui as ui_mod


class _StubRenderer:
    def __init__(self):
        self.events = []

    def mouse_button_callback(self, window, button, action, mods):
        self.events.append((button, action))


class _StubSdfManager:
    def __init__(self):
        self.uploads = 0

    def upload_if_dirty(self):
        self.uploads += 1


def _make_ui(monkeypatch, capture):
    io = SimpleNamespace(want_capture_mouse=capture)
    monkeypatch.setattr(ui_mod, "imgui", SimpleNamespace(get_io=lambda: io))
    u = ui_mod.UI.__new__(ui_mod.UI)
    u._imgui_renderer = _StubRenderer()
    u._selected_material = None
    u._selected_sdf_id = None
    u._dragging_sdf = False
    u._sdf_manager_ref = None
    u._camera_ref = None
    calls = []
    u._user_mouse_button_cb = lambda w, b, a, m: calls.append((b, a))
    return u, calls


BUTTONS = [glfw.MOUSE_BUTTON_LEFT, glfw.MOUSE_BUTTON_RIGHT, glfw.MOUSE_BUTTON_MIDDLE]


@pytest.mark.parametrize("button", BUTTONS)
def test_release_forwarded_when_imgui_captures(monkeypatch, button):
    u, calls = _make_ui(monkeypatch, capture=True)
    u._on_mouse_button(None, button, glfw.RELEASE, 0)
    assert calls == [(button, glfw.RELEASE)]
    assert u._imgui_renderer.events == [(button, glfw.RELEASE)]


@pytest.mark.parametrize("button", BUTTONS)
def test_press_blocked_when_imgui_captures(monkeypatch, button):
    u, calls = _make_ui(monkeypatch, capture=True)
    u._on_mouse_button(None, button, glfw.PRESS, 0)
    assert calls == []
    assert u._imgui_renderer.events == [(button, glfw.PRESS)]


def test_press_blocked_does_not_spawn_with_brush(monkeypatch):
    u, calls = _make_ui(monkeypatch, capture=True)
    u._selected_material = 2
    u._pending_spawn = None
    u._pending_kill = None
    u._on_mouse_button(None, glfw.MOUSE_BUTTON_LEFT, glfw.PRESS, 0)
    assert calls == []
    assert u._pending_spawn is None and u._pending_kill is None


@pytest.mark.parametrize("capture", [True, False])
def test_release_ends_sdf_drag(monkeypatch, capture):
    u, calls = _make_ui(monkeypatch, capture=capture)
    mgr = _StubSdfManager()
    u._sdf_manager_ref = mgr
    u._dragging_sdf = True
    u._on_mouse_button(None, glfw.MOUSE_BUTTON_LEFT, glfw.RELEASE, 0)
    assert u._dragging_sdf is False
    assert mgr.uploads == 1
    assert calls == [(glfw.MOUSE_BUTTON_LEFT, glfw.RELEASE)]


def test_press_forwarded_without_capture(monkeypatch):
    u, calls = _make_ui(monkeypatch, capture=False)
    u._on_mouse_button(None, glfw.MOUSE_BUTTON_MIDDLE, glfw.PRESS, 0)
    assert calls == [(glfw.MOUSE_BUTTON_MIDDLE, glfw.PRESS)]
