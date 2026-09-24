"""SDF manager upload regression tests (bd-r5-review-n28.40).

Objects panel edits only mark the SDFManager dirty. The main frame loop must
push dirty objects to constant memory before step_frame, or a newly added box
is drawn but has no collision.
"""

import ast
import os

import rigid_bodies
from rigid_bodies import SDFManager

HERE = os.path.dirname(os.path.abspath(__file__))
MAIN_PY = os.path.join(HERE, "main.py")


def test_upload_if_dirty_uploads_once(monkeypatch):
    calls = []
    monkeypatch.setattr(
        rigid_bodies, "upload_sdf_objects", lambda arr, count: calls.append(count)
    )
    mgr = SDFManager()
    mgr.upload_if_dirty()  # clear the initial dirty flag
    calls.clear()

    mgr.add_sdf_object(0, position=(0.1, 0.2, 0.3))
    assert mgr._dirty
    mgr.upload_if_dirty()
    mgr.upload_if_dirty()
    assert calls == [1]
    assert mgr._dirty is False

    mgr.update_sdf_object(0, position=(0.0, 0.5, 0.0))
    mgr.upload_if_dirty()
    mgr.remove_sdf_object(0)
    mgr.upload_if_dirty()
    assert calls == [1, 1, 0]


def _calls_in_order(tree):
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            out.append((node.lineno, node.col_offset, ast.unparse(node.func)))
    return sorted(out)


def test_main_loop_uploads_sdf_before_step_frame():
    with open(MAIN_PY, encoding="utf-8") as f:
        tree = ast.parse(f.read())
    calls = _calls_in_order(tree)
    step_lines = [ln for ln, _, name in calls if name == "sim.step_frame"]
    assert step_lines, "main.py no longer calls sim.step_frame"
    upload_lines = [
        ln for ln, _, name in calls if name == "sim.sdf_manager.upload_if_dirty"
    ]
    for step in step_lines:
        before = [ln for ln in upload_lines if step - 6 <= ln < step]
        assert before, (
            f"sim.step_frame at main.py:{step} has no "
            "sim.sdf_manager.upload_if_dirty() call right before it"
        )
