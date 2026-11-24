"""Edit history, keyboard numeric-edit validation, and state rollback for ButterflyWorkbench."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import math
from typing import Any

from ._common import _read
from .qt_compat import QT_AVAILABLE, QtCore, QtWidgets

_TRACE_METHOD_ANNULAR_PEAK = "annular_peak"


def parse_numeric_edit_inputs(
    kind: str,
    qx_text: str,
    qy_text: str,
    points_text: str,
    *,
    english: bool = True,
) -> tuple[dict[str, Any] | None, str | None]:
    """Validate keyboard-entered q-space edit inputs without touching Qt widgets.

    Returns ``(edit_dict, None)`` on success or ``(None, warning_message)`` when
    the entered values are empty, malformed, or non-finite.
    """

    if kind == "seed":
        try:
            qx_val = float(qx_text.strip())
            qy_val = float(qy_text.strip())
        except ValueError:
            return None, (
                "Enter finite numeric values for qx and qy."
                if english
                else "请输入有效的有限数值作为 qx 和 qy。"
            )
        if not (math.isfinite(qx_val) and math.isfinite(qy_val)):
            return None, (
                "Coordinates must be finite numbers (NaN and Inf are not allowed)."
                if english
                else "坐标必须为有限数值（不允许 NaN 或 Inf）。"
            )
        return {"type": "seed", "qx": qx_val, "qy": qy_val}, None

    pairs: list[list[float]] = []
    try:
        for token in points_text.split(";"):
            cleaned = token.strip()
            if not cleaned:
                continue
            x_text, y_text = cleaned.split(",", 1)
            x_val, y_val = float(x_text.strip()), float(y_text.strip())
            if not (math.isfinite(x_val) and math.isfinite(y_val)):
                raise ValueError("non-finite coordinate")
            pairs.append([x_val, y_val])
    except (TypeError, ValueError):
        return None, (
            "Enter finite coordinate pairs formatted as 'qx,qy; qx,qy; …'."
            if english
            else "请输入有限坐标点对，格式如“qx,qy; qx,qy; …”。"
        )
    resolved_kind = str(kind)
    if resolved_kind in {"exclude_rectangle", "include_rectangle"} and len(pairs) == 2:
        (x0, y0), (x1, y1) = pairs
        pairs = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
        resolved_kind = (
            "exclude_polygon" if resolved_kind == "exclude_rectangle" else "include_polygon"
        )
    if len(pairs) < 3:
        return None, (
            "Enter 2 opposite corners for a rectangle or at least 3 vertices for a polygon."
            if english
            else "矩形需输入 2 个对角点，多边形至少需输入 3 个顶点。"
        )
    return {"type": resolved_kind, "points": pairs}, None


def exclude_selected_point(workbench: Any) -> None:
    item = workbench.point_list.currentItem()
    point = item.data(QtCore.Qt.ItemDataRole.UserRole) if item is not None else None
    if workbench._trace_method() == _TRACE_METHOD_ANNULAR_PEAK:
        return
    if isinstance(point, Mapping) and bool(_read(point, ("profile_only",), False)):
        return
    point_id = _read(point, ("point_id",), None)
    if point_id not in (None, ""):
        workbench._on_edit_requested({"type": "exclude_point", "point_id": str(point_id)})


def apply_edit_request(workbench: Any, edit: Any) -> None:
    if not isinstance(edit, Mapping):
        return
    item = dict(edit)
    if item.get("type") == "seed":
        branch = workbench.seed_branch_combo.currentData()
        side = workbench.seed_side_combo.currentData()
        if branch in (0, 1):
            item["branch_id"] = int(branch)
        item["side"] = str(side or "unknown")
    workbench._edits.append(deepcopy(item))
    workbench._redo_edits.clear()
    workbench._settings["edits"] = deepcopy(workbench._edits)
    workbench.qspace.set_edits(workbench._edits)
    workbench.clear_result()
    workbench.editChanged.emit(workbench.butterfly_settings)
    workbench._update_edit_buttons()


def start_correction(workbench: Any) -> None:
    mode = str(workbench.correction_mode_combo.currentData() or "seed")
    workbench.qspace.set_interaction_mode(mode)
    workbench.status_label.setText(
        f"Correction mode · {mode}"
        if workbench._language.lower().startswith("en")
        else f"校正模式 · {mode}"
    )
    workbench.qspace.setFocus(QtCore.Qt.FocusReason.OtherFocusReason)


def open_numeric_edit_dialog(workbench: Any) -> None:
    if not QT_AVAILABLE:
        return
    english = workbench._language.lower().startswith("en")
    title = "Keyboard edit" if english else "键盘编辑"
    dialog = QtWidgets.QDialog(workbench)
    dialog.setObjectName("butterflyNumericDialog")
    dialog.setWindowTitle(title)
    form = QtWidgets.QFormLayout(dialog)
    mode = QtWidgets.QComboBox(dialog)
    mode.setObjectName("butterflyNumericMode")
    mode.addItem("Seed point" if english else "种子点", "seed")
    mode.addItem("Exclude polygon" if english else "排除多边形", "exclude_polygon")
    mode.addItem("Include polygon" if english else "包含多边形", "include_polygon")
    mode.addItem("Exclude rectangle" if english else "排除矩形", "exclude_rectangle")
    mode.addItem("Include rectangle" if english else "包含矩形", "include_rectangle")
    form.addRow("Mode" if english else "模式", mode)
    qx_edit = QtWidgets.QLineEdit(dialog)
    qy_edit = QtWidgets.QLineEdit(dialog)
    qx_edit.setObjectName("butterflyNumericQx")
    qy_edit.setObjectName("butterflyNumericQy")
    form.addRow("qx", qx_edit)
    form.addRow("qy", qy_edit)
    points_edit = QtWidgets.QLineEdit(dialog)
    points_edit.setObjectName("butterflyNumericPoints")
    points_edit.setPlaceholderText("qx,qy; qx,qy; …")
    form.addRow("Points" if english else "顶点", points_edit)
    buttons = QtWidgets.QDialogButtonBox(
        QtWidgets.QDialogButtonBox.StandardButton.Ok
        | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
        parent=dialog,
    )
    form.addRow(buttons)
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).setObjectName(
        "butterflyNumericApply"
    )
    buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Cancel).setObjectName(
        "butterflyNumericCancel"
    )
    from .help import apply_help

    apply_help(dialog, workbench._language)
    if dialog.exec() != int(QtWidgets.QDialog.DialogCode.Accepted):
        return
    edit, error = parse_numeric_edit_inputs(
        str(mode.currentData()),
        qx_edit.text(),
        qy_edit.text(),
        points_edit.text(),
        english=english,
    )
    if error is not None or edit is None:
        QtWidgets.QMessageBox.warning(workbench, title, error or "Invalid edit")
        return
    workbench._on_edit_requested(edit)


def snapshot_workbench_state(workbench: Any) -> dict[str, Any]:
    """Capture visual/edit/result state for atomic project rollback."""

    return {
        "frame_data": dict(workbench._frame_data),
        "current_frame": workbench._current_frame,
        "edits": deepcopy(workbench._edits),
        "redo_edits": deepcopy(workbench._redo_edits),
        "result": deepcopy(workbench._result),
        "result_fresh": bool(workbench._result_fresh),
        "requested_q_window": workbench._requested_q_window,
        "legacy_method": workbench._legacy_method,
        "page_status_state": workbench._page_status_state,
        "page_status_kind": workbench._page_status_kind,
        "page_status_error": workbench._page_status_error,
    }


def restore_workbench_state(workbench: Any, snapshot: Mapping[str, Any]) -> None:
    """Restore frame data, edits, and derived results after a failed load."""

    if not isinstance(snapshot, Mapping):
        return
    frame_data = snapshot.get("frame_data")
    if isinstance(frame_data, Mapping):
        workbench.set_data(
            frame_data.get("observed"),
            qx=frame_data.get("qx"),
            qy=frame_data.get("qy"),
            valid_mask=frame_data.get("valid_mask"),
            q_unit=frame_data.get("q_unit"),
            source=snapshot.get("current_frame"),
        )
    workbench._edits = deepcopy(list(snapshot.get("edits") or []))
    workbench._redo_edits = deepcopy(list(snapshot.get("redo_edits") or []))
    workbench._settings["edits"] = deepcopy(workbench._edits)
    workbench.qspace.set_edits(workbench._edits)
    workbench._update_edit_buttons()
    workbench.set_q_window(snapshot.get("requested_q_window"))
    legacy_method = snapshot.get("legacy_method")
    workbench._legacy_method = None if not legacy_method else str(legacy_method)
    workbench._render_legacy_banner()
    saved_result = snapshot.get("result")
    if saved_result and snapshot.get("result_fresh"):
        workbench.set_result(deepcopy(saved_result))
    else:
        workbench._page_status_state = str(
            snapshot.get("page_status_state")
            or ("ready" if workbench._has_loaded_data else "empty")
        )
        workbench._page_status_kind = str(snapshot.get("page_status_kind") or "")
        workbench._page_status_error = snapshot.get("page_status_error")
        workbench._render_page_status()
    workbench._sync_action_state()


def undo_edit(workbench: Any) -> None:
    if not workbench._edits:
        return
    workbench._redo_edits.append(workbench._edits.pop())
    workbench._settings["edits"] = deepcopy(workbench._edits)
    workbench.qspace.set_edits(workbench._edits)
    workbench.clear_result()
    workbench.editChanged.emit(workbench.butterfly_settings)
    workbench._update_edit_buttons()


def redo_edit(workbench: Any) -> None:
    if not workbench._redo_edits:
        return
    workbench._edits.append(workbench._redo_edits.pop())
    workbench._settings["edits"] = deepcopy(workbench._edits)
    workbench.qspace.set_edits(workbench._edits)
    workbench.clear_result()
    workbench.editChanged.emit(workbench.butterfly_settings)
    workbench._update_edit_buttons()


def update_edit_buttons(workbench: Any) -> None:
    workbench.undo_button.setEnabled(bool(workbench._edits))
    workbench.redo_button.setEnabled(bool(workbench._redo_edits))


__all__ = [
    "apply_edit_request",
    "exclude_selected_point",
    "open_numeric_edit_dialog",
    "parse_numeric_edit_inputs",
    "redo_edit",
    "restore_workbench_state",
    "snapshot_workbench_state",
    "start_correction",
    "undo_edit",
    "update_edit_buttons",
]
