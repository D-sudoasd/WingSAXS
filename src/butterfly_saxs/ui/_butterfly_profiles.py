"""Diagnostic and peak profile widgets/renderers for the butterfly workbench."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any

from ._common import _read
from .i18n import translate
from .qspace import _point_is_accepted
from .qt_compat import QT_AVAILABLE, QtCore, QtWidgets, require_qt

try:
    import numpy as _np
except Exception:  # pragma: no cover - numpy is a core dependency normally
    _np = None

try:
    import pyqtgraph as _pg
except Exception:  # pragma: no cover - optional plotting dependency
    _pg = None

_TRACE_METHOD_RADIAL_SECTOR = "radial_sector"
_TRACE_METHOD_ANNULAR_PEAK = "annular_peak"


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _fmt(value: Any) -> str:
    if value is None or value == "":
        return "—"
    if isinstance(value, float):
        return f"{value:.6g}" if math.isfinite(value) else "—"
    return str(value)


def extract_profile_series(
    profile: Mapping[str, Any],
    x_names: tuple[str, ...],
    y_names: tuple[str, ...],
) -> tuple[list[Any], list[Any]]:
    x = _read(profile, x_names, [])
    y = _read(profile, y_names, [])
    if isinstance(x, Mapping):
        x = _read(x, ("values", "data"), [])
    if isinstance(y, Mapping):
        y = _read(y, ("values", "data"), [])
    try:
        return ([] if x is None else list(x)), ([] if y is None else list(y))
    except TypeError:
        return [], []


def sector_failure_code(
    point: Mapping[str, Any] | None,
    profile: Mapping[str, Any] | None = None,
) -> str:
    source = point if isinstance(point, Mapping) else {}
    profile_map = profile if isinstance(profile, Mapping) else {}
    reason = _read(
        source,
        ("failure_reason", "reason", "status"),
        _read(profile_map, ("failure_reason", "reason", "status"), None),
    )
    if reason not in (None, ""):
        return str(reason)
    if not bool(_read(source, ("accepted", "valid"), True)):
        return "rejected"
    return "supported"


def sector_failure_text(
    language: str,
    point: Mapping[str, Any] | None,
    profile: Mapping[str, Any] | None = None,
) -> str:
    english = str(language).lower().startswith("en")
    source = point if isinstance(point, Mapping) else {}
    profile_map = profile if isinstance(profile, Mapping) else {}
    reason = _read(
        source,
        ("failure_reason", "reason", "status"),
        _read(profile_map, ("failure_reason", "reason", "status"), None),
    )
    raw_reason = str(reason or "").strip().lower()
    translated = {
        "selected": ("located" if english else "已定位"),
        "no_peak": ("no peak located" if english else "未找到主峰"),
        "ambiguous": ("ambiguous peaks" if english else "峰不唯一"),
        "ambiguous_multiple_peaks": ("ambiguous peaks" if english else "峰不唯一"),
        "low_coverage": ("insufficient coverage" if english else "覆盖不足"),
        "insufficient_coverage": ("insufficient coverage" if english else "覆盖不足"),
        "excluded_point_edit": ("manually excluded" if english else "手动排除"),
    }
    if raw_reason:
        return translated.get(raw_reason, str(reason))
    if not bool(_read(source, ("accepted", "valid"), True)):
        return "rejected" if english else "未通过"
    return "supported" if english else "可用"


def radial_profile_title(
    workbench: Any,
    _point_id: str,
    point: Mapping[str, Any] | None,
    profile: Mapping[str, Any] | None,
) -> str:
    source = point if isinstance(point, Mapping) else {}
    profile_map = profile if isinstance(profile, Mapping) else {}
    center = _read(
        source,
        ("sector_center_deg", "chi_deg", "angular_peak_deg"),
        _read(profile_map, ("sector_center_deg", "chi_deg"), None),
    )
    width = _read(
        source,
        ("sector_width_deg",),
        _read(profile_map, ("sector_width_deg",), workbench._settings.get("sector_width_deg")),
    )
    selected_q = _read(
        profile_map,
        ("selected_peak_q",),
        _read(source, ("q_star", "selected_peak_q", "q"), None),
    )
    reason = sector_failure_text(workbench._language, source, profile_map)
    if workbench._language.lower().startswith("en"):
        return (
            f"Radial sector I(q) · χ={_fmt(center)}° · "
            f"width={_fmt(width)}° · q*={_fmt(selected_q)} · {reason}"
        )
    return (
        f"扇区积分 I(q) · χ={_fmt(center)}° · "
        f"宽度={_fmt(width)}° · q*={_fmt(selected_q)} · {reason}"
    )


def radial_profile_tooltip(
    point_id: str,
    point: Mapping[str, Any] | None,
    profile: Mapping[str, Any] | None,
    title: str,
) -> str:
    source = point if isinstance(point, Mapping) else {}
    profile_map = profile if isinstance(profile, Mapping) else {}
    source_method = _read(
        source,
        ("source_method",),
        _read(profile_map, ("source_method",), _TRACE_METHOD_RADIAL_SECTOR),
    )
    reason = sector_failure_code(source, profile_map)
    return f"{title}\npoint_id={point_id}\nsource_method={source_method}\nreason={reason}"


def annular_profile_title(
    workbench: Any,
    _profile_id: str,
    annulus: Mapping[str, Any] | None,
    profile: Mapping[str, Any] | None,
) -> str:
    source = annulus if isinstance(annulus, Mapping) else {}
    profile_map = profile if isinstance(profile, Mapping) else {}
    q_center = _read(source, ("q_center",), _read(profile_map, ("q_center",), None))
    q_min = _read(source, ("q_min",), _read(profile_map, ("q_min",), None))
    q_max = _read(source, ("q_max",), _read(profile_map, ("q_max",), None))
    peaks = _read(source, ("selected_peaks",), _read(profile_map, ("peak_angles_deg",), ()))
    peak_count = (
        len(peaks)
        if isinstance(peaks, Sequence) and not isinstance(peaks, (str, bytes))
        else 0
    )
    reason = sector_failure_text(workbench._language, source, profile_map)
    q_unit = str(_read(profile_map, ("q_unit",), _read(source, ("q_unit",), "q")) or "q")
    if q_min not in (None, "") and q_max not in (None, ""):
        q_text = f"q=[{_fmt(q_min)}, {_fmt(q_max)}]"
    else:
        q_text = f"q={_fmt(q_center)}"
    if workbench._language.lower().startswith("en"):
        return f"Annular I(χ) · {q_text} {q_unit} · {peak_count} peaks · {reason}"
    readable_reason = {
        "four_observed_lobes": "四瓣均有支撑",
        "partial_or_missing_lobe_support": "部分花瓣未形成连续轨迹",
    }.get(reason, reason)
    return f"q 环积分 I(χ) · {q_text} {q_unit} · {peak_count} 个峰 · {readable_reason}"


def annular_profile_tooltip(
    profile_id: str,
    annulus: Mapping[str, Any] | None,
    profile: Mapping[str, Any] | None,
    title: str,
) -> str:
    source = annulus if isinstance(annulus, Mapping) else {}
    profile_map = profile if isinstance(profile, Mapping) else {}
    q_center = _read(source, ("q_center",), _read(profile_map, ("q_center",), None))
    q_min = _read(source, ("q_min",), _read(profile_map, ("q_min",), None))
    q_max = _read(source, ("q_max",), _read(profile_map, ("q_max",), None))
    status = _read(source, ("status", "reason"), _read(profile_map, ("status", "reason"), ""))
    return (
        f"{title}\nprofile_id={profile_id}\n"
        f"q_center={_fmt(q_center)}\nq_range=[{_fmt(q_min)}, {_fmt(q_max)}]\n"
        f"status={status or '—'}\n"
        "Each ring is an angular I(χ) profile; selected peaks are observed candidates."
    )


def render_peak_profiles(
    workbench: Any,
    record: Mapping[str, Any],
    landmark_id: str,
) -> None:
    profiles = _read(workbench._peak_landmarks, ("profiles",), {})
    if not isinstance(profiles, Mapping):
        profiles = {}
    angular = _read(profiles, ("angular",), {})
    radial = _read(profiles, ("radial",), {})
    if not isinstance(angular, Mapping):
        angular = {}
    if not isinstance(radial, Mapping):
        radial = {}
    angle = _finite(_read(record, ("angular_peak_deg", "chi_deg"), None))
    qx = _finite(_read(record, ("qx",), None))
    qy = _finite(_read(record, ("qy",), None))
    if angle is None and qx is not None and qy is not None:
        angle = math.degrees(math.atan2(qy, qx))
    radius = _finite(_read(record, ("q",), None))
    if radius is None and qx is not None and qy is not None:
        radius = math.hypot(qx, qy)
    angular_x, angular_raw = extract_profile_series(angular, ("angle_deg",), ("intensity_raw",))
    _, angular_smoothed = extract_profile_series(angular, ("angle_deg",), ("intensity_smoothed",))
    _, angular_reference = extract_profile_series(
        angular, ("angle_deg",), ("intensity_isotropic_reference",)
    )
    _, angular_detection = extract_profile_series(angular, ("angle_deg",), ("intensity_detection",))
    radial_x, radial_raw = extract_profile_series(radial, ("q",), ("mean_intensity_raw",))
    _, radial_smoothed = extract_profile_series(radial, ("q",), ("mean_intensity_smoothed",))
    angular_count = min(
        len(angular_x),
        len(angular_raw),
        len(angular_smoothed) if angular_smoothed else len(angular_x),
    )
    radial_count = min(
        len(radial_x),
        len(radial_raw),
        len(radial_smoothed) if radial_smoothed else len(radial_x),
    )
    english = workbench._language.lower().startswith("en")
    if angular_count:
        angular_series = {"raw": angular_raw[:angular_count]}
        if angular_smoothed:
            angular_series["smoothed"] = angular_smoothed[:angular_count]
        if angular_reference:
            angular_series["isotropic_reference"] = (
                angular_reference[:angular_count]
                + [math.nan] * max(0, angular_count - len(angular_reference))
            )
        if angular_detection:
            angular_series["detection"] = (
                angular_detection[:angular_count]
                + [math.nan] * max(0, angular_count - len(angular_detection))
            )
        workbench.peak_angular_profile._title = "Angular peak signal"
        workbench.peak_angular_profile.title_label.setText(
            workbench._tr("profile.angular_selected", id=landmark_id)
        )
        workbench.peak_angular_profile.set_series(
            angular_x[:angular_count],
            angular_series,
            x_label="chi (deg)" if english else "方位角 chi（deg）",
            y_label="intensity" if english else "强度",
            markers=[] if angle is None else [(angle, landmark_id)],
        )
    else:
        workbench.peak_angular_profile.clear(message_key="landmark.no_profiles")
    if radial_count:
        radial_series = {"raw": radial_raw[:radial_count]}
        if radial_smoothed:
            radial_series["smoothed"] = radial_smoothed[:radial_count]
        workbench.peak_radial_profile._title = "Radial peak signal"
        workbench.peak_radial_profile.title_label.setText(
            workbench._tr("profile.radial_selected", id=landmark_id)
        )
        q_unit = workbench._peak_landmarks.get("q_unit") or "q"
        workbench.peak_radial_profile.set_series(
            radial_x[:radial_count],
            radial_series,
            x_label=f"q ({q_unit})" if english else f"q（{q_unit}）",
            y_label="intensity" if english else "强度",
            markers=[] if radius is None else [(radius, landmark_id)],
        )
    else:
        workbench.peak_radial_profile.clear(message_key="landmark.no_profiles")


def render_annular_profile(
    workbench: Any,
    profile_id: str,
    annulus: Mapping[str, Any] | None,
    profile: Mapping[str, Any] | None,
) -> None:
    english = workbench._language.lower().startswith("en")
    title = workbench._annular_profile_title(profile_id, annulus, profile)
    workbench.normal_profile._title = "Annular I(χ)"
    workbench.normal_profile.title_label.setText(title)
    workbench.normal_profile.title_label.setToolTip(
        workbench._annular_profile_tooltip(profile_id, annulus, profile, title)
    )
    workbench._selected_ellipse_point = {}
    if not isinstance(profile, Mapping):
        workbench.normal_profile.clear(
            message=(
                "No angular annulus profile; see the ring reason."
                if english
                else "暂无 q 环方位剖面；请查看该环的状态原因。"
            )
        )
        workbench.ellipse_diagnostic.clear(
            "No ellipse-local diagnostic in annular mode"
            if english
            else "q 环轨迹模式不提供椭圆局部诊断"
        )
        return
    x, raw = extract_profile_series(
        profile, ("angle_deg", "chi_deg", "chi", "x"), ("raw_intensity", "raw", "intensity")
    )
    _, smoothed = extract_profile_series(
        profile, ("angle_deg", "chi_deg", "chi", "x"), ("smoothed_intensity", "smoothed")
    )
    _, counts = extract_profile_series(
        profile, ("angle_deg", "chi_deg", "chi", "x"), ("counts", "valid_counts")
    )
    _, coverage = extract_profile_series(
        profile, ("angle_deg", "chi_deg", "chi", "x"), ("coverage",)
    )
    series = {
        name: values
        for name, values in (
            ("raw", raw),
            ("smoothed", smoothed),
            ("counts", counts),
            ("coverage", coverage),
        )
        if values
    }
    marker_values: list[float] = []
    peak_angles = _read(profile, ("peak_angles_deg",), None)
    if isinstance(peak_angles, Sequence) and not isinstance(peak_angles, (str, bytes)):
        marker_values.extend(
            value for value in (_finite(item) for item in peak_angles) if value is not None
        )
    if not marker_values and isinstance(annulus, Mapping):
        selected = _read(annulus, ("selected_peaks",), ())
        if isinstance(selected, Sequence) and not isinstance(selected, (str, bytes)):
            marker_values.extend(
                value
                for value in (
                    _finite(_read(item, ("chi_deg", "angle_deg", "chi"), None))
                    if isinstance(item, Mapping)
                    else _finite(item)
                    for item in selected
                )
                if value is not None
            )
    markers = [(value, f"χ={value:g}°") for value in marker_values]
    if x and series:
        workbench.normal_profile.set_series(
            x,
            series,
            x_label="χ (deg)" if english else "χ（deg）",
            y_label="I(χ)" if english else "I(χ) 强度",
            markers=markers,
            plot_names=("raw", "smoothed"),
        )
        workbench.normal_profile.title_label.setText(title)
        workbench.normal_profile.title_label.setToolTip(
            workbench._annular_profile_tooltip(profile_id, annulus, profile, title)
        )
        if workbench.normal_profile.plot is not None:
            workbench.normal_profile.plot.getAxis("bottom").enableAutoSIPrefix(False)
            workbench.normal_profile.plot.setAccessibleDescription(
                "Angular I(χ) profile; raw and locator-only smoothed intensity; vertical lines mark selected lobe peaks"
                if english
                else "方位 I(χ) 剖面；显示原始强度和仅用于定位的平滑强度；竖线标记已选瓣峰"
            )
    else:
        workbench.normal_profile.clear(
            message=(
                "Angular annulus profile is empty; see the ring reason."
                if english
                else "q 环方位剖面为空；请查看该环的状态原因。"
            )
        )
        workbench.normal_profile.title_label.setText(title)
        workbench.normal_profile.title_label.setToolTip(
            workbench._annular_profile_tooltip(profile_id, annulus, profile, title)
        )
    workbench.ellipse_diagnostic.clear(
        "No ellipse-local diagnostic in annular mode"
        if english
        else "q 环轨迹模式不提供椭圆局部诊断"
    )


def render_radial_profile(
    workbench: Any,
    point_id: str,
    point: Mapping[str, Any] | None,
    profile: Mapping[str, Any] | None,
) -> None:
    english = workbench._language.lower().startswith("en")
    title = workbench._radial_profile_title(point_id, point, profile)
    workbench.normal_profile._title = "Radial sector I(q)"
    workbench.normal_profile.title_label.setText(title)
    workbench.normal_profile.title_label.setToolTip(
        workbench._radial_profile_tooltip(point_id, point, profile, title)
    )
    workbench._selected_ellipse_point = dict(point or {})
    if not isinstance(profile, Mapping):
        workbench.normal_profile.clear(
            message=(
                "No radial sector profile; see the point reason."
                if english
                else "暂无扇区径向剖面；请查看该点的失败原因。"
            )
        )
        workbench.ellipse_diagnostic.clear(
            "No ellipse-local diagnostic in radial sector mode"
            if english
            else "扇区积分模式不提供椭圆局部诊断"
        )
        return
    x, raw = extract_profile_series(profile, ("q",), ("raw_intensity",))
    _, smoothed = extract_profile_series(profile, ("q",), ("smoothed_intensity",))
    _, counts = extract_profile_series(profile, ("q",), ("counts", "valid_counts"))
    _, coverage = extract_profile_series(profile, ("q",), ("coverage",))
    series = {
        name: values
        for name, values in (
            ("raw", raw),
            ("smoothed", smoothed),
            ("counts", counts),
            ("coverage", coverage),
        )
        if values
    }
    selected_q = _finite(_read(profile, ("selected_peak_q",), None))
    if selected_q is None:
        selected_q = _finite(_read(point, ("q_star", "selected_peak_q"), None))
    q_unit = str(_read(profile, ("q_unit",), None) or workbench._frame_data.get("q_unit") or "q")
    if x and series:
        workbench.normal_profile.set_series(
            x,
            series,
            x_label=(f"q ({q_unit})" if english else f"q（{q_unit}）"),
            y_label="I(q)" if english else "I(q) 强度",
            markers=[] if selected_q is None else [(selected_q, "q*")],
            plot_names=("raw", "smoothed"),
        )
        workbench.normal_profile.title_label.setText(title)
        workbench.normal_profile.title_label.setToolTip(
            workbench._radial_profile_tooltip(point_id, point, profile, title)
        )
        if workbench.normal_profile.plot is not None:
            workbench.normal_profile.plot.getAxis("bottom").enableAutoSIPrefix(False)
            workbench.normal_profile.plot.setAccessibleDescription(
                "Radial sector I(q); smoothed curve is only used for peak localization"
                if english
                else "扇区径向 I(q)；平滑曲线仅用于定位主峰"
            )
    else:
        workbench.normal_profile.clear(
            message=(
                "Radial sector profile is empty; see the point reason."
                if english
                else "扇区径向剖面为空；请查看该点的失败原因。"
            )
        )
        workbench.normal_profile.title_label.setText(title)
        workbench.normal_profile.title_label.setToolTip(
            workbench._radial_profile_tooltip(point_id, point, profile, title)
        )
    workbench.ellipse_diagnostic.clear(
        "No ellipse-local diagnostic in radial sector mode"
        if english
        else "扇区积分模式不提供椭圆局部诊断"
    )


def render_profile(
    workbench: Any,
    point_id: str,
    point: Mapping[str, Any] | None = None,
) -> None:
    workbench._selected_profile_point_id = str(point_id)
    profile = workbench._profiles.get(point_id)
    if profile is None:
        profile = workbench._profiles.get(str(point_id))
    profile_axis = str(_read(profile, ("profile_axis",), "") or "").strip().lower()
    is_annular = profile_axis in {"azimuthal", "angular", "chi"} or (
        workbench._trace_method() == _TRACE_METHOD_ANNULAR_PEAK
        and (
            profile is None
            or (
                isinstance(profile, Mapping)
                and any(key in profile for key in ("angle_deg", "chi_deg", "peak_angles_deg"))
            )
        )
    )
    if is_annular:
        workbench._render_annular_profile(point_id, point, profile)
        return
    is_radial = profile_axis == "radial" or (
        workbench._trace_method() == _TRACE_METHOD_RADIAL_SECTOR
        and (
            profile is None
            or (
                isinstance(profile, Mapping)
                and "q" in profile
                and "raw_intensity" in profile
            )
        )
    )
    if is_radial:
        workbench._render_radial_profile(point_id, point, profile)
        return
    if not isinstance(profile, Mapping):
        workbench.normal_profile.clear()
        workbench.ellipse_diagnostic.clear()
        return
    x, raw = extract_profile_series(
        profile, ("offset_q", "q_offset", "normal_q", "x"), ("raw", "raw_intensity", "intensity", "observed")
    )
    _, fit = extract_profile_series(
        profile, ("offset_q", "q_offset", "normal_q", "x"), ("fit", "fit_intensity", "fitted", "model")
    )
    _, residual = extract_profile_series(
        profile, ("offset_q", "q_offset", "normal_q", "x"), ("residual", "resid", "fit_residual")
    )
    series = {
        name: values
        for name, values in (("raw", raw), ("fit", fit), ("residual", residual))
        if values
    }
    if x and series:
        workbench.normal_profile.set_series(
            x, series, x_label="normal q offset", y_label="intensity"
        )
    else:
        workbench.normal_profile.clear()
    workbench._selected_ellipse_point = dict(point or {})
    workbench._render_ellipse_local()


def render_ellipse_local(workbench: Any) -> None:
    bundle = workbench._ellipse_local
    english = workbench._language.lower().startswith("en")
    if not isinstance(bundle, Mapping) or not bundle:
        workbench.ellipse_diagnostic.clear(
            "No ellipse-local diagnostic" if english else "暂无椭圆局部诊断"
        )
        return
    selected = getattr(workbench, "_selected_ellipse_point", {})
    source_branch = _read(selected, ("branch_id", "source_branch", "branch"), None)
    side = str(_read(selected, ("side",), "") or "").lower()
    if "points" not in bundle and "curves" not in bundle:
        branch_bundle = None
        if source_branch is not None:
            branch_bundle = bundle.get(str(source_branch), bundle.get(source_branch))
        if branch_bundle is None and side:
            branch_bundle = bundle.get(side)
        if isinstance(branch_bundle, Mapping):
            bundle = branch_bundle
    point_rows = _read(bundle, ("points",), []) or []
    curves = _read(bundle, ("curves",), {}) or {}
    selected_points = [
        row
        for row in point_rows
        if isinstance(row, Mapping)
        and (
            source_branch is None
            or str(row.get("branch_id", row.get("source_branch", source_branch)))
            == str(source_branch)
        )
        and (not side or str(row.get("side", "")).lower() == side)
    ]
    if not selected_points:
        selected_points = [row for row in point_rows if isinstance(row, Mapping)]
    workbench.ellipse_diagnostic.set_uv_series(
        selected_points,
        curves if isinstance(curves, Mapping) else {},
        v_scale=workbench._uv_display_magnification,
        x_label="u",
        y_label=(
            f"v ×{workbench._uv_display_magnification:g}"
            if english
            else f"v ×{workbench._uv_display_magnification:g}（显示）"
        ),
    )
    workbench._diagnostic_magnification = float(workbench._uv_display_magnification)
    workbench._render_magnification_label()


if QT_AVAILABLE:

    class _ProfilePanel(QtWidgets.QWidget):
        """Small plot/table fallback used below the q-space canvas."""

        def __init__(
            self,
            title: str,
            parent: Any = None,
            *,
            data_view: bool = True,
        ) -> None:
            super().__init__(parent)
            self._title = str(title)
            self._english = True
            self._empty_message_key = "profile.empty.select"
            self._x_label: str | None = None
            self._y_label: str | None = None
            self._series_names: tuple[str, ...] = ()
            self.setObjectName(title.replace(" ", "") + "Panel")
            layout = QtWidgets.QVBoxLayout(self)
            layout.setContentsMargins(4, 4, 4, 4)
            self.title_label = QtWidgets.QLabel(title, self)
            self.title_label.setStyleSheet("font-weight: 600;")
            self.title_label.setWordWrap(True)
            self.title_label.setMaximumHeight(32)
            layout.addWidget(self.title_label)
            self.plot = None
            self.table = QtWidgets.QTableWidget(0, 1, self)
            self.table.setHorizontalHeaderLabels((translate("en", "profile.column.coordinate"),))
            self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
            self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectItems)
            self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
            self.table.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
            self.table.horizontalHeader().setStretchLastSection(True)
            self.table.setMinimumHeight(112)
            self.table.setAccessibleName(self._title)
            self.table.setAccessibleDescription(translate("en", "a11y.profile_data_empty"))
            self.view_tabs = None
            if _pg is not None:
                self.plot = _pg.PlotWidget(self)
                self.plot.setProperty("plotTheme", "dark")
                self.plot.setBackground("#16181e")
                self.plot.showGrid(x=True, y=True, alpha=0.18)
                self.plot.setMinimumHeight(112)
                self.plot.setAccessibleName(self._title)
                self.plot.setAccessibleDescription(
                    f"{self._title}; measured and fitted diagnostic series"
                )
                self.view_tabs = QtWidgets.QTabWidget(self)
                self.view_tabs.setObjectName("profileDataViews")
                self.view_tabs.setAccessibleName(translate("en", "a11y.profile_tabs"))
                self.view_tabs.addTab(self.plot, translate("en", "profile.view.plot"))
                if data_view:
                    self.view_tabs.addTab(self.table, translate("en", "profile.view.data"))
                self.view_tabs.setCurrentWidget(self.plot)
                layout.addWidget(self.view_tabs, 1)
            else:
                layout.addWidget(self.table, 1)
            self.empty_label = QtWidgets.QLabel("Select a measured point", self)
            self.empty_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            self.empty_label.setStyleSheet("color: #8c95a5;")
            layout.addWidget(self.empty_label)

        def clear(
            self,
            message: str | None = None,
            *,
            message_key: str | None = None,
        ) -> None:
            if self.plot is not None:
                self.plot.clear()
            self.table.setRowCount(0)
            self.table.setColumnCount(1)
            language = "en" if self._english else "zh_CN"
            self.table.setHorizontalHeaderLabels(
                (translate(language, "profile.column.coordinate"),)
            )
            self._x_label = None
            self._y_label = None
            self._series_names = ()
            self._empty_message_key = message_key
            if message_key is not None:
                message = translate(language, message_key)
            elif message is None:
                self._empty_message_key = "profile.empty.select"
                message = translate(language, self._empty_message_key)
            self.empty_label.setText(message or "")
            self.empty_label.setVisible(True)
            self._update_data_accessibility()

        def set_language(self, *, english: bool) -> None:
            """Refresh visible and assistive text without changing data."""

            self._english = bool(english)
            language = "en" if english else "zh_CN"
            title = self._title
            if not english:
                title = {
                    "Angular peak signal": "方位峰信号",
                    "Radial peak signal": "径向峰信号",
                    "Radial sector I(q)": "扇区积分 I(q)",
                    "Annular I(χ)": "q 环积分 I(χ)",
                    "Normal profile · raw / fit / residual": "法向剖面 · 原始 / 拟合 / 残差",
                    "Ellipse-local u/v · narrow-axis": "椭圆局部 u/v · 短轴诊断",
                }.get(title, title)
            self.title_label.setText(title)
            if self._empty_message_key is not None:
                self.empty_label.setText(translate(language, self._empty_message_key))
            if self.plot is not None:
                self.plot.setAccessibleName(title)
                self.plot.setAccessibleDescription(
                    f"{title}; measured and fitted diagnostic series"
                    if english
                    else f"{title}；显示测量与拟合诊断序列"
                )
            if self.view_tabs is not None:
                self.view_tabs.setAccessibleName(translate(language, "a11y.profile_tabs"))
                self.view_tabs.setTabText(0, translate(language, "profile.view.plot"))
                if self.view_tabs.count() > 1:
                    self.view_tabs.setTabText(1, translate(language, "profile.view.data"))
            self._update_data_accessibility()

        def _display_series_names(self) -> list[str]:
            return [self._display_series_name(name) for name in self._series_names]

        def _display_series_name(self, name: str) -> str:
            if name == "smoothed" and self._title in {
                "Radial sector I(q)",
                "Annular I(χ)",
            }:
                return "smoothed (locator only)" if self._english else "平滑（仅用于定位）"
            if name in {
                "raw",
                "smoothed",
                "isotropic_reference",
                "detection",
                "fit",
                "residual",
                "u",
                "v",
            }:
                language = "en" if self._english else "zh_CN"
                return translate(language, f"profile.series.{name}")
            if name == "counts":
                return "valid pixels (count)" if self._english else "有效像素数（像素）"
            if name == "coverage":
                return "coverage (dimensionless)" if self._english else "覆盖率（无量纲）"
            return name

        def _update_data_accessibility(self) -> None:
            language = "en" if self._english else "zh_CN"
            self.table.setAccessibleName(self.title_label.text())
            if self.table.rowCount() == 0 or self._x_label is None:
                self.table.setColumnCount(1)
                self.table.setHorizontalHeaderLabels(
                    (translate(language, "profile.column.coordinate"),)
                )
                description = translate(language, "a11y.profile_data_empty")
            else:
                display_names = self._display_series_names()
                self.table.setColumnCount(1 + len(display_names))
                self.table.setHorizontalHeaderLabels([self._x_label, *display_names])
                description = translate(
                    language,
                    "a11y.profile_data_table",
                    x_axis=self._x_label,
                    y_axis=self._y_label or "",
                    series=", ".join(display_names),
                )
            self.table.setAccessibleDescription(description)

        def set_series(
            self,
            x: Sequence[Any],
            series: Mapping[str, Sequence[Any]],
            *,
            x_label: str = "q offset",
            y_label: str = "value",
            markers: Sequence[tuple[float, str]] = (),
            plot_names: Sequence[str] | None = None,
        ) -> None:
            values: Any = []
            if _np is not None:
                try:
                    values = _np.asarray(x, dtype=float)
                except (TypeError, ValueError):
                    values = []
            names = list(series)
            x_values = list(values) if hasattr(values, "__len__") and len(values) else list(x)
            self._x_label = str(x_label)
            self._y_label = str(y_label)
            self._series_names = tuple(str(name) for name in names)
            self.table.setColumnCount(1 + len(names))
            self.table.setRowCount(len(x_values))
            for row, value in enumerate(x_values):
                self.table.setItem(row, 0, QtWidgets.QTableWidgetItem(_fmt(value)))
                for col, name in enumerate(names, 1):
                    data = series[name]
                    cell = data[row] if row < len(data) else None
                    self.table.setItem(row, col, QtWidgets.QTableWidgetItem(_fmt(cell)))
            self.table.resizeColumnsToContents()
            self._update_data_accessibility()
            if self.plot is not None:
                self.plot.clear()
                colors = {
                    "raw": (220, 230, 238),
                    "smoothed": (255, 166, 76),
                    "isotropic_reference": (155, 139, 232),
                    "detection": (255, 145, 106),
                    "fit": (72, 190, 242),
                    "residual": (243, 157, 73),
                    "u": (72, 190, 242),
                    "v": (243, 157, 73),
                }
                plot_name_set = None if plot_names is None else {str(name) for name in plot_names}
                for name, y in series.items():
                    if not name:
                        continue
                    if plot_name_set is not None and str(name) not in plot_name_set:
                        continue
                    try:
                        pen_style = (
                            QtCore.Qt.PenStyle.DashLine
                            if str(name) == "smoothed"
                            else QtCore.Qt.PenStyle.SolidLine
                        )
                        self.plot.plot(
                            values,
                            _np.asarray(y, dtype=float) if _np is not None else list(y),
                            pen=_pg.mkPen(
                                colors.get(name, (190, 190, 200)),
                                width=2,
                                style=pen_style,
                            ),
                            symbol="o" if name == "raw" else None,
                            symbolSize=4,
                            name=self._display_series_name(str(name)),
                        )
                    except (TypeError, ValueError):
                        continue
                for marker_value, marker_label in markers:
                    marker_position = _finite(marker_value)
                    if marker_position is None:
                        continue
                    marker = _pg.InfiniteLine(
                        pos=marker_position,
                        angle=90,
                        movable=False,
                        pen=_pg.mkPen(
                            (255, 205, 72),
                            width=1.3,
                            style=QtCore.Qt.PenStyle.DashLine,
                        ),
                    )
                    marker.setZValue(10)
                    marker.setToolTip(str(marker_label))
                    self.plot.addItem(marker)
                self.plot.setLabel("bottom", x_label)
                self.plot.setLabel("left", y_label)
                self.empty_label.setVisible(not bool(x_values) or not bool(series))
                return
            self.empty_label.setVisible(not bool(x_values) or not bool(series))

        def set_uv_series(
            self,
            points: Sequence[Mapping[str, Any]],
            curves: Mapping[str, Mapping[str, Sequence[Any]]],
            *,
            v_scale: float = 1.0,
            x_label: str = "u",
            y_label: str = "v",
        ) -> None:
            if self.plot is not None:
                self.clear(message_key="profile.empty.select")
                self.plot.clear()
                curve_colors = {
                    "upper": (72, 190, 242),
                    "lower": (243, 157, 73),
                }
                for side, curve in curves.items():
                    if not isinstance(curve, Mapping):
                        continue
                    try:
                        u_values = _np.asarray(curve.get("u", []), dtype=float)
                        v_values = _np.asarray(curve.get("v", []), dtype=float) * float(v_scale)
                        if u_values.size and v_values.size:
                            self.plot.plot(
                                u_values,
                                v_values,
                                pen=_pg.mkPen(
                                    curve_colors.get(str(side), (190, 190, 200)), width=2
                                ),
                                name=f"{side} fit",
                            )
                    except (TypeError, ValueError):
                        continue
                grouped: dict[tuple[str, bool], tuple[list[float], list[float]]] = {}
                for point in points or ():
                    if not isinstance(point, Mapping):
                        continue
                    try:
                        side = str(point.get("side", "unknown"))
                        accepted = _point_is_accepted(point)
                        group = grouped.setdefault((side, accepted), ([], []))
                        group[0].append(float(point.get("u")))
                        group[1].append(float(point.get("v")) * float(v_scale))
                    except (TypeError, ValueError):
                        continue
                for (side, accepted), (u_values, v_values) in grouped.items():
                    if u_values:
                        color = curve_colors.get(side, (180, 180, 185))
                        self.plot.plot(
                            u_values,
                            v_values,
                            pen=None,
                            symbol=("o" if side == "upper" else "s") if accepted else "x",
                            symbolSize=6 if accepted else 8,
                            symbolPen=_pg.mkPen(
                                color if accepted else (135, 135, 145), width=1.5
                            ),
                            symbolBrush=_pg.mkBrush(color) if accepted else None,
                            name=(
                                f"{side} points"
                                if accepted
                                else f"{side} excluded candidates"
                            ),
                        )
                self.plot.setLabel("bottom", x_label)
                self.plot.setLabel("left", y_label)
                self.empty_label.setVisible(not bool(grouped) and not bool(curves))
                return
            self.clear(message_key="profile.empty.select")

else:

    class _ProfilePanel:  # pragma: no cover - exercised without optional UI dependencies
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            require_qt()


__all__ = [
    "_ProfilePanel",
    "_finite",
    "_fmt",
    "annular_profile_title",
    "annular_profile_tooltip",
    "extract_profile_series",
    "radial_profile_title",
    "radial_profile_tooltip",
    "render_annular_profile",
    "render_ellipse_local",
    "render_peak_profiles",
    "render_profile",
    "render_radial_profile",
    "sector_failure_code",
    "sector_failure_text",
]
