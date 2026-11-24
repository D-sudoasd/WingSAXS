"""Secondary measurement tabs, profile/table widgets, and batch/evolution panel renderers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import math
from typing import Any

from ..butterfly_quality import (
    classify_ellipse_publication,
    ellipse_candidate_is_limited,
    unpublished_ellipse_shape,
)
from ._batch_records import (
    _BATCH_STATUS_KEYS,
    _BATCH_TABLE_HEADERS,
    _batch_record_geometry,
    _classify_evolution_row,
    _evolution_numeric_keys,
    _flatten_evolution_record,
    _format_metric,
    _is_finite,
    _numeric,
    _result_value,
    _sequence,
)
from ._common import _read
from .i18n import translate_q_star_source
from .qt_compat import QtCore, QtWidgets
from .views import PLOT_AVAILABLE, _disable_auto_si_prefix

try:
    import numpy as _np
except Exception:  # pragma: no cover - numpy is a core dependency normally
    _np = None

if PLOT_AVAILABLE:
    try:
        import pyqtgraph as _pg
    except Exception:  # pragma: no cover - handled by text fallback
        _pg = None
else:
    _pg = None


def build_image_measurement_pages(window: Any) -> None:
    from .azimuthal_page import AzimuthalPage
    from .density2d_page import Density2DPage
    from .local_measurement_page import LocalMeasurementPage

    window.local_measurement_page = LocalMeasurementPage(window.pages, language=window._language)
    window.azimuthal_page = AzimuthalPage(window.pages, language=window._language)
    window.density2d_page = Density2DPage(window.pages, language=window._language)
    window.pages.addTab(window.local_measurement_page, "局部测量")
    window.pages.addTab(window.azimuthal_page, "方位角峰分析")
    window.pages.addTab(window.density2d_page, "低 q 区域分析")


def sync_image_measurement_pages(window: Any) -> None:
    """Supply independent image tools with current coordinates and usable pixels."""

    window._refresh_workspace_context()
    if not hasattr(window, "local_measurement_page") or _np is None:
        return
    pages = (window.local_measurement_page, window.azimuthal_page, window.density2d_page)
    if window._observed is None:
        for page in pages:
            page.invalidate()
        return
    valid = _np.isfinite(window._observed)
    detector_valid = _read(window._qmap, ("valid_mask", "valid"), None)
    if detector_valid is not None:
        valid &= _np.asarray(detector_valid, dtype=bool)
    if window._external_mask is not None:
        valid &= ~_np.asarray(window._external_mask, dtype=bool)
    qx, qy, q_unit = window._qx, window._qy, window._active_q_unit()
    if qx is None and qy is None:
        yy, xx = _np.indices(valid.shape, dtype=float)
        qx = xx - (valid.shape[1] - 1) / 2.0
        qy = yy - (valid.shape[0] - 1) / 2.0
        q_unit = "pixel-q"
    source = window._source_path or f"in-memory:{id(window._observed):x}"
    selectors = []
    if window._frame is not None:
        selectors.append(f"frame={window._frame}")
    if window._dataset is not None:
        selectors.append(f"dataset={window._dataset}")
    if selectors:
        source = f"{source or 'in-memory'} [{'; '.join(selectors)}]"
    for page in pages:
        try:
            page.set_data(
                window._observed,
                qx=qx,
                qy=qy,
                valid_mask=valid,
                q_unit=q_unit,
                source=source,
            )
        except ValueError as exc:
            page.invalidate(str(exc))


def build_measurements_page(window: Any) -> None:
    """Build the measured-profile page without making pyqtgraph required."""

    window.measurements_page = QtWidgets.QWidget(window.pages)
    root = QtWidgets.QVBoxLayout(window.measurements_page)
    root.setContentsMargins(4, 4, 4, 4)
    window.measurement_observables = None

    profile_tabs = QtWidgets.QTabWidget(window.measurements_page)
    profile_tabs.setObjectName("profileTabs")
    profile_tabs.setDocumentMode(True)
    profile_tabs.setAccessibleName(window._tr("a11y.profile_tabs"))
    window.profile_tabs = profile_tabs
    window.profile_splitter = profile_tabs
    window.angular_plot = None
    window.coverage_plot = None
    window.ridge_plot = None
    window.radial_profile_plot = None
    if _pg is not None:
        for attr, obj_name, a11y_name, a11y_desc, x_lbl, y_lbl, tab_title in (
            (
                "angular_plot",
                "angularProfilePlot",
                "a11y.profile_angular_name",
                "a11y.profile_angular_description",
                "Azimuth (deg)",
                "Angular intensity (a.u.)",
                "Angular intensity",
            ),
            (
                "coverage_plot",
                "coverageProfilePlot",
                "a11y.profile_coverage_name",
                "a11y.profile_coverage_description",
                "Azimuth (deg)",
                "Detector coverage (0–1)",
                "Coverage",
            ),
            (
                "ridge_plot",
                "ridgeProfilePlot",
                "a11y.profile_ridge_name",
                "a11y.profile_ridge_description",
                "Azimuth (deg)",
                "Ridge q (map unit)",
                "Ridge q-angle",
            ),
            (
                "radial_profile_plot",
                "radialProfilePlot",
                "a11y.profile_radial_name",
                "a11y.profile_radial_description",
                "q (map unit)",
                "Radial intensity (a.u.)",
                "Lobe radial profiles",
            ),
        ):
            plot = _pg.PlotWidget(window.measurements_page)
            plot.setObjectName(obj_name)
            plot.setAccessibleName(window._tr(a11y_name))
            plot.setAccessibleDescription(window._tr(a11y_desc))
            plot.setLabel("bottom", x_lbl)
            plot.setLabel("left", y_lbl)
            plot.showGrid(x=True, y=True, alpha=0.22)
            _disable_auto_si_prefix(plot)
            setattr(window, attr, plot)
            profile_tabs.addTab(plot, tab_title)
    else:
        for attr, obj_name, text, tab_title in (
            (
                "angular_placeholder",
                "angularProfilePlaceholder",
                "安装 pyqtgraph 后显示角向强度与 coverage",
                "Angular intensity",
            ),
            (
                "coverage_placeholder",
                "coverageProfilePlaceholder",
                "安装 pyqtgraph 后显示 detector coverage",
                "Coverage",
            ),
            (
                "ridge_placeholder",
                "ridgeProfilePlaceholder",
                "安装 pyqtgraph 后显示 ridge q-angle/accepted",
                "Ridge q-angle",
            ),
            (
                "radial_profile_placeholder",
                "radialProfilePlaceholder",
                "安装 pyqtgraph 后显示 lobe radial profiles",
                "Lobe radial profiles",
            ),
        ):
            placeholder = QtWidgets.QLabel(text)
            placeholder.setObjectName(obj_name)
            placeholder.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            setattr(window, attr, placeholder)
            profile_tabs.addTab(placeholder, tab_title)
    root.addWidget(profile_tabs, 2)
    window.profile_summary_label = QtWidgets.QLabel(
        window._tr("profile.summary_empty"),
        window.measurements_page,
    )
    window.profile_summary_label.setObjectName("profileSummaryLabel")
    window.profile_summary_label.setWordWrap(True)
    window.profile_summary_label.setAccessibleName(window._tr("a11y.profile_tabs"))
    window.profile_summary_label.setAccessibleDescription(window._tr("profile.summary_empty"))
    window._set_profile_summary(window._tr("profile.summary_empty"))
    root.addWidget(window.profile_summary_label)

    lower = QtWidgets.QTabWidget(window.measurements_page)
    lower.setObjectName("measurementTableTabs")
    lower.setDocumentMode(True)
    window.measurement_table_tabs = lower
    for label_attr, label_text, table_attr, table_obj_name, headers, tab_title in (
        (
            "lobe_panel_label",
            "Four-lobe measurements",
            "lobe_table",
            "lobeTable",
            [
                "Angle (deg)",
                "Intensity",
                "Baseline",
                "SNR",
                "FWHM (deg)",
                "Coverage",
                "Valid",
                "Flags",
            ],
            "Four-lobe measurements",
        ),
        (
            "ridge_panel_label",
            "Ridge q vs angle / accepted",
            "ridge_table",
            "ridgeTable",
            ["Angle (deg)", "q", "Accepted", "Method"],
            "Ridge",
        ),
        (
            "ellipse_panel_label",
            "Ellipse core quantities / quality",
            "ellipse_table",
            "ellipseTable",
            ["Quantity", "Value"],
            "Ellipse",
        ),
        (
            "radial_panel_label",
            "Lobe radial q-star / spacing",
            "radial_table",
            "radialTable",
            [
                "Angle (deg)",
                "q-star",
                "Ln (nm)",
                "SNR",
                "FWHM (q)",
                "Coverage",
                "Valid",
                "Flags",
            ],
            "Radial peaks",
        ),
    ):
        panel = QtWidgets.QWidget(lower)
        panel_layout = QtWidgets.QVBoxLayout(panel)
        panel_label = QtWidgets.QLabel(label_text, panel)
        panel_layout.addWidget(panel_label)
        table = QtWidgets.QTableWidget(0, len(headers), panel)
        table.setObjectName(table_obj_name)
        table.setHorizontalHeaderLabels(headers)
        table.horizontalHeader().setStretchLastSection(True)
        table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setAlternatingRowColors(True)
        table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.ResizeToContents
        )
        panel_layout.addWidget(table, 1)
        setattr(window, label_attr, panel_label)
        setattr(window, table_attr, table)
        lower.addTab(panel, tab_title)
    root.addWidget(lower, 2)
    window.pages.addTab(window.measurements_page, "Measurements / Profiles")


def fill_batch_table_row(
    window: Any,
    row_index: int,
    frame: Any,
    record: Mapping[str, Any] | None = None,
) -> None:
    mapping = record if isinstance(record, Mapping) else {}
    geometry = _batch_record_geometry(mapping)
    status = mapping.get("status", "ready" if not mapping else "ok")
    raw_status = str(status)
    status_key = _BATCH_STATUS_KEYS.get(raw_status.casefold())
    status_item = QtWidgets.QTableWidgetItem(
        window._tr(status_key) if status_key is not None else raw_status
    )
    status_item.setData(QtCore.Qt.ItemDataRole.UserRole, raw_status)
    kind = str(geometry.get("ellipse_kind") or "")
    quality = geometry.get("quality")
    if kind == "ring" and quality == "WARN":
        quality_text = window._tr("quality.warn_ring")
        quality_tip = window._tr("tooltip.batch_quality_ring")
    elif kind == "ellipse" and geometry.get("ellipse_limited"):
        if quality == "WARN":
            quality_text = window._tr("quality.warn_ellipse_candidate")
        else:
            prefix = f"{quality} · " if quality else ""
            quality_text = prefix + window._tr("quality.ellipse_candidate_limited")
        quality_tip = window._tr("tooltip.batch_quality_ellipse_candidate")
    elif kind == "ellipse" and quality == "WARN":
        quality_text = window._tr("quality.warn_ellipse")
        quality_tip = window._tr("tooltip.batch_quality_ellipse")
    else:
        quality_text = str(quality or "—")
        quality_tip = ""
    ln_candidate = geometry.get("ln_candidate")
    lz_candidate = geometry.get("lz_candidate")
    show_ln_candidate = geometry.get("ln") is None and ln_candidate is not None
    show_lz_candidate = geometry.get("lz") is None and lz_candidate is not None
    show_ln_estimate = str(geometry.get("ln_status", "")).lower() in {"estimate", "candidate"}
    show_lz_estimate = str(geometry.get("lz_status", "")).lower() in {"estimate", "candidate"}
    ln_raw = geometry.get("ln") if geometry.get("ln") is not None else ln_candidate
    lz_raw = geometry.get("lz") if geometry.get("lz") is not None else lz_candidate
    ln_display = (
        f"≈{_format_metric(ln_raw)}"
        if (show_ln_candidate or show_ln_estimate)
        else _format_metric(ln_raw)
    )
    lz_display = (
        f"≈{_format_metric(lz_raw)}"
        if (show_lz_candidate or show_lz_estimate)
        else _format_metric(lz_raw)
    )

    def shape_display(name: str) -> tuple[str, str]:
        value = geometry.get(name)
        candidate = geometry.get(f"{name}_candidate")
        shape_status = str(geometry.get(f"{name}_status", "")).lower()
        confidence = str(geometry.get(f"{name}_confidence", "")).lower()
        is_candidate = value is None and candidate is not None
        display_value = value if value is not None else candidate
        marked = is_candidate or shape_status in {"estimate", "candidate"}
        display = _format_metric(display_value)
        if marked and display_value is not None:
            display = f"≈{display}"
        tooltip = ""
        if marked and display_value is not None:
            tooltip = window._tr(
                "tooltip.batch_shape_candidate",
                status=shape_status or "candidate",
                confidence=confidence or "limited",
            )
        return display, tooltip

    a_display, a_tip = shape_display("a")
    b_display, b_tip = shape_display("b")
    ratio_display, ratio_tip = shape_display("axis_ratio")
    theta_display, theta_tip = shape_display("theta_deg")
    values = (
        str(mapping.get("frame", mapping.get("path", frame))),
        status_item,
        quality_text,
        a_display,
        b_display,
        ratio_display,
        theta_display,
        str(geometry["arcs"] or "—"),
        ln_display,
        lz_display,
        _format_metric(geometry["l_radial"]),
        _format_metric(geometry["rmse"] if geometry["rmse"] is not None else mapping.get("rmse")),
        geometry["flags"] or "—",
    )
    if geometry.get("ln_candidate") is not None or geometry.get("lz_candidate") is not None:
        ln_tip = window._tr(
            "tooltip.batch_ln_candidate",
            ln=_format_metric(geometry.get("ln_candidate")),
            lz=_format_metric(geometry.get("lz_candidate")),
            l_major=_format_metric(geometry.get("l_major_candidate")),
        )
    elif geometry["ln"] is None:
        ln_tip = window._tr("tooltip.batch_ln_empty")
    else:
        ln_tip = ""
    tips = {
        2: quality_tip,
        3: a_tip,
        4: b_tip,
        5: ratio_tip
        or (window._tr("tooltip.batch_ratio_empty") if geometry["axis_ratio"] is None else ""),
        6: theta_tip
        or (
            window._tr("tooltip.batch_theta_empty")
            if geometry["theta_deg"] is None
            and "axis_ratio_at_bound" in (geometry.get("flags") or "")
            else ""
        ),
        8: ln_tip,
        9: ln_tip if geometry["lz"] is None else "",
    }
    for column, value in enumerate(values):
        item = (
            value
            if isinstance(value, QtWidgets.QTableWidgetItem)
            else QtWidgets.QTableWidgetItem(str(value))
        )
        tip = tips.get(column)
        if tip:
            item.setToolTip(tip)
        window.batch_table.setItem(row_index, column, item)


def update_batch_rows(window: Any, records: Iterable[Any]) -> None:
    rows = list(records)
    if not rows:
        return
    window.batch_table.setColumnCount(len(_BATCH_TABLE_HEADERS))
    window.batch_table.setRowCount(len(rows))
    for row_index, record in enumerate(rows):
        mapping = record if isinstance(record, Mapping) else {"value": record}
        frame = mapping.get("frame", mapping.get("path", row_index))
        fill_batch_table_row(window, row_index, frame, mapping)


def update_measurements(window: Any, result: Any) -> None:
    """Render measured profiles and quality tables from plain results."""

    observables = _result_value(result, ("observables", "measurements"), None)
    window.measurement_observables = observables
    angular = _read(observables, ("angular", "angular_spectrum"), None)
    ridge = _read(observables, ("ridge", "ridges", "ridge_track"), None)
    lobes = _sequence(_read(observables, ("lobes", "lobe_metrics", "four_lobes"), []))
    ellipse = _result_value(result, ("ellipse_fit", "ellipse", "ellipse_result"), None)
    if ellipse is None:
        ellipse = _read(observables, ("ellipse",), None)

    if window.angular_plot is not None:
        window.angular_plot.clear()
        for plot in (
            window.angular_plot,
            window.coverage_plot,
            window.ridge_plot,
            window.radial_profile_plot,
        ):
            if plot.plotItem.legend is None:
                plot.addLegend(offset=(8, 8))
        angle = _read(angular, ("angle_deg",), None)
        if angle is None:
            raw_angle = _sequence(_read(angular, ("angle", "azimuth"), []))
            angle = [math.degrees(float(item)) for item in raw_angle]
        intensity = _sequence(_read(angular, ("intensity", "profile"), []))
        coverage = _sequence(_read(angular, ("coverage",), []))
        try:
            angle_values = [
                float(value) for value in _sequence(angle) if math.isfinite(float(value))
            ]
            intensity_values = [
                float(value) for value in intensity if math.isfinite(float(value))
            ]
            coverage_values = [
                float(value) for value in coverage if math.isfinite(float(value))
            ]
        except (TypeError, ValueError):
            angle_values, intensity_values, coverage_values = [], [], []
        if angle_values or intensity_values or coverage_values:
            window._set_profile_summary(
                window._tr(
                    "profile.summary",
                    count=max(len(angle_values), len(intensity_values), len(coverage_values)),
                    amin=_format_metric(min(angle_values)) if angle_values else "—",
                    amax=_format_metric(max(angle_values)) if angle_values else "—",
                    imin=_format_metric(min(intensity_values)) if intensity_values else "—",
                    imax=_format_metric(max(intensity_values)) if intensity_values else "—",
                    cmin=_format_metric(min(coverage_values)) if coverage_values else "—",
                    cmax=_format_metric(max(coverage_values)) if coverage_values else "—",
                )
            )
        else:
            window._set_profile_summary(window._tr("profile.summary_empty"))
        try:
            window.angular_plot.plot(
                list(angle),
                list(intensity),
                pen=_pg.mkPen(50, 150, 255, width=2),
                name=window._tr("legend.intensity"),
            )
        except (TypeError, ValueError):
            pass
        window.coverage_plot.clear()
        try:
            if len(coverage):
                window.coverage_plot.setYRange(0.0, 1.0, padding=0.02)
                window.coverage_plot.plot(
                    list(angle),
                    list(coverage),
                    pen=_pg.mkPen(255, 190, 55, width=2),
                    symbol="o",
                    symbolSize=4,
                    symbolBrush=_pg.mkBrush(255, 190, 55, 180),
                    name=window._tr("legend.coverage"),
                )
        except (TypeError, ValueError):
            pass

    point_source = _read(ridge, ("points", "observed_points"), None)
    if point_source is None:
        point_source = _result_value(result, ("ridge_points", "ridges"), [])
    point_rows = _sequence(point_source)
    radial_profiles = _sequence(
        _read(
            observables,
            ("lobe_radial_profiles", "radial_profiles"),
            _result_value(result, ("lobe_radial_profiles", "radial_profiles"), []),
        )
    )
    radial_peaks = _sequence(
        _read(
            observables,
            ("lobe_radial_peaks", "radial_peaks"),
            _result_value(result, ("lobe_radial_peaks", "radial_peaks"), []),
        )
    )
    if window.radial_profile_plot is not None:
        window.radial_profile_plot.clear()
        profile_q_unit = "unknown"
        for profile_index, profile in enumerate(radial_profiles):
            q_values = _sequence(_read(profile, ("q",), []))
            intensities = _sequence(_read(profile, ("intensity", "profile"), []))
            if not q_values or not intensities:
                continue
            profile_q_unit = str(_read(profile, ("q_unit",), profile_q_unit) or profile_q_unit)
            try:
                window.radial_profile_plot.plot(
                    [float(value) for value in q_values],
                    [float(value) for value in intensities],
                    pen=_pg.mkPen(
                        70 + (profile_index * 53) % 160,
                        170,
                        220 - (profile_index * 37) % 130,
                        width=1.5,
                    ),
                    name=window._tr("legend.lobe", index=profile_index + 1),
                )
            except (TypeError, ValueError):
                continue
        window._ridge_plot_q_unit = profile_q_unit
        window.radial_profile_plot.setLabel(
            "bottom",
            window._tr("axis.radial_q", unit=profile_q_unit),
        )

    if window.ridge_plot is not None:
        window.ridge_plot.clear()
        accepted_x: list[float] = []
        accepted_y: list[float] = []
        rejected_x: list[float] = []
        rejected_y: list[float] = []
        q_unit = str(_read(ridge, ("q_unit",), "unknown") or "unknown")
        window._ridge_plot_q_unit = q_unit
        window.ridge_plot.setLabel("left", window._tr("axis.ridge_q", unit=q_unit))
        for point in point_rows:
            raw_angle = _read(point, ("angle_deg",), None)
            if raw_angle is None:
                raw_angle = _read(point, ("angle", "azimuth", "phi"), None)
                raw_angle = math.degrees(float(raw_angle)) if raw_angle is not None else None
            q_value = _read(point, ("q", "q_star", "q_position"), None)
            if raw_angle is None or q_value is None:
                continue
            try:
                x_value, y_value = float(raw_angle), float(q_value)
            except (TypeError, ValueError):
                continue
            accepted = bool(_read(point, ("accepted", "valid"), True))
            (accepted_x if accepted else rejected_x).append(x_value)
            (accepted_y if accepted else rejected_y).append(y_value)
        if accepted_x:
            window.ridge_plot.plot(
                accepted_x,
                accepted_y,
                pen=None,
                symbol="o",
                symbolSize=6,
                symbolBrush=_pg.mkBrush(70, 210, 120, 210),
                name=window._tr("legend.accepted"),
            )
        if rejected_x:
            window.ridge_plot.plot(
                rejected_x,
                rejected_y,
                pen=None,
                symbol="x",
                symbolSize=8,
                symbolPen=_pg.mkPen(230, 90, 90, width=2),
                name=window._tr("legend.rejected"),
            )

    window.ridge_table.setRowCount(len(point_rows))
    for row_index, point in enumerate(point_rows):
        raw_angle = _read(point, ("angle_deg",), None)
        if raw_angle is None:
            raw_angle = _read(point, ("angle", "azimuth", "phi"), None)
            raw_angle = math.degrees(float(raw_angle)) if raw_angle is not None else None
        accepted = bool(_read(point, ("accepted", "valid"), True))
        method_value = str(_read(point, ("method",), "observed"))
        branch_value = _read(point, ("branch_id", "branch"), None)
        quadrant_value = _read(point, ("quadrant_pair", "quadrant"), None)
        if branch_value is not None or quadrant_value is not None:
            method_value += (
                f"; branch={branch_value if branch_value is not None else '—'}"
                f"; quadrant={quadrant_value if quadrant_value is not None else '—'}"
            )
        values = (
            _format_metric(raw_angle),
            _format_metric(_read(point, ("q", "q_star", "q_position"), None)),
            accepted,
            method_value,
        )
        for column, value in enumerate(values):
            item = (
                window._boolean_table_item(value)
                if column == 2
                else QtWidgets.QTableWidgetItem(str(value))
            )
            window.ridge_table.setItem(row_index, column, item)

    window.lobe_table.setRowCount(len(lobes))
    for row_index, lobe in enumerate(lobes):
        angle = _read(lobe, ("angle_deg",), None)
        if angle is None:
            raw_angle = _read(lobe, ("angle", "azimuth"), None)
            angle = math.degrees(float(raw_angle)) if raw_angle is not None else None
        valid = bool(_read(lobe, ("valid", "accepted"), True))
        values = (
            _format_metric(angle),
            _format_metric(_read(lobe, ("intensity",), None)),
            _format_metric(_read(lobe, ("baseline",), None)),
            _format_metric(_read(lobe, ("snr",), None)),
            _format_metric(_read(lobe, ("fwhm_deg",), None)),
            _format_metric(_read(lobe, ("coverage",), None)),
            valid,
            ", ".join(str(item) for item in _sequence(_read(lobe, ("flags",), ()))),
        )
        for column, value in enumerate(values):
            item = (
                window._boolean_table_item(value)
                if column == 6
                else QtWidgets.QTableWidgetItem(str(value))
            )
            window.lobe_table.setItem(row_index, column, item)

    window.radial_table.setRowCount(len(radial_peaks))
    for row_index, peak in enumerate(radial_peaks):
        raw_angle = _read(peak, ("angle_deg",), None)
        if raw_angle is None:
            raw_angle = _read(peak, ("angle", "azimuth"), None)
            raw_angle = math.degrees(float(raw_angle)) if raw_angle is not None else None
        raw_q_star = _read(peak, ("q_star", "q"), None)
        values = (
            _format_metric(raw_angle),
            _format_metric(raw_q_star),
            _format_metric(_read(peak, ("lamellar_spacing", "spacing", "Ln"), None)),
            _format_metric(_read(peak, ("snr",), None)),
            _format_metric(_read(peak, ("radial_fwhm", "fwhm"), None)),
            _format_metric(_read(peak, ("coverage",), None)),
            bool(_read(peak, ("valid", "accepted"), True)),
            ", ".join(str(item) for item in _sequence(_read(peak, ("flags",), ()))),
        )
        for column, value in enumerate(values):
            item = (
                window._boolean_table_item(value)
                if column == 6
                else QtWidgets.QTableWidgetItem(str(value))
            )
            window.radial_table.setItem(row_index, column, item)

    def compact_summary(value: Any) -> str:
        if value is None:
            return "—"
        if isinstance(value, Mapping):
            return (
                "; ".join(f"{key}={compact_summary(item)}" for key, item in value.items())
                or "—"
            )
        if isinstance(value, (list, tuple)):
            return "[" + ", ".join(compact_summary(item) for item in value) + "]"
        return str(value)

    quality_payload = _read(ellipse, ("quality",), {})
    quality_metrics = _read(quality_payload, ("metrics",), {})
    symmetry_payload = _read(ellipse, ("symmetry",), None)
    if not isinstance(symmetry_payload, Mapping):
        symmetry_payload = _read(quality_metrics, ("symmetry",), None)
    if isinstance(symmetry_payload, Mapping):
        symmetry_status = _read(
            symmetry_payload,
            ("symmetry_status", "status"),
            _read(quality_metrics, ("symmetry_status",), "unknown"),
        )
        reference_axis = _read(
            symmetry_payload,
            ("reference_axis_deg",),
            _read(ellipse, ("reference_axis_deg",), None),
        )
        quadrant_counts = _read(
            symmetry_payload, ("quadrant_counts", "quadrant_count"), None
        )
        branch_quadrant_counts = _read(
            symmetry_payload, ("branch_quadrant_counts",), None
        )
        paired_support = _read(symmetry_payload, ("paired_support",), None)
        branch_leaks = _read(symmetry_payload, ("branch_leaks", "leaks"), None)
        unassigned = _read(symmetry_payload, ("unassigned_count", "unassigned"), None)
        symmetry_summary = window._tr(
            "profile.symmetry_summary",
            status=compact_summary(symmetry_status),
            axis=compact_summary(reference_axis),
            quadrants=(
                f"{compact_summary(quadrant_counts)}; "
                f"branches={compact_summary(branch_quadrant_counts)}"
            ),
            support=compact_summary(paired_support),
            leaks=compact_summary(branch_leaks),
            unassigned=compact_summary(unassigned),
        )
        window._set_profile_summary(
            f"{window.profile_summary_label.text()}\n{symmetry_summary}"
        )

    def flag_values(value: Any) -> list[str]:
        if isinstance(value, str):
            return [part.strip() for part in value.split(",") if part.strip()]
        return [str(item) for item in _sequence(value) if item]

    quality_flags = flag_values(_read(quality_payload, ("flags",), ()))
    ellipse_flags = flag_values(_read(ellipse, ("flags",), ()))
    p4_flags = list(dict.fromkeys(str(item) for item in (*quality_flags, *ellipse_flags)))
    bound_flags = _read(ellipse, ("bound_flags", "bound_status"), {}) or {}
    ellipse_quality_status = _read(
        quality_payload,
        ("status", "engineering_status"),
        _read(ellipse, ("quality_status", "status"), None),
    )
    axis_ratio = _read(ellipse, ("axis_ratio", "axes_ratio"), None)
    unpublished_shape = unpublished_ellipse_shape(
        quality_status=ellipse_quality_status,
        axis_ratio=axis_ratio,
        flags=p4_flags,
    )
    solver_failed = _read(ellipse, ("success",), None) is False or str(
        ellipse_quality_status or ""
    ).upper() in {"FAIL", "FAILED", "INVALID"}
    ellipse_kind = classify_ellipse_publication(
        quality_status=ellipse_quality_status,
        axis_ratio=axis_ratio,
        flags=p4_flags,
    )
    ellipse_limited = ellipse_candidate_is_limited(
        {"flags": p4_flags, "bound_flags": bound_flags}
    )
    shape_candidate = ellipse_limited or solver_failed or ellipse_kind != "ellipse"
    quality_status_value = _read(
        ellipse,
        ("quality_status", "status"),
        _read(quality_payload, ("status",), None),
    )
    if ellipse_limited and ellipse_kind == "ellipse":
        qualifier = window._tr("quality.ellipse_candidate_limited")
        quality_status_value = (
            f"{quality_status_value} · {qualifier}"
            if quality_status_value not in (None, "")
            else qualifier
        )
    ellipse_rows = (
        ("ellipse.a", None if unpublished_shape else _read(ellipse, ("a",), None)),
        ("ellipse.b", None if unpublished_shape else _read(ellipse, ("b",), None)),
        (
            "ellipse.axis_ratio",
            None if unpublished_shape else _read(ellipse, ("axis_ratio", "axes_ratio"), None),
        ),
        (
            "ellipse.ellipticity",
            None
            if unpublished_shape
            else _read(ellipse, ("ellipticity", "eccentricity"), None),
        ),
        (
            "ellipse.theta",
            None if unpublished_shape else _read(ellipse, ("theta_deg", "angle_deg"), None),
        ),
        (
            "ellipse.ln",
            None if unpublished_shape else _read(ellipse, ("Ln_from_minor_axis_nm",), None),
        ),
        (
            "ellipse.lz",
            None if unpublished_shape else _read(ellipse, ("Lz_from_draw_axis_nm",), None),
        ),
        (
            "ellipse.l_major",
            None if unpublished_shape else _read(ellipse, ("L_from_major_axis_nm",), None),
        ),
        ("ellipse.l_radial", _read(ellipse, ("L_from_observed_radius_nm",), None)),
        ("ellipse.q_star", _read(ellipse, ("q_star_from_arcs",), None)),
        (
            "ellipse.q_star_source",
            translate_q_star_source(
                window._language, _read(ellipse, ("q_star_source",), None)
            ),
        ),
        ("ellipse.rmse", _read(ellipse, ("rmse", "residual_rms"), None)),
        ("ellipse.rss", _read(ellipse, ("rss",), None)),
        ("ellipse.n_points", _read(ellipse, ("n_points", "n_data"), None)),
        ("ellipse.quality", _read(ellipse, ("success",), None)),
        (
            "ellipse.flags",
            ", ".join(str(item) for item in _sequence(_read(ellipse, ("flags",), ()))),
        ),
        ("ellipse.phi_app", _read(observables, ("phi_app_deg",), None)),
        ("ellipse.alpha_candidate", _read(observables, ("alpha_candidate_deg",), None)),
        ("ellipse.psi_candidate", _read(observables, ("psi_candidate_deg",), None)),
        (
            "ellipse.stderr",
            ", ".join(
                f"{key}={_format_metric(value)}"
                for key, value in (_read(ellipse, ("stderr",), {}) or {}).items()
            ),
        ),
        (
            "ellipse.bound_flags",
            ", ".join(
                f"{key}={bool(value)}"
                for key, value in (
                    _read(ellipse, ("bound_flags", "bound_status"), {}) or {}
                ).items()
            ),
        ),
        ("ellipse.quality_status", quality_status_value),
        (
            "ellipse.p4_flags",
            ", ".join(p4_flags)
            or ", ".join(
                str(item) for item in _sequence(_read(observables, ("flags",), ()))
            ),
        ),
        (
            "ellipse.q_unit",
            _read(ellipse, ("q_unit",), _read(observables, ("q_unit",), "unknown")),
        ),
        ("ellipse.geometry_action", _result_value(result, ("geometry_action",), None)),
        ("ellipse.symmetry_status", _read(symmetry_payload, ("symmetry_status", "status"), None)),
        ("ellipse.reference_axis", _read(symmetry_payload, ("reference_axis_deg",), None)),
        (
            "ellipse.quadrant_counts",
            (
                f"{compact_summary(_read(symmetry_payload, ('quadrant_counts', 'quadrant_count'), None))}; "
                f"branches={compact_summary(_read(symmetry_payload, ('branch_quadrant_counts',), None))}"
            )
            if isinstance(symmetry_payload, Mapping)
            else None,
        ),
        (
            "ellipse.paired_support",
            compact_summary(_read(symmetry_payload, ("paired_support",), None))
            if isinstance(symmetry_payload, Mapping)
            else None,
        ),
        (
            "ellipse.branch_leaks",
            compact_summary(
                {
                    "leaks": _read(symmetry_payload, ("branch_leaks", "leaks"), None),
                    "unassigned": _read(
                        symmetry_payload, ("unassigned_count", "unassigned"), None
                    ),
                }
            )
            if isinstance(symmetry_payload, Mapping)
            else None,
        ),
    )
    has_rows = ellipse is not None or observables is not None
    window.ellipse_table.setRowCount(len(ellipse_rows) if has_rows else 0)
    for row_index, (key, value) in enumerate(ellipse_rows if has_rows else ()):
        window.ellipse_table.setItem(row_index, 0, QtWidgets.QTableWidgetItem(window._tr(key)))
        value_item = (
            window._boolean_table_item(value)
            if key == "ellipse.quality" and value is not None
            else QtWidgets.QTableWidgetItem(_format_metric(value))
        )
        if shape_candidate and value is not None and key in {
            "ellipse.a",
            "ellipse.b",
            "ellipse.axis_ratio",
            "ellipse.ellipticity",
            "ellipse.theta",
            "ellipse.ln",
            "ellipse.lz",
            "ellipse.l_major",
        }:
            value_item.setText("≈" + _format_metric(value))
            tooltip_key = (
                "tooltip.ellipse_solver_diagnostic"
                if solver_failed
                else "tooltip.batch_quality_ellipse_candidate"
            )
            tooltip = window._tr(tooltip_key)
            if p4_flags:
                tooltip += "\n" + ", ".join(p4_flags)
            value_item.setToolTip(tooltip)
        window.ellipse_table.setItem(row_index, 1, value_item)


def plot_evolution(window: Any, records: Iterable[Any]) -> None:
    window.evolution_records = list(records)
    window.lamellar_page.set_series(
        window.evolution_records,
        context_signature=window._fit_state_signature(),
    )
    if not window.evolution_records:
        window.evolution_table.setRowCount(0)
        window.evolution_parameter_combo.clear()
        if window.evolution_plot is not None:
            window.evolution_plot.clear()
        return
    rows: list[Mapping[str, Any]] = [
        _flatten_evolution_record(record, index)
        for index, record in enumerate(window.evolution_records)
    ]
    window._evolution_rows = rows
    keys = list(dict.fromkeys(key for row in rows for key in row.keys()))
    window.evolution_table.setColumnCount(len(keys))
    window.evolution_table.setHorizontalHeaderLabels([str(key) for key in keys])
    window.evolution_table.setRowCount(len(rows))
    for row_index, row in enumerate(rows):
        for column, key in enumerate(keys):
            cell_value = row.get(key)
            if cell_value is None or (
                isinstance(cell_value, float) and not math.isfinite(cell_value)
            ):
                cell_text = ""
            else:
                cell_text = str(cell_value)
            window.evolution_table.setItem(
                row_index, column, QtWidgets.QTableWidgetItem(cell_text)
            )
    numeric_keys = _evolution_numeric_keys(rows, keys)
    window.evolution_parameter_combo.blockSignals(True)
    window.evolution_parameter_combo.clear()
    window.evolution_parameter_combo.addItems(numeric_keys)
    window._refresh_evolution_item_tooltips()
    preferred = (
        window.evolution_y_key
        if window.evolution_y_key in numeric_keys
        else ("rmse" if "rmse" in numeric_keys else (numeric_keys[0] if numeric_keys else ""))
    )
    if preferred:
        window.evolution_parameter_combo.setCurrentText(preferred)
    window.evolution_parameter_combo.blockSignals(False)
    window.evolution_y_key = preferred
    window._render_evolution(preferred)


def render_evolution(window: Any, y_key: str | None = None) -> None:
    """Render the selected flattened parameter against time/frame."""

    if y_key is None:
        y_key = window.evolution_parameter_combo.currentText()
    window.evolution_y_key = str(y_key or "")
    if window.evolution_plot is None:
        return
    window.evolution_plot.clear()
    window.evolution_plot.setLabel("left", window.evolution_y_key or window._tr("axis.value"))
    rows = window._evolution_rows
    x_values = [
        _numeric(row.get("time", row.get("time_s", row.get("frame", index))), index)
        for index, row in enumerate(rows)
    ]
    y_values = [
        _numeric(row.get(window.evolution_y_key, float("nan")), float("nan"))
        for row in rows
    ]
    categories = [_classify_evolution_row(row, window.evolution_y_key) for row in rows]
    line_y = [
        value if category != "failed" else float("nan")
        for value, category in zip(y_values, categories)
    ]
    if any(_is_finite(x) and _is_finite(y) for x, y in zip(x_values, line_y)):
        window.evolution_plot.plot(
            x_values,
            line_y,
            pen=_pg.mkPen(70, 170, 255, width=2),
            connect="finite",
        )
    styles = {
        "measured": ((42, 154, 220), "o"),
        "limited": ((215, 142, 30), "t"),
        "failed": ((130, 130, 138), "x"),
    }
    for category, (color, symbol) in styles.items():
        points = [
            (x, y)
            for x, y, current in zip(x_values, y_values, categories)
            if current == category and _is_finite(x) and _is_finite(y)
        ]
        if points:
            window.evolution_plot.plot(
                [point[0] for point in points],
                [point[1] for point in points],
                pen=None,
                symbol=symbol,
                symbolSize=8 if category != "measured" else 6,
                symbolPen=_pg.mkPen(color, width=1.5),
                symbolBrush=None if category == "failed" else _pg.mkBrush(color),
            )


__all__ = [
    "build_image_measurement_pages",
    "build_measurements_page",
    "fill_batch_table_row",
    "plot_evolution",
    "render_evolution",
    "sync_image_measurement_pages",
    "update_batch_rows",
    "update_measurements",
]
