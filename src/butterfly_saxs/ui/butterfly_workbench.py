"""Qt butterfly-analysis page.

This page is a thin workflow adapter.  It owns interaction state (selected
frame, branch visibility, serializable edits and display contrast), while the
existing :class:`RefinementMainWindow` continues to own the worker lifecycle
and service calls.  In particular, changing contrast or selecting a point
never starts an analysis request.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import math
from pathlib import Path
from typing import Any

from ..butterfly_quality import (
    classify_ellipse_publication,
    ellipse_candidate_is_limited,
    unpublished_ellipse_shape,
)
from ..butterfly_settings import normalize_butterfly_settings
from ..fit_overlays import fit_geometry_layers
from ..settings import canonical_q_unit
from ._butterfly_edits import (
    apply_edit_request as _apply_edit_request,
    exclude_selected_point as _exclude_selected_point_impl,
    open_numeric_edit_dialog as _open_numeric_edit_dialog_impl,
    redo_edit as _redo_edit_impl,
    restore_workbench_state as _restore_workbench_state,
    snapshot_workbench_state as _snapshot_workbench_state,
    start_correction as _start_correction_impl,
    undo_edit as _undo_edit_impl,
    update_edit_buttons as _update_edit_buttons_impl,
)
from ._butterfly_profiles import (
    _ProfilePanel,
    _finite,
    _fmt,
    annular_profile_title as _annular_profile_title_impl,
    annular_profile_tooltip as _annular_profile_tooltip_impl,
    extract_profile_series as _extract_profile_series,
    radial_profile_title as _radial_profile_title_impl,
    radial_profile_tooltip as _radial_profile_tooltip_impl,
    render_annular_profile as _render_annular_profile_impl,
    render_ellipse_local as _render_ellipse_local_impl,
    render_peak_profiles as _render_peak_profiles_impl,
    render_profile as _render_profile_impl,
    render_radial_profile as _render_radial_profile_impl,
    sector_failure_code as _sector_failure_code_impl,
    sector_failure_text as _sector_failure_text_impl,
)
from ._common import _read
from .qt_compat import QT_AVAILABLE, QtCore, QtGui, QtWidgets, require_qt
from .qspace import QSpaceView, _point_is_accepted
from .butterfly_export import export_butterfly_analysis
from .i18n import translate, translate_q_star_source
from .butterfly_summary import ButterflyQualitySummary

try:
    import numpy as _np
except Exception:  # pragma: no cover - numpy is a core dependency normally
    _np = None

try:
    import pyqtgraph as _pg
except Exception:  # pragma: no cover - optional plotting dependency
    _pg = None


DEFAULT_BUTTERFLY_SETTINGS: dict[str, Any] = {
    "stage": "trace",
    # A fresh workbench follows the q-ring -> I(chi) -> four-lobe trajectory
    # observable.  Recipes/results without an explicit method are still
    # interpreted as the historical curvature workflow below.
    "trace_method": "annular_peak",
    "sector_width_deg": 10.0,
    "sector_step_deg": 5.0,
    "annular_radial_bins": 40,
    "annular_angle_bins": 72,
    "edits": [],
    "resamples": 0,
    "evaluation_resamples": 32,
    "seed": 20260906,
    "sensitivity": True,
}

_TRACE_METHOD_RADIAL_SECTOR = "radial_sector"
_TRACE_METHOD_ANNULAR_PEAK = "annular_peak"
_TRACE_METHOD_CURVATURE = "curvature"


def _canonical_trace_method(value: Any, *, default: str = _TRACE_METHOD_CURVATURE) -> str:
    """Map persisted/UI aliases to the butterfly tracing choices."""

    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if text in {
        "annular_peak",
        "annular_peaks",
        "annular_trajectory",
        "q_ring_peak",
        "q_ring_trajectory",
        "azimuthal_peak",
    }:
        return _TRACE_METHOD_ANNULAR_PEAK
    if text in {"radial_sector", "sector", "radial", "sector_peak"}:
        return _TRACE_METHOD_RADIAL_SECTOR
    if text in {
        "curvature",
        "butterfly_curvature",
        "surface_curvature",
        "curvature_ridge",
    }:
        return _TRACE_METHOD_CURVATURE
    return default


def _finite_positive(value: Any, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) and result > 0.0 else float(default)


def _bounded_int(value: Any, default: int, *, minimum: int, maximum: int) -> int:
    """Read a persisted integer control without accepting booleans/fractions."""

    if isinstance(value, bool):
        return int(default)
    try:
        number = int(value)
    except (TypeError, ValueError):
        return int(default)
    if number != float(value) or not minimum <= number <= maximum:
        return int(default)
    return number

_ANALYSIS_JOB_KINDS = frozenset(
    {
        "preview",
        "optimize",
        "measure_geometry",
        "refine_geometry",
        "trace",
        "evaluate",
    }
)
_DETACHED_JOB_KINDS = frozenset({"butterfly_figure_export"})


def _flag_values(value: Any) -> list[str]:
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [str(item) for item in value if item]
    return []


if QT_AVAILABLE:

    class ButterflyWorkbench(QtWidgets.QWidget):
        """First-page butterfly tracing/evaluation workflow."""

        identifyRequested = QtCore.Signal(object)
        evaluateRequested = QtCore.Signal(object)
        applyToBatchRequested = QtCore.Signal(object)
        cancelRequested = QtCore.Signal()
        editChanged = QtCore.Signal(object)
        frameSelected = QtCore.Signal(object)
        pointSelected = QtCore.Signal(object)
        displayChanged = QtCore.Signal(str, float)
        analysisChanged = QtCore.Signal(object)
        exportRequested = QtCore.Signal()
        figureExportRequested = QtCore.Signal()

        def __init__(self, parent: Any = None, *, language: str = "zh_CN") -> None:
            super().__init__(parent)
            self.setObjectName("butterflyWorkbench")
            self._language = str(language)
            self._settings = deepcopy(DEFAULT_BUTTERFLY_SETTINGS)
            self._result: dict[str, Any] = {}
            self._frames: list[Any] = []
            self._profiles: dict[str, Any] = {}
            self._ellipse_local: dict[str, Any] = {}
            self._uv_display_magnification = 1.0
            self._selected_ellipse_point: dict[str, Any] = {}
            self._edits: list[dict[str, Any]] = []
            self._redo_edits: list[dict[str, Any]] = []
            self._legacy_method: str | None = None
            self._frame_data: dict[str, Any] = {}
            self._export_context: dict[str, Any] = {}
            self._result_fresh = False
            self._current_frame: Any = None
            self._diagnostic_magnification = 1.0
            self._busy = False
            self._has_loaded_data = False
            self._page_status_state = "ready"
            self._page_status_kind = ""
            self._page_status_error: Any = None
            self._q_range_error = ""
            self._job_elapsed_s: float | None = None
            self._job_progress_percent: int | None = None
            self._job_progress_phase = ""
            self._last_valid_analysis_range: tuple[float | None, float | None, float] = (
                None,
                None,
                0.0,
            )
            self._result_revision = 0
            self._peak_landmarks: dict[str, Any] = {}
            self._fit_layers: dict[str, Any] = {}
            self._model_parameters: Any = None
            self._model_reference_axis_deg: float | None = None
            self._model_status: str | None = None
            self._model_diagnostics: dict[str, Any] = {}
            self._selected_landmark: dict[str, Any] = {}
            self._selected_landmark_id: str | None = None
            self._requested_q_window: tuple[float, float] | None = None
            self._landmark_zoomed = False
            self._excluded_count = 0
            self._batch_success_count: int | None = None
            self._batch_limited_count = 0
            self._batch_failure_items: list[Any] = []
            self._detached_status_restore: tuple[str, str, Any] | None = None
            self._radial_sector_seen = False
            self._auto_hidden_radial_landmarks = False
            self._landmark_visibility_user_override = False
            self._suppress_landmark_visibility_tracking = False
            self._selected_profile_point_id: str | None = None
            self._rail_user_visible: bool | None = None
            self._manual_review: dict[str, Any] = {
                "manual_status": "unreviewed",
                "reviewed_by": "",
                "reviewed_at": None,
                "review_notes": "",
                "result_revision": None,
            }

            root = QtWidgets.QVBoxLayout(self)
            root.setContentsMargins(8, 6, 8, 6)
            root.setSpacing(6)

            header = QtWidgets.QHBoxLayout()
            self.title_label = QtWidgets.QLabel("蝴蝶分析 / Butterfly analysis", self)
            self.title_label.setObjectName("butterflyTitle")
            self.title_label.setStyleSheet("font-size: 17px; font-weight: 700;")
            header.addWidget(self.title_label)
            self.method_label = QtWidgets.QLabel("method: butterfly_curvature", self)
            self.method_label.setObjectName("butterflyMethodLabel")
            self.method_label.setToolTip("ridge_method=butterfly_curvature · arc tracing method")
            self.method_label.setStyleSheet("color: #56667a;")
            header.addWidget(self.method_label)
            header.addStretch(1)
            self.status_label = QtWidgets.QLabel("Ready · trace", self)
            self.status_label.setObjectName("butterflyStatusLabel")
            header.addWidget(self.status_label)
            self.rail_toggle = QtWidgets.QToolButton(self)
            self.rail_toggle.setObjectName("butterflyRailToggle")
            self.rail_toggle.setCheckable(True)
            self.rail_toggle.toggled.connect(self._on_rail_toggled)
            header.addWidget(self.rail_toggle)
            root.addLayout(header)
            self.quality_summary = ButterflyQualitySummary(self, language=self._language)
            root.addWidget(self.quality_summary)
            self.workflow_hint_label = QtWidgets.QLabel(self)
            self.workflow_hint_label.setObjectName("butterflyWorkflowHint")
            self.workflow_hint_label.setWordWrap(True)
            self.workflow_hint_label.setStyleSheet("color: #56667a; font-size: 11px;")
            root.addWidget(self.workflow_hint_label)

            self.legacy_banner = QtWidgets.QLabel(self)
            self.legacy_banner.setObjectName("butterflyLegacyBanner")
            self.legacy_banner.setWordWrap(True)
            self.legacy_banner.setStyleSheet("color: #875d08; background: #fff4cc; padding: 4px;")
            self.legacy_banner.setVisible(False)
            root.addWidget(self.legacy_banner)

            splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal, self)
            splitter.setObjectName("butterflyMainSplitter")
            splitter.setChildrenCollapsible(False)
            root.addWidget(splitter, 1)

            left = QtWidgets.QWidget(splitter)
            left_layout = QtWidgets.QHBoxLayout(left)
            left_layout.setContentsMargins(0, 0, 4, 0)
            frame_rail = QtWidgets.QWidget(left)
            frame_rail.setObjectName("butterflyFrameRail")
            self.frame_rail = frame_rail
            frame_rail.setMinimumWidth(142)
            frame_rail.setMaximumWidth(190)
            frame_layout = QtWidgets.QVBoxLayout(frame_rail)
            frame_layout.setContentsMargins(0, 0, 4, 0)
            frame_row = QtWidgets.QHBoxLayout()
            self.frame_title_label = QtWidgets.QLabel("Frames / 帧", frame_rail)
            frame_row.addWidget(self.frame_title_label)
            self.frame_count_label = QtWidgets.QLabel("0", frame_rail)
            frame_row.addWidget(self.frame_count_label)
            frame_row.addStretch(1)
            self.frame_source_label = QtWidgets.QLabel("loaded frame", frame_rail)
            self.frame_source_label.setWordWrap(True)
            self.frame_source_label.setStyleSheet("color: #56667a;")
            frame_layout.addLayout(frame_row)
            frame_layout.addWidget(self.frame_source_label)
            self.frame_list = QtWidgets.QListWidget(frame_rail)
            self.frame_list.setObjectName("butterflyFrameList")
            self.frame_list.setAccessibleName("Frames")
            self.frame_list.setAccessibleDescription("Select the active SAXS frame")
            self.frame_list.setMaximumHeight(160)
            self.frame_list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
            self.frame_list.currentRowChanged.connect(self._on_frame_changed)
            frame_layout.addWidget(self.frame_list, 1)
            self.point_title_label = QtWidgets.QLabel("Points / 点", frame_rail)
            self.point_list = QtWidgets.QListWidget(frame_rail)
            self.point_list.setObjectName("butterflyPointList")
            self.point_list.setAccessibleName("Points")
            self.point_list.setAccessibleDescription("Select a measured butterfly point")
            self.point_list.setMaximumHeight(150)
            self.point_list.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
            self.point_list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
            self.point_list.currentRowChanged.connect(self._on_point_list_changed)
            self.point_list.itemActivated.connect(lambda _item: self._on_point_list_changed(self.point_list.currentRow()))
            self.point_list.installEventFilter(self)
            delete_shortcut = QtGui.QShortcut(QtGui.QKeySequence("Delete"), self.point_list)
            delete_shortcut.setContext(QtCore.Qt.ShortcutContext.WidgetShortcut)
            delete_shortcut.activated.connect(self._exclude_selected_point)
            self._point_delete_shortcut = delete_shortcut
            frame_layout.addWidget(self.point_title_label)
            frame_layout.addWidget(self.point_list)
            frame_layout.addStretch(1)
            left_layout.addWidget(frame_rail, 0)
            canvas = QtWidgets.QWidget(left)
            canvas_layout = QtWidgets.QVBoxLayout(canvas)
            canvas_layout.setContentsMargins(0, 0, 0, 0)
            self.qspace = QSpaceView(canvas)
            self.qspace.setObjectName("butterflyQSpace")
            canvas_layout.addWidget(self.qspace, 5)

            self.diagnostics_tabs = QtWidgets.QTabWidget(canvas)
            self.diagnostics_tabs.setObjectName("butterflyDiagnosticsTabs")
            diagnostics = QtWidgets.QSplitter(
                QtCore.Qt.Orientation.Horizontal, self.diagnostics_tabs
            )
            diagnostics.setObjectName("butterflyDiagnosticsSplitter")
            self.normal_profile = _ProfilePanel("Normal profile · raw / fit / residual", diagnostics)
            self.normal_profile.setObjectName("normalProfilePanel")
            self.ellipse_diagnostic = _ProfilePanel(
                "Ellipse-local u/v · narrow-axis", diagnostics, data_view=False
            )
            self.ellipse_diagnostic.setObjectName("ellipseDiagnosticPanel")
            diagnostics.addWidget(self.normal_profile)
            diagnostics.addWidget(self.ellipse_diagnostic)
            diagnostics.setStretchFactor(0, 1)
            diagnostics.setStretchFactor(1, 1)
            self.diagnostics_tabs.addTab(diagnostics, "Point diagnostics")
            peak_profiles = QtWidgets.QSplitter(
                QtCore.Qt.Orientation.Horizontal, self.diagnostics_tabs
            )
            peak_profiles.setObjectName("peakLandmarkProfiles")
            self.peak_angular_profile = _ProfilePanel("Angular peak signal", peak_profiles)
            self.peak_angular_profile.setObjectName("peakAngularProfilePanel")
            self.peak_radial_profile = _ProfilePanel("Radial peak signal", peak_profiles)
            self.peak_radial_profile.setObjectName("peakRadialProfilePanel")
            peak_profiles.addWidget(self.peak_angular_profile)
            peak_profiles.addWidget(self.peak_radial_profile)
            peak_profiles.setStretchFactor(0, 1)
            peak_profiles.setStretchFactor(1, 1)
            self.diagnostics_tabs.addTab(peak_profiles, "Peak profiles")
            canvas_layout.addWidget(self.diagnostics_tabs, 2)
            self.magnification_label = QtWidgets.QLabel(
                "Display magnification: ×1.0 (diagnostic display only)", canvas
            )
            self.magnification_label.setObjectName("diagnosticMagnificationLabel")
            self.magnification_label.setStyleSheet("color: #56667a; font-size: 11px;")
            canvas_layout.addWidget(self.magnification_label)
            left_layout.addWidget(canvas, 1)
            splitter.addWidget(left)

            right_scroll = QtWidgets.QScrollArea(splitter)
            right_scroll.setObjectName("butterflyControlsScroll")
            right_scroll.setWidgetResizable(True)
            right_scroll.setMinimumWidth(340)
            right_scroll.setMaximumWidth(440)
            right_scroll.setHorizontalScrollBarPolicy(
                QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
            )
            right_panel = QtWidgets.QWidget(right_scroll)
            right_panel.setObjectName("butterflyControlsPanel")
            # QGroupBox titles start at the frame's x=0 in the native style;
            # shift them inward so the first glyph remains visible when the
            # narrow scroll viewport clips the content widget's right side.
            right_panel.setStyleSheet(
                "QGroupBox::title { subcontrol-origin: margin; left: 6px; }"
            )
            right_layout = QtWidgets.QVBoxLayout(right_panel)
            # Leave a few extra pixels before group-box titles.  The scroll
            # viewport otherwise clips the first CJK glyph at its left edge
            # on the narrow 980 px workbench layout.
            right_layout.setContentsMargins(8, 2, 8, 4)
            right_layout.setSpacing(7)

            analysis_group = QtWidgets.QGroupBox("Analysis range", right_panel)
            analysis_group.setObjectName("butterflyAnalysisRange")
            analysis_form = QtWidgets.QFormLayout(analysis_group)
            self.q_min_edit = QtWidgets.QLineEdit("Auto", analysis_group)
            self.q_min_edit.setObjectName("butterflyQMin")
            self.q_max_edit = QtWidgets.QLineEdit("Auto", analysis_group)
            self.q_max_edit.setObjectName("butterflyQMax")
            self.q_range_error_label = QtWidgets.QLabel(analysis_group)
            self.q_range_error_label.setObjectName("butterflyQRangeError")
            self.q_range_error_label.setWordWrap(True)
            self.q_range_error_label.setStyleSheet("color: #b42318; font-size: 11px;")
            self.q_range_error_label.setVisible(False)
            self.reference_axis_spin = QtWidgets.QDoubleSpinBox(analysis_group)
            self.reference_axis_spin.setObjectName("butterflyReferenceAxis")
            self.reference_axis_spin.setRange(-360.0, 360.0)
            self.reference_axis_spin.setDecimals(2)
            analysis_form.addRow("q min", self.q_min_edit)
            analysis_form.addRow("q max", self.q_max_edit)
            analysis_form.addRow(self.q_range_error_label)
            analysis_form.addRow("Reference axis (deg)", self.reference_axis_spin)
            self.q_min_edit.editingFinished.connect(self._on_analysis_range_changed)
            self.q_max_edit.editingFinished.connect(self._on_analysis_range_changed)
            self.reference_axis_spin.valueChanged.connect(self._on_analysis_range_changed)
            right_layout.addWidget(analysis_group)

            trace_group = QtWidgets.QGroupBox("Trace localization", right_panel)
            trace_group.setObjectName("butterflyTraceLocalization")
            trace_form = QtWidgets.QFormLayout(trace_group)
            self._trace_form = trace_form
            self.trace_method_combo = QtWidgets.QComboBox(trace_group)
            self.trace_method_combo.setObjectName("butterflyTraceMethod")
            self.trace_method_combo.addItem(
                "Annular q-ring angular peaks · I(χ)",
                _TRACE_METHOD_ANNULAR_PEAK,
            )
            self.trace_method_combo.addItem(
                "Sector-integrated radial peak · I(q)",
                _TRACE_METHOD_RADIAL_SECTOR,
            )
            self.trace_method_combo.addItem(
                "Curvature local candidates · advanced",
                _TRACE_METHOD_CURVATURE,
            )
            self.trace_method_combo.setAccessibleName("Butterfly tracing method")
            self.trace_method_combo.setToolTip(
                "For annular mode, each q ring is integrated over angle to find up to four lobe peaks and track them across q; "
                "radial sectors and curvature remain compatibility modes."
            )
            trace_form.addRow("Identification method", self.trace_method_combo)
            self.annular_radial_bins = QtWidgets.QSpinBox(trace_group)
            self.annular_radial_bins.setObjectName("butterflyAnnularRadialBins")
            self.annular_radial_bins.setRange(4, 192)
            self.annular_radial_bins.setSingleStep(1)
            self.annular_radial_bins.setValue(40)
            self.annular_radial_bins.setToolTip(
                "Number of q annuli used to build I(χ) profiles and link lobe trajectories."
            )
            trace_form.addRow("q-ring bins", self.annular_radial_bins)
            self.annular_angle_bins = QtWidgets.QSpinBox(trace_group)
            self.annular_angle_bins.setObjectName("butterflyAnnularAngleBins")
            self.annular_angle_bins.setRange(16, 720)
            self.annular_angle_bins.setSingleStep(1)
            self.annular_angle_bins.setValue(72)
            self.annular_angle_bins.setToolTip(
                "Number of angular bins in each I(χ) profile; no missing angular support is synthesized."
            )
            trace_form.addRow("angular bins", self.annular_angle_bins)
            self.sector_width_spin = QtWidgets.QDoubleSpinBox(trace_group)
            self.sector_width_spin.setObjectName("butterflySectorWidth")
            self.sector_width_spin.setRange(0.5, 180.0)
            self.sector_width_spin.setDecimals(1)
            self.sector_width_spin.setSingleStep(0.5)
            self.sector_width_spin.setValue(10.0)
            self.sector_width_spin.setSuffix("°")
            self.sector_width_spin.setToolTip(
                "Azimuth width integrated into each radial I(q) profile."
            )
            trace_form.addRow("Sector width", self.sector_width_spin)
            self.sector_step_spin = QtWidgets.QDoubleSpinBox(trace_group)
            self.sector_step_spin.setObjectName("butterflySectorStep")
            self.sector_step_spin.setRange(0.5, 180.0)
            self.sector_step_spin.setDecimals(1)
            self.sector_step_spin.setSingleStep(0.5)
            self.sector_step_spin.setValue(5.0)
            self.sector_step_spin.setSuffix("°")
            self.sector_step_spin.setToolTip(
                "Azimuth step between adjacent sector centers."
            )
            trace_form.addRow("Sector step", self.sector_step_spin)
            self.trace_method_combo.currentIndexChanged.connect(
                self._on_trace_settings_changed
            )
            self.sector_width_spin.valueChanged.connect(self._on_trace_settings_changed)
            self.sector_step_spin.valueChanged.connect(self._on_trace_settings_changed)
            self.annular_radial_bins.valueChanged.connect(self._on_trace_settings_changed)
            self.annular_angle_bins.valueChanged.connect(self._on_trace_settings_changed)
            right_layout.addWidget(trace_group)

            evaluation_group = QtWidgets.QGroupBox("Evaluation", right_panel)
            evaluation_group.setObjectName("butterflyEvaluationControls")
            evaluation_layout = QtWidgets.QFormLayout(evaluation_group)
            self.evaluation_resamples_combo = QtWidgets.QComboBox(evaluation_group)
            self.evaluation_resamples_combo.setObjectName("butterflyEvaluationResamples")
            for label, value in (("Quick fit · 0", 0), ("Standard · 32", 32), ("Extended · 128", 128)):
                self.evaluation_resamples_combo.addItem(label, value)
            self.evaluation_resamples_combo.setCurrentIndex(1)
            evaluation_layout.addRow("Uncertainty resamples", self.evaluation_resamples_combo)
            self.sensitivity_check = QtWidgets.QCheckBox("Run sensitivity checks", evaluation_group)
            self.sensitivity_check.setObjectName("butterflySensitivityChecks")
            self.sensitivity_check.setChecked(True)
            evaluation_layout.addRow(self.sensitivity_check)
            self.evaluation_resamples_combo.currentIndexChanged.connect(self._on_evaluation_settings_changed)
            self.sensitivity_check.toggled.connect(self._on_evaluation_settings_changed)
            right_layout.addWidget(evaluation_group)

            display_group = QtWidgets.QGroupBox("Display", right_panel)
            display_group.setObjectName("butterflyDisplayControls")
            display_form = QtWidgets.QFormLayout(display_group)
            self.display_scale_combo = QtWidgets.QComboBox(display_group)
            self.display_scale_combo.setObjectName("butterflyDisplayScale")
            self.display_scale_combo.addItem("Linear", "linear")
            self.display_scale_combo.addItem("Log1p", "log1p")
            self.display_scale_combo.addItem("Asinh", "asinh")
            display_form.addRow("Scale", self.display_scale_combo)
            self.display_percentile_spin = QtWidgets.QDoubleSpinBox(display_group)
            self.display_percentile_spin.setObjectName("butterflyDisplayPercentile")
            self.display_percentile_spin.setRange(50.0, 100.0)
            self.display_percentile_spin.setDecimals(1)
            self.display_percentile_spin.setValue(99.5)
            self.display_percentile_spin.setSuffix(" %")
            display_form.addRow("Upper percentile", self.display_percentile_spin)
            self.uv_magnification_spin = QtWidgets.QDoubleSpinBox(display_group)
            self.uv_magnification_spin.setObjectName("butterflyUvMagnification")
            self.uv_magnification_spin.setRange(0.1, 100.0)
            self.uv_magnification_spin.setDecimals(1)
            self.uv_magnification_spin.setValue(1.0)
            self.uv_magnification_spin.setSuffix(" ×")
            display_form.addRow("u/v magnification", self.uv_magnification_spin)
            self.display_reset_button = QtWidgets.QPushButton("Reset contrast", display_group)
            self.display_reset_button.setObjectName("butterflyDisplayReset")
            display_form.addRow(self.display_reset_button)
            self.display_scale_combo.currentIndexChanged.connect(self._on_display_changed)
            self.display_percentile_spin.valueChanged.connect(self._on_display_changed)
            self.uv_magnification_spin.valueChanged.connect(self._on_uv_magnification_changed)
            self.display_reset_button.clicked.connect(self._reset_display)
            right_layout.addWidget(display_group)

            overlay_group = QtWidgets.QGroupBox("Fit-source overlays", right_panel)
            overlay_group.setObjectName("butterflyFitOverlays")
            overlay_layout = QtWidgets.QVBoxLayout(overlay_group)
            overlay_layout.setContentsMargins(7, 5, 7, 5)
            overlay_form = QtWidgets.QFormLayout()
            self.overlay_mode_combo = QtWidgets.QComboBox(overlay_group)
            self.overlay_mode_combo.setObjectName("butterflyOverlayMode")
            for key, mode in (
                ("overlay.measured_only", "measured_only"),
                ("overlay.observed_ridges", "observed_ridges"),
                ("overlay.geometry_candidate", "geometry_candidate"),
                ("overlay.full2d_model", "full2d_model"),
                ("overlay.compare", "compare"),
            ):
                self.overlay_mode_combo.addItem(self._tr(key), mode)
            self.overlay_mode_combo.setCurrentIndex(
                self.overlay_mode_combo.findData("observed_ridges")
            )
            overlay_form.addRow("Displayed layer", self.overlay_mode_combo)
            overlay_layout.addLayout(overlay_form)
            self.fit_source_label = QtWidgets.QLabel(overlay_group)
            self.fit_source_label.setObjectName("butterflyFitSourceLabel")
            self.fit_source_label.setWordWrap(True)
            self.fit_source_label.setStyleSheet("color: #56667a; font-size: 11px;")
            overlay_layout.addWidget(self.fit_source_label)
            self.fit_assessment_label = QtWidgets.QLabel(overlay_group)
            self.fit_assessment_label.setObjectName("butterflyFitAssessment")
            self.fit_assessment_label.setWordWrap(True)
            self.fit_assessment_label.setStyleSheet(
                "color: #9f1d16; background: #fff0ee; padding: 5px; font-weight: 700;"
            )
            self.fit_assessment_label.setVisible(False)
            overlay_layout.addWidget(self.fit_assessment_label)
            self.overlay_mode_combo.currentIndexChanged.connect(
                self._on_overlay_mode_changed
            )
            right_layout.addWidget(overlay_group)

            landmark_group = QtWidgets.QGroupBox(
                "Peak landmarks · display only", right_panel
            )
            landmark_group.setObjectName("butterflyPeakLandmarks")
            landmark_layout = QtWidgets.QVBoxLayout(landmark_group)
            landmark_layout.setContentsMargins(7, 5, 7, 5)
            landmark_toggles = QtWidgets.QVBoxLayout()
            landmark_toggles.setSpacing(2)
            self.global_max_check = QtWidgets.QCheckBox(landmark_group)
            self.global_max_check.setObjectName("showGlobalRawMaximum")
            self.global_max_check.setChecked(True)
            landmark_toggles.addWidget(self.global_max_check)
            self.supported_peaks_check = QtWidgets.QCheckBox(landmark_group)
            self.supported_peaks_check.setObjectName("showSupportedPeaks")
            self.supported_peaks_check.setChecked(True)
            landmark_toggles.addWidget(self.supported_peaks_check)
            landmark_layout.addLayout(landmark_toggles)
            self.peak_table = QtWidgets.QTableWidget(0, 4, landmark_group)
            self.peak_table.setObjectName("butterflyPeakLandmarkTable")
            self.peak_table.setEditTriggers(
                QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
            )
            self.peak_table.setSelectionBehavior(
                QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
            )
            self.peak_table.setSelectionMode(
                QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
            )
            self.peak_table.setHorizontalScrollBarPolicy(
                QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
            )
            self.peak_table.verticalHeader().hide()
            self.peak_table.verticalHeader().setDefaultSectionSize(19)
            self.peak_table.horizontalHeader().setMinimumSectionSize(28)
            self.peak_table.horizontalHeader().setStretchLastSection(True)
            self.peak_table.setMinimumHeight(76)
            self.peak_table.setMaximumHeight(136)
            self.peak_table.setAccessibleName("Peak landmark diagnostics")
            self.peak_table.setAccessibleDescription(
                "Global raw maximum and supported lobe peaks; selecting a row focuses its actual q coordinate"
            )
            for column, width in enumerate((34, 66, 72, 138)):
                self.peak_table.setColumnWidth(column, width)
            landmark_layout.addWidget(self.peak_table)
            self.reset_peak_zoom_button = QtWidgets.QPushButton(landmark_group)
            self.reset_peak_zoom_button.setObjectName("resetPeakLandmarkZoom")
            self.reset_peak_zoom_button.setEnabled(False)
            self.reset_peak_zoom_button.clicked.connect(self._reset_landmark_zoom)
            landmark_layout.addWidget(self.reset_peak_zoom_button)
            self.peak_table.currentCellChanged.connect(
                self._on_peak_table_selection_changed
            )
            self.global_max_check.toggled.connect(self._on_landmark_visibility_changed)
            self.supported_peaks_check.toggled.connect(
                self._on_landmark_visibility_changed
            )
            right_layout.addWidget(landmark_group)

            branch_group = QtWidgets.QGroupBox("Branch / side visibility", right_panel)
            branch_group.setObjectName("branchSideVisibility")
            branch_layout = QtWidgets.QVBoxLayout(branch_group)
            branch_layout.setContentsMargins(8, 5, 8, 5)
            self._branch_checks: dict[tuple[int, str], QtWidgets.QCheckBox] = {}
            rows = (
                (0, "upper", "A · upper", QtGui.QColor(42, 154, 220)),
                (0, "lower", "A · lower", QtGui.QColor(42, 154, 220)),
                (1, "upper", "B · upper", QtGui.QColor(239, 143, 44)),
                (1, "lower", "B · lower", QtGui.QColor(239, 143, 44)),
            )
            for branch, side, text, color in rows:
                row_widget = QtWidgets.QWidget(branch_group)
                row_widget.setFixedHeight(20)
                row_layout = QtWidgets.QHBoxLayout(row_widget)
                row_layout.setContentsMargins(0, 0, 0, 0)
                swatch = QtWidgets.QLabel("●", row_widget)
                swatch.setStyleSheet(f"color: {color.name()};")
                row_layout.addWidget(swatch)
                check = QtWidgets.QCheckBox(text, row_widget)
                check.setObjectName(f"branch{branch}{side.capitalize()}Visible")
                check.setChecked(True)
                check.toggled.connect(
                    lambda enabled, branch=branch, side=side: self.qspace.set_visible_branch(branch, side, enabled)
                )
                row_layout.addWidget(check, 1)
                branch_layout.addWidget(row_widget)
                self._branch_checks[(branch, side)] = check
            unknown_check = QtWidgets.QCheckBox("Unknown / 未知", branch_group)
            unknown_check.setObjectName("branchUnknownVisible")
            unknown_check.setFixedHeight(20)
            unknown_check.setChecked(True)
            unknown_check.toggled.connect(lambda enabled: self.qspace.set_visible_branch(-1, "unknown", enabled))
            self._branch_checks[(-1, "unknown")] = unknown_check
            excluded_row = QtWidgets.QHBoxLayout()
            excluded_row.addWidget(unknown_check)
            self.show_excluded_check = QtWidgets.QCheckBox("Show excluded", branch_group)
            self.show_excluded_check.setObjectName("showExcludedPoints")
            self.show_excluded_check.setFixedHeight(20)
            self.show_excluded_check.setChecked(True)
            self.show_excluded_check.setToolTip(self._tr("tooltip.show_excluded_points"))
            self.show_excluded_check.toggled.connect(self.qspace.set_show_excluded)
            excluded_row.addWidget(self.show_excluded_check)
            self.excluded_count_label = QtWidgets.QLabel("0", branch_group)
            self.excluded_count_label.setStyleSheet("color: #56667a;")
            excluded_row.addWidget(self.excluded_count_label)
            excluded_row.addStretch(1)
            branch_layout.addLayout(excluded_row)
            branch_group.setMaximumHeight(132)
            right_layout.addWidget(branch_group)

            quantity_group = QtWidgets.QGroupBox("Quantitative parameters", right_panel)
            quantity_group.setObjectName("butterflyQuantitativeParameters")
            quantity_layout = QtWidgets.QVBoxLayout(quantity_group)
            quantity_layout.setContentsMargins(5, 5, 5, 5)
            self.quantity_table = QtWidgets.QTableWidget(0, 6, quantity_group)
            self.quantity_table.setObjectName("butterflyQuantitativeTable")
            self.quantity_table.setHorizontalHeaderLabels(
                ("Param", "Value", "State", "Cand.", "CI", "Why")
            )
            self.quantity_table.horizontalHeader().setStretchLastSection(True)
            self.quantity_table.horizontalHeader().setMinimumSectionSize(34)
            self.quantity_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
            self.quantity_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
            self.quantity_table.setMinimumHeight(112)
            self.quantity_table.setMaximumHeight(146)
            self.quantity_table.verticalHeader().setDefaultSectionSize(18)
            self.quantity_table.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            for column, width in enumerate((42, 38, 42, 46, 40, 48)):
                self.quantity_table.setColumnWidth(column, width)
            quantity_layout.addWidget(self.quantity_table)
            quantity_group.setMinimumHeight(126)
            right_layout.addWidget(quantity_group)

            correction_group = QtWidgets.QGroupBox("Corrections / 编辑", right_panel)
            correction_group.setObjectName("butterflyCorrections")
            correction_layout = QtWidgets.QFormLayout(correction_group)
            self.correction_mode_combo = QtWidgets.QComboBox(correction_group)
            self.correction_mode_combo.setObjectName("butterflyCorrectionMode")
            self.correction_mode_combo.addItem("Seed point", "seed")
            self.correction_mode_combo.addItem("Exclude point", "exclude_point")
            self.correction_mode_combo.addItem("Exclude rectangle", "rectangle_exclude")
            self.correction_mode_combo.addItem("Include rectangle", "rectangle_include")
            self.correction_mode_combo.addItem("Exclude polygon", "polygon_exclude")
            self.correction_mode_combo.addItem("Include polygon", "polygon_include")
            correction_layout.addRow("Edit mode", self.correction_mode_combo)
            self.seed_branch_combo = QtWidgets.QComboBox(correction_group)
            self.seed_branch_combo.addItem("Auto / unknown", -1)
            self.seed_branch_combo.addItem("A", 0)
            self.seed_branch_combo.addItem("B", 1)
            correction_layout.addRow("Seed branch", self.seed_branch_combo)
            self.seed_side_combo = QtWidgets.QComboBox(correction_group)
            self.seed_side_combo.addItem("Unknown", "unknown")
            self.seed_side_combo.addItem("Upper", "upper")
            self.seed_side_combo.addItem("Lower", "lower")
            correction_layout.addRow("Seed side", self.seed_side_combo)
            edit_buttons = QtWidgets.QHBoxLayout()
            self.correct_button = QtWidgets.QPushButton("Correct / 编辑", correction_group)
            self.correct_button.setObjectName("butterflyCorrectButton")
            self.correct_button.clicked.connect(self.start_correction)
            edit_buttons.addWidget(self.correct_button)
            self.undo_button = QtWidgets.QPushButton("Undo", correction_group)
            self.undo_button.setObjectName("butterflyUndoButton")
            self.undo_button.clicked.connect(self.undo_edit)
            edit_buttons.addWidget(self.undo_button)
            self.redo_button = QtWidgets.QPushButton("Redo", correction_group)
            self.redo_button.setObjectName("butterflyRedoButton")
            self.redo_button.clicked.connect(self.redo_edit)
            edit_buttons.addWidget(self.redo_button)
            correction_layout.addRow(edit_buttons)
            self.numeric_edit_button = QtWidgets.QPushButton("Keyboard edit / 键盘编辑", correction_group)
            self.numeric_edit_button.setObjectName("butterflyNumericEditButton")
            self.numeric_edit_button.clicked.connect(self._open_numeric_edit_dialog)
            correction_layout.addRow(self.numeric_edit_button)
            right_layout.addWidget(correction_group)

            actions_group = QtWidgets.QGroupBox("Workflow", right_panel)
            actions_group.setObjectName("butterflyWorkflow")
            actions_layout = QtWidgets.QVBoxLayout(actions_group)
            actions_layout.setContentsMargins(4, 3, 4, 3)
            actions_layout.setSpacing(2)
            self.identify_button = QtWidgets.QPushButton("Identify arcs / 识别", actions_group)
            self.identify_button.setObjectName("butterflyIdentifyButton")
            self.identify_button.setProperty("role", "primary")
            self.identify_button.setToolTip("Trace observed arcs using butterfly curvature")
            self.identify_button.clicked.connect(self.request_identify)
            actions_layout.addWidget(self.identify_button)
            self.evaluate_button = QtWidgets.QPushButton("Evaluate / 评估", actions_group)
            self.evaluate_button.setObjectName("butterflyEvaluateButton")
            self.evaluate_button.setToolTip("Evaluate the traced arcs with 32 uncertainty resamples")
            self.evaluate_button.clicked.connect(self.request_evaluate)
            actions_layout.addWidget(self.evaluate_button)
            self.apply_batch_button = QtWidgets.QPushButton("Apply to batch / 应用到批处理", actions_group)
            self.apply_batch_button.setObjectName("butterflyApplyBatchButton")
            self.apply_batch_button.clicked.connect(self.apply_to_batch)
            actions_layout.addWidget(self.apply_batch_button)
            self.cancel_button = QtWidgets.QPushButton("Cancel", actions_group)
            self.cancel_button.setObjectName("butterflyCancelButton")
            self.cancel_button.setShortcut(QtGui.QKeySequence("Esc"))
            self.cancel_button.clicked.connect(lambda _checked=False: self.cancelRequested.emit())
            actions_layout.addWidget(self.cancel_button)
            self.export_button = QtWidgets.QPushButton("Export analysis / 导出分析", actions_group)
            self.export_button.setObjectName("butterflyExportButton")
            self.export_button.clicked.connect(lambda _checked=False: self.exportRequested.emit())
            export_row = QtWidgets.QHBoxLayout()
            export_row.setSpacing(4)
            export_row.addWidget(self.export_button, 1)
            self.figure_export_button = QtWidgets.QPushButton("Measurement figure", actions_group)
            self.figure_export_button.setObjectName("butterflyFigureExportButton")
            self.figure_export_button.clicked.connect(
                lambda _checked=False: self.figureExportRequested.emit()
            )
            export_row.addWidget(self.figure_export_button, 1)
            actions_layout.addLayout(export_row)
            for action_button in (
                self.identify_button,
                self.evaluate_button,
                self.apply_batch_button,
                self.cancel_button,
                self.export_button,
                self.figure_export_button,
            ):
                action_button.setMinimumHeight(30)
                action_button.setMaximumHeight(34)
            actions_group.setMaximumHeight(210)
            right_layout.addWidget(actions_group)
            self.batch_feedback_label = QtWidgets.QLabel(actions_group)
            self.batch_feedback_label.setObjectName("butterflyBatchFeedback")
            self.batch_feedback_label.setWordWrap(True)
            self.batch_feedback_label.setStyleSheet("color: #56667a;")
            actions_layout.addWidget(self.batch_feedback_label)
            right_layout.addStretch(1)
            right_layout.removeWidget(actions_group)
            right_layout.insertWidget(1, actions_group)
            for form in right_panel.findChildren(QtWidgets.QFormLayout):
                form.setRowWrapPolicy(QtWidgets.QFormLayout.RowWrapPolicy.WrapLongRows)
                form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
            right_scroll.setWidget(right_panel)
            splitter.addWidget(right_scroll)
            splitter.setStretchFactor(0, 5)
            splitter.setStretchFactor(1, 2)

            self.qspace.pointSelected.connect(self._on_point_selected)
            self.qspace.landmarkSelected.connect(self._on_landmark_selected)
            self.qspace.editRequested.connect(self._on_edit_requested)
            self.qspace.set_overlay_mode(str(self.overlay_mode_combo.currentData()))
            self.qspace.set_landmark_visibility(
                global_raw_max=True,
                supported_peaks=True,
            )
            self._update_edit_buttons()
            self._sync_trace_method_controls()
            self.set_language(self._language)
            self._sync_action_state()

        def _on_rail_toggled(self, checked: bool) -> None:
            self._rail_user_visible = bool(checked)
            self.frame_rail.setVisible(checked)

        def _update_rail_layout(self) -> None:
            visible = self.width() >= 1180 if self._rail_user_visible is None else self._rail_user_visible
            self.frame_rail.setVisible(visible)
            blocker = QtCore.QSignalBlocker(self.rail_toggle)
            self.rail_toggle.setChecked(visible)
            del blocker

        def resizeEvent(self, event: Any) -> None:  # noqa: N802 - Qt API
            super().resizeEvent(event)
            if hasattr(self, "frame_rail"):
                self._update_rail_layout()

        @property
        def butterfly_settings(self) -> dict[str, Any]:
            result = deepcopy(self._settings)
            result["trace_method"] = self._trace_method()
            result["sector_width_deg"] = float(self.sector_width_spin.value())
            result["sector_step_deg"] = float(self.sector_step_spin.value())
            result["annular_radial_bins"] = int(self.annular_radial_bins.value())
            result["annular_angle_bins"] = int(self.annular_angle_bins.value())
            result["edits"] = deepcopy(self._edits)
            return result

        @property
        def analysis_settings(self) -> dict[str, Any]:
            return {
                # Keep the butterfly workflow family seam stable.  The
                # selected observable is carried by butterfly.trace_method;
                # generic azimuthal_peak is a separate observables workflow.
                "ridge_method": "butterfly_curvature",
                "butterfly": self.butterfly_settings,
            }

        @property
        def edits(self) -> list[dict[str, Any]]:
            return deepcopy(self._edits)

        @property
        def current_result(self) -> dict[str, Any]:
            return deepcopy(self._result)

        @property
        def manual_review(self) -> dict[str, Any]:
            """Return the page-local review state tied to this result revision."""

            review = deepcopy(self._manual_review)
            if review.get("result_revision") != self._result_revision or not self._result_fresh:
                review.update(
                    {
                        "manual_status": "unreviewed",
                        "reviewed_by": "",
                        "reviewed_at": None,
                        "review_notes": "",
                        "result_revision": None,
                    }
                )
            return review

        @property
        def result_revision(self) -> int:
            return int(self._result_revision)

        def set_manual_review(self, review: Mapping[str, Any] | None) -> None:
            """Record an explicit page-local review for the current result.

            The shared intensity-fit review session is deliberately not reused
            here: butterfly geometry jobs have a separate result lifecycle.
            """

            values = dict(review or {}) if isinstance(review, Mapping) else {}
            status = str(values.get("manual_status", values.get("status", "unreviewed")) or "unreviewed").lower()
            if status not in {"unreviewed", "accepted", "rejected"}:
                status = "unreviewed"
            revision = values.get("result_revision", self._result_revision)
            try:
                revision = int(revision)
            except (TypeError, ValueError):
                revision = self._result_revision
            if revision != self._result_revision or not self._result_fresh:
                status = "unreviewed"
                revision = None
            self._manual_review = {
                "manual_status": status,
                "reviewed_by": str(values.get("reviewed_by", values.get("reviewer", "")) or ""),
                "reviewed_at": values.get("reviewed_at"),
                "review_notes": str(values.get("review_notes", values.get("notes", "")) or ""),
                "result_revision": revision,
            }

        def _tr(self, key: str, **values: Any) -> str:
            try:
                return translate(self._language, key, **values)
            except (KeyError, ValueError):
                return key

        def _trace_method(self) -> str:
            return _canonical_trace_method(
                self._settings.get("trace_method"),
                default=_TRACE_METHOD_ANNULAR_PEAK,
            )

        @staticmethod
        def _result_trace_method(result: Mapping[str, Any]) -> str:
            """Infer the method that produced a payload before rendering it."""

            settings = result.get("settings")
            settings = settings if isinstance(settings, Mapping) else {}
            for source in (settings, result):
                explicit = source.get("trace_method")
                if explicit not in (None, ""):
                    return _canonical_trace_method(
                        explicit, default=_TRACE_METHOD_CURVATURE
                    )
            # ``ridge_method=butterfly_curvature`` is the stable family seam
            # for this page.  The payload-specific annular/sector blocks are
            # therefore checked before that historical family label.
            if isinstance(result.get("annular_peaks"), Mapping):
                return _TRACE_METHOD_ANNULAR_PEAK
            if isinstance(result.get("sector_peaks"), Mapping):
                return _TRACE_METHOD_RADIAL_SECTOR
            for source in (settings, result):
                ridge_method = source.get("ridge_method")
                if ridge_method not in (None, ""):
                    return _canonical_trace_method(
                        ridge_method, default=_TRACE_METHOD_CURVATURE
                    )
            method_version = str(result.get("method_version", "") or "").lower()
            if method_version.startswith("butterfly-annular-"):
                return _TRACE_METHOD_ANNULAR_PEAK
            if method_version.startswith("butterfly-radial-sector-"):
                return _TRACE_METHOD_RADIAL_SECTOR
            points = result.get("points")
            if isinstance(points, Sequence) and not isinstance(points, (str, bytes)):
                if any(
                    isinstance(point, Mapping)
                    and str(point.get("source_method", "") or "").lower()
                    in {_TRACE_METHOD_ANNULAR_PEAK, "annular_trajectory"}
                    for point in points
                ):
                    return _TRACE_METHOD_ANNULAR_PEAK
                if any(
                    isinstance(point, Mapping)
                    and str(point.get("source_method", "") or "").lower()
                    == _TRACE_METHOD_RADIAL_SECTOR
                    for point in points
                ):
                    return _TRACE_METHOD_RADIAL_SECTOR
            # A payload without explicit method metadata is a legacy curvature
            # result. Do not let the new-session annular default reinterpret
            # its landmarks.
            return _TRACE_METHOD_CURVATURE

        def _adopt_result_trace_method(self, result: Mapping[str, Any]) -> None:
            """Synchronize controls with a loaded result without invalidating it."""

            method = self._result_trace_method(result)
            self._settings["trace_method"] = method
            result_settings = result.get("settings")
            result_settings = result_settings if isinstance(result_settings, Mapping) else {}
            annular_bundle = result.get("annular_peaks")
            annular_settings = (
                annular_bundle.get("settings")
                if isinstance(annular_bundle, Mapping)
                else None
            )
            annular_settings = annular_settings if isinstance(annular_settings, Mapping) else {}
            for key in ("annular_radial_bins", "annular_angle_bins"):
                if key in result_settings:
                    self._settings[key] = result_settings[key]
                elif key in annular_settings:
                    self._settings[key] = annular_settings[key]
            for key in ("sector_width_deg", "sector_step_deg"):
                if key in result_settings:
                    self._settings[key] = result_settings[key]
            self._sync_trace_method_controls()
            self._render_trace_method_label()
            self._apply_trace_method_landmark_visibility()
            self._apply_trace_method_diagnostic_visibility()

        def _render_trace_method_label(self) -> None:
            english = self._language.lower().startswith("en")
            self.rail_toggle.setText("Frames / points" if english else "帧与测量点")
            self.rail_toggle.setToolTip(
                "Show or hide frame and measured-point lists; the analysis stays unchanged."
                if english else "展开或收起帧与测量点列表，不改变分析结果。"
            )
            self.rail_toggle.setAccessibleName(self.rail_toggle.text())
            method = self._trace_method()
            if method == _TRACE_METHOD_ANNULAR_PEAK:
                text = "Annular I(χ) four-lobe tracks" if english else "环积分 I(χ) 四瓣轨迹"
                tooltip = (
                    "Each q annulus is integrated over angle; up to four supported lobe peaks are linked across q."
                    if english
                    else "对每个 q 环沿方位积分得到 I(χ)，每环最多保留四个有支撑峰并沿 q 连成轨迹。"
                )
            elif method == _TRACE_METHOD_RADIAL_SECTOR:
                text = "Radial sector I(q) peak" if english else "扇区积分 I(q) 主峰"
                tooltip = (
                    "One radial I(q) profile is integrated per azimuth sector; q* is its selected peak."
                    if english
                    else "沿每个方位扇区积分得到 I(q)，每个扇区只定位一个主峰 q*。"
                )
            else:
                text = "Curvature local candidates · advanced" if english else "曲率局部候选（高级）"
                tooltip = (
                    "Advanced pixel-curvature candidates; use only when the sector profile is insufficient."
                    if english
                    else "高级像素曲率候选；仅在扇区积分剖面不足时使用。"
                )
            self.method_label.setText(text)
            self.method_label.setToolTip(tooltip)
            self.method_label.setAccessibleName(text)

        def _render_trace_method_controls(self) -> None:
            english = self._language.lower().startswith("en")
            group = self.findChild(QtWidgets.QGroupBox, "butterflyTraceLocalization")
            if group is None:
                return
            group.setTitle("Trace localization" if english else "峰位识别方式")
            form = group.layout()
            if isinstance(form, QtWidgets.QFormLayout):
                form.labelForField(self.trace_method_combo).setText(
                    "Identification method" if english else "识别方式"
                )
                form.labelForField(self.annular_radial_bins).setText(
                    "q-ring bins" if english else "q 环数量"
                )
                form.labelForField(self.annular_angle_bins).setText(
                    "Angular bins" if english else "方位角分箱"
                )
                form.labelForField(self.sector_width_spin).setText(
                    "Sector width" if english else "扇区宽度"
                )
                form.labelForField(self.sector_step_spin).setText(
                    "Sector step" if english else "扇区步长"
                )
            labels = (
                (
                    "Annular q-ring peaks · I(χ)"
                    if english
                    else "q 环方位峰 · I(χ)",
                    _TRACE_METHOD_ANNULAR_PEAK,
                ),
                (
                    "Sector-integrated radial peak · I(q)"
                    if english
                    else "扇区积分径向主峰 · I(q)",
                    _TRACE_METHOD_RADIAL_SECTOR,
                ),
                (
                    "Curvature local candidates · advanced"
                    if english
                    else "曲率局部候选 · 高级",
                    _TRACE_METHOD_CURVATURE,
                ),
            )
            for text, value in labels:
                index = self.trace_method_combo.findData(value)
                if index >= 0:
                    self.trace_method_combo.setItemText(index, text)
            self.trace_method_combo.setToolTip(
                "Annular mode integrates I(χ) on each q ring and links up to four lobe peaks; radial sectors and curvature are compatibility modes."
                if english
                else "环积分模式在每个 q 环上得到 I(χ)，沿 q 连接最多四个瓣峰；扇区积分和曲率保留为兼容模式。"
            )
            self.annular_radial_bins.setToolTip(
                "Number of q annuli used for I(χ) profiles and trajectory linking."
                if english
                else "用于生成 I(χ) 和连接轨迹的 q 环数量。"
            )
            self.annular_angle_bins.setToolTip(
                "Angular bins per annulus; masked angular gaps remain unsupported."
                if english
                else "每个 q 环的方位角分箱数；掩膜造成的方位缺口不补点。"
            )
            if hasattr(self, "identify_button"):
                if self._trace_method() == _TRACE_METHOD_ANNULAR_PEAK:
                    identify_tip = (
                        "Build I(χ) on each q annulus and link up to four lobe tracks."
                        if english
                        else "在每个 q 环构建 I(χ)，并沿 q 连接最多四条瓣轨迹。"
                    )
                else:
                    identify_tip = (
                        "Trace observed arcs using the selected butterfly method"
                        if english
                        else "使用当前蝴蝶识别方式提取观测轨迹"
                    )
                self.identify_button.setToolTip(identify_tip)
            self.sector_width_spin.setToolTip(
                "Azimuth width integrated into each radial I(q) profile."
                if english
                else "每个径向 I(q) 剖面所积分的方位角宽度。"
            )
            self.sector_step_spin.setToolTip(
                "Azimuth step between adjacent sector centers."
                if english
                else "相邻扇区中心之间的方位角步长。"
            )

        def _sync_trace_method_controls(self) -> None:
            method = self._trace_method()
            width = _finite_positive(self._settings.get("sector_width_deg"), 10.0)
            step = _finite_positive(self._settings.get("sector_step_deg"), 5.0)
            radial_bins = _bounded_int(
                self._settings.get("annular_radial_bins"),
                40,
                minimum=4,
                maximum=192,
            )
            angle_bins = _bounded_int(
                self._settings.get("annular_angle_bins"),
                72,
                minimum=16,
                maximum=720,
            )
            self.trace_method_combo.blockSignals(True)
            self.sector_width_spin.blockSignals(True)
            self.sector_step_spin.blockSignals(True)
            self.annular_radial_bins.blockSignals(True)
            self.annular_angle_bins.blockSignals(True)
            try:
                index = self.trace_method_combo.findData(method)
                self.trace_method_combo.setCurrentIndex(max(0, index))
                self.sector_width_spin.setValue(min(180.0, max(0.5, width)))
                self.sector_step_spin.setValue(min(180.0, max(0.5, step)))
                self.annular_radial_bins.setValue(radial_bins)
                self.annular_angle_bins.setValue(angle_bins)
            finally:
                self.trace_method_combo.blockSignals(False)
                self.sector_width_spin.blockSignals(False)
                self.sector_step_spin.blockSignals(False)
                self.annular_radial_bins.blockSignals(False)
                self.annular_angle_bins.blockSignals(False)
            self._settings["trace_method"] = method
            self._settings["sector_width_deg"] = float(self.sector_width_spin.value())
            self._settings["sector_step_deg"] = float(self.sector_step_spin.value())
            self._settings["annular_radial_bins"] = int(self.annular_radial_bins.value())
            self._settings["annular_angle_bins"] = int(self.annular_angle_bins.value())
            annular = method == _TRACE_METHOD_ANNULAR_PEAK
            for widget in (self.annular_radial_bins, self.annular_angle_bins):
                widget.setVisible(annular)
            for widget in (self.sector_width_spin, self.sector_step_spin):
                widget.setVisible(method == _TRACE_METHOD_RADIAL_SECTOR)
            if isinstance(getattr(self, "_trace_form", None), QtWidgets.QFormLayout):
                for widget, visible in (
                    (self.annular_radial_bins, annular),
                    (self.annular_angle_bins, annular),
                    (self.sector_width_spin, method == _TRACE_METHOD_RADIAL_SECTOR),
                    (self.sector_step_spin, method == _TRACE_METHOD_RADIAL_SECTOR),
                ):
                    label = self._trace_form.labelForField(widget)
                    if label is not None:
                        label.setVisible(visible)

        def _on_trace_settings_changed(self, *_: Any) -> None:
            method = _canonical_trace_method(
                self.trace_method_combo.currentData(),
                default=_TRACE_METHOD_CURVATURE,
            )
            width = float(self.sector_width_spin.value())
            step = float(self.sector_step_spin.value())
            radial_bins = int(self.annular_radial_bins.value())
            angle_bins = int(self.annular_angle_bins.value())
            changed = (
                method != self._trace_method()
                or width != _finite_positive(self._settings.get("sector_width_deg"), 10.0)
                or step != _finite_positive(self._settings.get("sector_step_deg"), 5.0)
                or radial_bins != _bounded_int(
                    self._settings.get("annular_radial_bins"),
                    40,
                    minimum=4,
                    maximum=192,
                )
                or angle_bins != _bounded_int(
                    self._settings.get("annular_angle_bins"),
                    72,
                    minimum=16,
                    maximum=720,
                )
            )
            self._settings.update(
                {
                    "trace_method": method,
                    "sector_width_deg": width,
                    "sector_step_deg": step,
                    "annular_radial_bins": radial_bins,
                    "annular_angle_bins": angle_bins,
                }
            )
            self._sync_trace_method_controls()
            self._render_trace_method_label()
            self._apply_trace_method_landmark_visibility()
            self._apply_trace_method_diagnostic_visibility()
            if not changed:
                return
            self.clear_result()
            self.analysisChanged.emit({"butterfly": self.butterfly_settings})
            self._sync_action_state()

        def _refresh_fit_layers(self) -> None:
            if not self._result_fresh or not self._result:
                self._fit_layers = {}
            else:
                try:
                    self._fit_layers = fit_geometry_layers(
                        self._result,
                        model_parameters=self._model_parameters,
                        model_reference_axis_deg=self._model_reference_axis_deg,
                        model_status=self._model_status,
                        model_diagnostics=self._model_diagnostics,
                    )
                except (TypeError, ValueError, OverflowError):
                    self._fit_layers = {}
            self.qspace.set_fit_layers(self._fit_layers)
            self._render_fit_source_state()
            self._render_fit_assessment()

        def set_model_fit_context(
            self,
            parameters: Any,
            *,
            reference_axis_deg: Any,
            solver_status: Any,
            diagnostics: Mapping[str, Any] | None = None,
        ) -> None:
            """Set the actual current Optimize model geometry for diagnosis.

            MainWindow calls this only for a generation-current Optimize result
            whose diagnostic signature still matches the live fit state.
            Preview arrays and edited/stale parameter tables never enter here.
            """

            if not self._result_fresh:
                self.clear_model_fit_context()
                return
            self._model_parameters = deepcopy(parameters)
            self._model_reference_axis_deg = _finite(reference_axis_deg)
            self._model_status = (
                None if solver_status in (None, "") else str(solver_status)
            )
            self._model_diagnostics = deepcopy(dict(diagnostics or {}))
            self._refresh_fit_layers()
            self._render_page_status()

        def clear_model_fit_context(self) -> None:
            self._model_parameters = None
            self._model_reference_axis_deg = None
            self._model_status = None
            self._model_diagnostics = {}
            self._refresh_fit_layers()

        def _geometry_assessment(self) -> Mapping[str, Any]:
            assessment = _read(self._fit_layers, ("assessment",), {})
            return assessment if isinstance(assessment, Mapping) else {}

        def _poor_geometry_fit(self) -> bool:
            assessment = self._geometry_assessment()
            if str(assessment.get("geometry_status", "")).lower() == "poor_match":
                return True
            ratio = _finite(assessment.get("residual_sigma_ratio"))
            limit = _finite(assessment.get("residual_limit"))
            return ratio is not None and limit is not None and ratio > limit

        def _render_fit_source_state(self) -> None:
            mode = str(self.overlay_mode_combo.currentData() or "measured_only")
            self.qspace.set_overlay_mode(mode)
            geometry = _read(self._fit_layers, ("geometry",), {})
            model = _read(self._fit_layers, ("intensity_model",), {})
            geometry_status = str(_read(geometry, ("status",), "unavailable"))
            model_status = str(_read(model, ("status",), "unavailable"))
            if mode == "measured_only":
                key = "fit.source_measured_only"
                values: dict[str, Any] = {}
            elif mode == "observed_ridges":
                key = "fit.source_observed_ridges"
                values = {}
            elif mode == "geometry_candidate":
                key = "fit.source_geometry"
                values = {"status": geometry_status}
            elif mode == "full2d_model":
                model_curves = _read(model, ("curves",), ())
                if not model_curves:
                    key = "fit.model_unavailable"
                    values = {}
                else:
                    key = "fit.source_model"
                    diagnostics = _read(model, ("diagnostics",), {})
                    condition = _finite(_read(diagnostics, ("condition_number",), None))
                    rmse = _finite(_read(diagnostics, ("rmse",), None))
                    bound_flags = _read(diagnostics, ("bound_flags",), {})
                    bound_names = (
                        ", ".join(
                            str(name)
                            for name, active in bound_flags.items()
                            if active
                        )
                        if isinstance(bound_flags, Mapping)
                        else ""
                    )
                    detail_parts = []
                    if rmse is not None:
                        detail_parts.append(f"RMSE {rmse:.3g}")
                    if condition is not None:
                        detail_parts.append(f"condition {condition:.3g}")
                    if bound_names:
                        detail_parts.append(f"bound flags: {bound_names}")
                    values = {
                        "status": model_status,
                        "details": "; ".join(detail_parts) or "diagnostics unavailable",
                    }
            else:
                key = "fit.source_compare"
                diagnostics = _read(model, ("diagnostics",), {})
                condition = _finite(_read(diagnostics, ("condition_number",), None))
                rmse = _finite(_read(diagnostics, ("rmse",), None))
                bound_flags = _read(diagnostics, ("bound_flags",), {})
                bound_names = (
                    ", ".join(
                        str(name)
                        for name, active in bound_flags.items()
                        if active
                    )
                    if isinstance(bound_flags, Mapping)
                    else ""
                )
                detail_parts = []
                if rmse is not None:
                    detail_parts.append(f"RMSE {rmse:.3g}")
                if condition is not None:
                    detail_parts.append(f"condition {condition:.3g}")
                if bound_names:
                    detail_parts.append(f"bound flags: {bound_names}")
                values = {
                    "geometry_status": geometry_status,
                    "model_status": model_status,
                    "details": "; ".join(detail_parts) or "diagnostics unavailable",
                }
            self.fit_source_label.setText(self._tr(key, **values))

        def _render_fit_assessment(self) -> None:
            assessment = self._geometry_assessment()
            if not assessment:
                self.fit_assessment_label.clear()
                self.fit_assessment_label.setAccessibleName("")
                self.fit_assessment_label.setVisible(False)
                return
            if self._poor_geometry_fit():
                ratio = _finite(assessment.get("residual_sigma_ratio"))
                limit = _finite(assessment.get("residual_limit"))
                text = self._tr(
                    "fit.geometry_poor",
                    ratio=_fmt(ratio),
                    limit=_fmt(limit),
                )
                self.fit_assessment_label.setStyleSheet(
                    "color: #9f1d16; background: #fff0ee; padding: 5px; font-weight: 700;"
                )
            else:
                status = str(assessment.get("geometry_status", "unavailable"))
                text = self._tr("fit.geometry_status", status=status)
                self.fit_assessment_label.setStyleSheet(
                    "color: #664d03; background: #fff8dc; padding: 5px;"
                )
            self._set_dynamic_accessible_text(self.fit_assessment_label, text)
            self.fit_assessment_label.setVisible(True)

        @staticmethod
        def _set_dynamic_accessible_text(label: Any, text: str) -> None:
            label.setText(text)
            label.setAccessibleName(text)
            try:
                QtGui.QAccessible.updateAccessibility(
                    QtGui.QAccessibleEvent(label, QtGui.QAccessible.Event.NameChanged)
                )
            except (AttributeError, TypeError):
                pass

        def _clear_diagnostic_layers(self) -> None:
            self._peak_landmarks = {}
            self._fit_layers = {}
            self._model_parameters = None
            self._model_reference_axis_deg = None
            self._model_status = None
            self._model_diagnostics = {}
            self._selected_landmark = {}
            self._selected_landmark_id = None
            self.qspace.set_peak_landmarks(None)
            self.qspace.set_selected_landmark(None)
            self.qspace.set_fit_layers(None)
            self.peak_table.blockSignals(True)
            self.peak_table.setRowCount(0)
            self.peak_table.clearSelection()
            self.peak_table.blockSignals(False)
            self.peak_angular_profile.clear(message_key="landmark.no_profiles")
            self.peak_radial_profile.clear(message_key="landmark.no_profiles")
            self.diagnostics_tabs.setCurrentIndex(0)
            self._reset_landmark_zoom()
            self._render_fit_source_state()
            self.fit_assessment_label.clear()
            self.fit_assessment_label.setAccessibleName("")
            self.fit_assessment_label.setVisible(False)

        def _render_peak_table(self) -> None:
            self.peak_table.blockSignals(True)
            self.peak_table.setRowCount(0)
            q_unit = str(
                self._peak_landmarks.get("q_unit")
                or self._frame_data.get("q_unit")
                or "q"
            )
            headers = (
                self._tr("header.landmark_id"),
                f"{self._tr('header.landmark_q')} ({q_unit})",
                self._tr("header.landmark_intensity"),
                self._tr("header.landmark_kind_status"),
            )
            self.peak_table.setHorizontalHeaderLabels(headers)
            english = self._language.lower().startswith("en")
            self.peak_table.setAccessibleName(
                "Peak landmark diagnostics" if english else "峰位标记诊断"
            )
            self.peak_table.setAccessibleDescription(
                "Global raw maximum and supported lobe peaks; selecting a row focuses its actual q coordinate"
                if english
                else "全局原始最大值与受支持瓣峰；选择行后聚焦该记录的实际 q 坐标"
            )
            rows: list[tuple[str, str, Mapping[str, Any]]] = []
            raw_max = self._peak_landmarks.get("raw_global_max")
            if isinstance(raw_max, Mapping):
                rows.append(("G", "raw_global_max", raw_max))
            peaks = self._peak_landmarks.get("peaks", ())
            if isinstance(peaks, Sequence) and not isinstance(peaks, (str, bytes)):
                for index, peak in enumerate(peaks):
                    if not isinstance(peak, Mapping):
                        continue
                    peak_id = str(_read(peak, ("peak_id",), f"P{index + 1}") or "")
                    if peak_id:
                        rows.append((peak_id, "supported_peak", peak))
            for landmark_id, kind, record in rows:
                row = self.peak_table.rowCount()
                self.peak_table.insertRow(row)
                if kind == "raw_global_max":
                    kind_text = self._tr("landmark.kind_raw_max")
                    status_text = self._tr("landmark.status_raw_only")
                else:
                    kind_text = self._tr("landmark.kind_supported")
                    flags = _read(record, ("flags",), ())
                    if isinstance(flags, str):
                        flags = (flags,) if flags else ()
                    flag_text = ", ".join(str(flag) for flag in (flags or ()) if flag)
                    status_text = (
                        self._tr("landmark.status_flags", flags=flag_text)
                        if flag_text
                        else self._tr("landmark.status_supported")
                    )
                values = (
                    landmark_id,
                    _fmt(_read(record, ("q",), None)),
                    _fmt(_read(record, ("raw_intensity",), None)),
                    f"{kind_text} · {status_text}",
                )
                for column, value in enumerate(values):
                    item = QtWidgets.QTableWidgetItem(value)
                    item.setData(
                        QtCore.Qt.ItemDataRole.UserRole,
                        {
                            "landmark_id": landmark_id,
                            "kind": kind,
                            "record": dict(record),
                        },
                    )
                    item.setToolTip(
                        str(record.get("interpretation"))
                        if kind == "raw_global_max"
                        else str(record.get("flags", ""))
                    )
                    self.peak_table.setItem(row, column, item)
            self.peak_table.blockSignals(False)

        def _on_landmark_visibility_changed(self, *_: Any) -> None:
            if not self._suppress_landmark_visibility_tracking:
                # A manual toggle takes ownership of the pixel-diagnostic
                # visibility. Switching methods or loading a result must not
                # undo the user's explicit choice.
                self._landmark_visibility_user_override = True
                self._auto_hidden_radial_landmarks = False
            self.qspace.set_landmark_visibility(
                global_raw_max=self.global_max_check.isChecked(),
                supported_peaks=self.supported_peaks_check.isChecked(),
            )

        def _apply_trace_method_landmark_visibility(self) -> None:
            """Keep pixel extrema secondary to radial sector profiles."""

            if self._trace_method() in {
                _TRACE_METHOD_RADIAL_SECTOR,
                _TRACE_METHOD_ANNULAR_PEAK,
            }:
                self._radial_sector_seen = True
                if self._landmark_visibility_user_override:
                    return
                self._auto_hidden_radial_landmarks = True
                self._suppress_landmark_visibility_tracking = True
                try:
                    self.global_max_check.setChecked(False)
                    self.supported_peaks_check.setChecked(False)
                finally:
                    self._suppress_landmark_visibility_tracking = False
                self.qspace.set_landmark_visibility(
                    global_raw_max=False,
                    supported_peaks=False,
                )
                return
            if (
                not self._auto_hidden_radial_landmarks
                or self._landmark_visibility_user_override
            ):
                return
            self._auto_hidden_radial_landmarks = False
            self._suppress_landmark_visibility_tracking = True
            try:
                self.global_max_check.setChecked(True)
                self.supported_peaks_check.setChecked(True)
            finally:
                self._suppress_landmark_visibility_tracking = False
            # Keep the automatic state alive so returning to radial mode
            # hides the two diagnostic layers again. A real user toggle above
            # clears this state and takes precedence.
            self._auto_hidden_radial_landmarks = True
            self.qspace.set_landmark_visibility(
                global_raw_max=True,
                supported_peaks=True,
            )

        def _apply_trace_method_diagnostic_visibility(self) -> None:
            """Give the radial I(q) profile the diagnostic height it needs."""

            self.ellipse_diagnostic.setVisible(
                self._trace_method()
                not in {_TRACE_METHOD_RADIAL_SECTOR, _TRACE_METHOD_ANNULAR_PEAK}
            )

        def _on_overlay_mode_changed(self, *_: Any) -> None:
            self._render_fit_source_state()

        def _peak_table_row(self, landmark_id: str) -> int | None:
            for row in range(self.peak_table.rowCount()):
                item = self.peak_table.item(row, 0)
                payload = item.data(QtCore.Qt.ItemDataRole.UserRole) if item else None
                if isinstance(payload, Mapping) and payload.get("landmark_id") == landmark_id:
                    return row
            return None

        def _on_landmark_selected(self, record: Any) -> None:
            if not isinstance(record, Mapping):
                return
            landmark_id = str(_read(record, ("peak_id",), "G") or "G")
            raw_max = self._peak_landmarks.get("raw_global_max")
            if record is raw_max:
                landmark_id = "G"
            row = self._peak_table_row(landmark_id)
            if row is None:
                return
            self.peak_table.setCurrentCell(row, 0)
            self._focus_landmark_row(row)

        def _on_peak_table_selection_changed(
            self,
            row: int,
            _column: int,
            _previous_row: int,
            _previous_column: int,
        ) -> None:
            self._focus_landmark_row(row)

        def _focus_landmark_row(self, row: int) -> None:
            if row < 0:
                return
            item = self.peak_table.item(row, 0)
            payload = item.data(QtCore.Qt.ItemDataRole.UserRole) if item else None
            if not isinstance(payload, Mapping):
                return
            record = payload.get("record")
            if not isinstance(record, Mapping):
                return
            landmark_id = str(payload.get("landmark_id", ""))
            self._selected_landmark = dict(record)
            self._selected_landmark_id = landmark_id
            self.qspace.set_selected_landmark(landmark_id)
            self._focus_landmark_q(record)
            self._render_peak_profiles(record, landmark_id)
            self.diagnostics_tabs.setCurrentIndex(1)

        def _focus_landmark_q(self, record: Mapping[str, Any]) -> None:
            qx = _finite(_read(record, ("qx",), None))
            qy = _finite(_read(record, ("qy",), None))
            if qx is None or qy is None:
                return
            radius = _finite(_read(record, ("q",), None))
            if radius is None:
                radius = math.hypot(qx, qy)
            bounds = self.qspace.base_q_bounds
            if bounds is None:
                return
            span = max(bounds[1] - bounds[0], bounds[3] - bounds[2])
            delta = max(abs(radius) * 0.12, span * 0.05, 1e-12)
            low = max(0.0, radius - delta)
            high = radius + delta
            if high <= low:
                high = low + max(delta, 1e-12)
            self.qspace.set_q_window((low, high))
            self._landmark_zoomed = True
            self.reset_peak_zoom_button.setEnabled(True)

        def _reset_landmark_zoom(self) -> None:
            self.qspace.set_q_window(self._requested_q_window)
            self._landmark_zoomed = False
            if hasattr(self, "reset_peak_zoom_button"):
                self.reset_peak_zoom_button.setEnabled(False)

        def _render_peak_profiles(self, record: Mapping[str, Any], landmark_id: str) -> None:
            _render_peak_profiles_impl(self, record, landmark_id)

        def _render_frame_source(self) -> None:
            frame = self._current_frame
            if frame is None or not str(frame):
                self.frame_source_label.setText(
                    "in-memory frame" if self._language.lower().startswith("en") else "内存帧"
                )
                return
            self.frame_source_label.setText(Path(str(frame)).name or str(frame))

        def _render_magnification_label(self) -> None:
            value = _fmt(self._diagnostic_magnification)
            self.magnification_label.setText(
                f"Display magnification: ×{value} (diagnostic display only)"
                if self._language.lower().startswith("en")
                else f"显示放大倍数：×{value}（仅用于诊断显示）"
            )

        def _data_ready(self) -> bool:
            observed = getattr(self.qspace, "observed", None)
            return bool(self._has_loaded_data and observed is not None)

        def _sync_action_state(self) -> None:
            ready = self._data_ready()
            busy = bool(self._busy)
            valid_range = not bool(self._q_range_error)
            self.identify_button.setEnabled(ready and valid_range and not busy)
            self.evaluate_button.setEnabled(ready and valid_range and not busy)
            self.apply_batch_button.setEnabled(ready and valid_range and not busy)
            self.cancel_button.setEnabled(busy)
            self.export_button.setEnabled(bool(self._result_fresh and not busy))
            self.figure_export_button.setEnabled(bool(self._result_fresh and not busy))

        @staticmethod
        def _result_is_failed(result: Mapping[str, Any]) -> bool:
            for key in ("status", "measurement_status", "solver_status", "quality_status"):
                value = str(result.get(key, "") or "").strip().lower()
                if value in {"fail", "failed", "error", "invalid"}:
                    return True
            metrics = result.get("metrics", result.get("statistics", result.get("summary", {})))
            if isinstance(metrics, Mapping) and metrics.get("success") is False:
                return True
            return False

        def _result_points(self) -> list[Any]:
            points = _read(self._result, ("points",), None)
            if isinstance(points, Sequence) and not isinstance(points, (str, bytes)):
                return list(points)
            sector_bundle = _read(self._result, ("sector_peaks",), {})
            points = _read(sector_bundle, ("points",), [])
            return list(points) if isinstance(points, Sequence) and not isinstance(points, (str, bytes)) else []

        def _is_sector_result(self) -> bool:
            if self._trace_method() != _TRACE_METHOD_RADIAL_SECTOR:
                return False
            if isinstance(_read(self._result, ("sector_peaks",), None), Mapping):
                return True
            for point in self._result_points():
                if isinstance(point, Mapping) and str(
                    _read(point, ("source_method",), "") or ""
                ).strip().lower() == _TRACE_METHOD_RADIAL_SECTOR:
                    return True
            return any(
                isinstance(profile, Mapping)
                and str(_read(profile, ("profile_axis",), "") or "").lower() == "radial"
                for profile in self._profiles.values()
            )

        def _is_annular_result(self) -> bool:
            if self._trace_method() != _TRACE_METHOD_ANNULAR_PEAK:
                return False
            if isinstance(_read(self._result, ("annular_peaks",), None), Mapping):
                return True
            return any(
                isinstance(profile, Mapping)
                and str(_read(profile, ("profile_axis",), "") or "").lower()
                in {"azimuthal", "angular", "chi"}
                for profile in self._profiles.values()
            )

        def _point_list_points(self) -> list[Any]:
            """Return selectable rows, including profile-only sectors."""

            if self._trace_method() == _TRACE_METHOD_RADIAL_SECTOR:
                sector_bundle = _read(self._result, ("sector_peaks",), {})
                sectors = _read(sector_bundle, ("sectors",), [])
                if (
                    isinstance(sectors, Sequence)
                    and not isinstance(sectors, (str, bytes))
                    and sectors
                ):
                    return list(sectors)
            if self._trace_method() == _TRACE_METHOD_ANNULAR_PEAK:
                annular_bundle = _read(self._result, ("annular_peaks",), {})
                annuli = _read(annular_bundle, ("annuli",), [])
                if (
                    isinstance(annuli, Sequence)
                    and not isinstance(annuli, (str, bytes))
                    and annuli
                ):
                    return list(annuli)
            return self._result_points()

        def _render_quality_summary(self) -> None:
            """Keep the compact evidence summary synchronized with page state."""

            trace_method = self._settings.get("trace_method", _TRACE_METHOD_ANNULAR_PEAK)
            # The page defaults to the new method for a new session, while a
            # loaded legacy result may carry no method metadata at all. Do
            # not reinterpret that old candidate's arc radius as a sector
            # median merely because the current controls have a new default.
            if (
                trace_method == _TRACE_METHOD_RADIAL_SECTOR
                and self._result
                and not self._is_sector_result()
            ):
                trace_method = _TRACE_METHOD_CURVATURE
            if (
                trace_method == _TRACE_METHOD_ANNULAR_PEAK
                and self._result
                and not self._is_annular_result()
            ):
                trace_method = _TRACE_METHOD_CURVATURE
            self.quality_summary.set_state(
                self._result if self._result_fresh else {},
                stage=str(self._settings.get("stage", "trace")),
                page_state=self._page_status_state,
                result_fresh=self._result_fresh,
                data_ready=self._data_ready(),
                busy=self._busy,
                q_unit=self._frame_data.get("q_unit"),
                poor_match=self._poor_geometry_fit(),
                error=self._page_status_error,
                trace_method=trace_method,
            )

        def _render_excluded_count(self) -> None:
            english = self._language.lower().startswith("en")
            self.excluded_count_label.setText(
                f"{self._excluded_count} excluded"
                if english
                else f"已排除 {self._excluded_count} 个"
            )
            self.excluded_count_label.setToolTip(
                "Points excluded from the active butterfly result"
                if english
                else "当前蝴蝶结果中未接受或无效的点数"
            )

        def _render_batch_feedback(self) -> None:
            if self._batch_success_count is None:
                self.batch_feedback_label.clear()
                return
            english = self._language.lower().startswith("en")
            success_count = int(self._batch_success_count)
            limited_count = int(self._batch_limited_count)
            failures = list(self._batch_failure_items)
            if failures:
                details = "; ".join(str(item) for item in failures[:4])
                if len(failures) > 4:
                    details += f" (+{len(failures) - 4})"
                self.batch_feedback_label.setText(
                    f"Batch completed: {success_count} frames retained, {limited_count} need review; failures: {details}"
                    if english
                    else f"批处理已完成：保留 {success_count} 帧结果供判读，其中 {limited_count} 帧需复核；失败：{details}"
                )
            elif limited_count:
                self.batch_feedback_label.setText(
                    f"Batch completed: {success_count} frames retained, {limited_count} need review"
                    if english
                    else f"批处理已完成：保留 {success_count} 帧结果供判读，其中 {limited_count} 帧需复核"
                )
            else:
                self.batch_feedback_label.setText(
                    f"Batch completed: {success_count} frame(s) retained for review"
                    if english
                    else f"批处理已完成：保留 {success_count} 帧结果供判读"
                )

        def _render_page_status(self) -> None:
            """Render the retained readiness/job/result state in the active language."""

            english = self._language.lower().startswith("en")
            state = self._page_status_state
            kind = self._page_status_kind or "analysis"
            kind_label = self._tr(f"job.{kind}")
            if kind_label == f"job.{kind}":
                kind_label = kind
            if state == "empty":
                text = (
                    "Load a frame before Identify or Evaluate"
                    if english
                    else "请先载入图像，再识别或评估"
                )
            elif state == "running":
                elapsed = max(0.0, float(self._job_elapsed_s or 0.0))
                text = self._tr("workflow.running", kind=kind_label, elapsed_s=elapsed)
                if self._job_progress_percent is not None:
                    text += self._tr(
                        "workflow.progress_suffix",
                        percent=self._job_progress_percent,
                        phase=self._job_progress_phase,
                    )
            elif state == "cancelling":
                elapsed = max(0.0, float(self._job_elapsed_s or 0.0))
                text = self._tr("workflow.cancelling", kind=kind_label, elapsed_s=elapsed)
            elif state == "cancelled":
                text = "Cancelled" if english else "已取消"
            elif state == "failed":
                detail = str(self._page_status_error or "").strip()
                visible_detail = detail if len(detail) <= 100 else detail[:97] + "…"
                suffix = f": {visible_detail}" if visible_detail else ""
                text = f"Failed · {kind_label}{suffix}" if english else f"失败 · {kind_label}{suffix}"
            elif state == "result":
                count = len(self._result_points())
                stage = self._settings.get("stage", "trace")
                text = (
                    f"Result · {stage} · {count} points"
                    if english
                    else f"结果 · {'评估' if stage == 'evaluate' else '追踪'} · {count} 个点"
                )
            elif state == "completed":
                count = len(self._result_points())
                text = (
                    f"Completed · {kind_label} · {count} points"
                    if english
                    else f"已完成 · {kind_label} · {count} 个点"
                )
            else:
                if english:
                    text = "Ready · identify arcs" if self._data_ready() else "Load a frame to begin"
                else:
                    text = "就绪 · 请识别弧线" if self._data_ready() else "请先载入一帧图像"
            self.status_label.setText(text)
            if state in {"result", "completed"} and self._poor_geometry_fit():
                self.status_label.setText(
                    f"{text} · {self._tr('fit.poor_short')}"
                )
            self.status_label.setToolTip(
                str(self._page_status_error or "") if state == "failed" else ""
            )
            self._render_workflow_hint()
            self._render_quality_summary()

        def _render_workflow_hint(self) -> None:
            """Show the next useful operation without implying scientific acceptance."""

            if self._q_range_error:
                key = "workflow.fix_q_range"
                values: dict[str, Any] = {}
            elif self._page_status_state in {"running", "cancelling"}:
                key = "workflow.running_hint"
                values = {}
            elif not self._data_ready():
                key = "workflow.next_load"
                values = {}
            elif self._page_status_state == "failed":
                key = "workflow.next_retry"
                values = {}
            elif self._result_fresh:
                if self._settings.get("stage", "trace") == "trace":
                    key = "workflow.next_evaluate"
                    values = {"resamples": self._evaluation_resamples()}
                else:
                    key = "workflow.next_review"
                    values = {}
            else:
                key = "workflow.next_identify"
                values = {}
            self._set_dynamic_accessible_text(
                self.workflow_hint_label, self._tr(key, **values)
            )

        def set_language(self, language: str) -> None:
            self._language = str(language)
            english = self._language.lower().startswith("en")
            self.qspace.set_language(self._language)
            self.quality_summary.set_language(self._language)
            self.title_label.setText("Butterfly analysis" if english else "蝴蝶分析 / Butterfly analysis")
            self._render_trace_method_controls()
            self._render_trace_method_label()
            self.frame_title_label.setText("Frames" if english else "帧 / Frames")
            self.point_title_label.setText("Points" if english else "点 / Points")
            self.frame_list.setAccessibleName("Frames" if english else "帧列表")
            self.frame_list.setAccessibleDescription(
                "Select the active SAXS frame" if english else "选择当前 SAXS 帧"
            )
            self.point_list.setAccessibleName("Points" if english else "测量点")
            self.point_list.setAccessibleDescription(
                "Select a measured butterfly point" if english else "选择一个蝴蝶测量点"
            )
            self._render_frame_source()
            analysis_group = self.findChild(QtWidgets.QGroupBox, "butterflyAnalysisRange")
            analysis_group.setTitle("Analysis range" if english else "分析范围")
            analysis_form = analysis_group.layout()
            analysis_form.labelForField(self.q_min_edit).setText("q min" if english else "q 下限")
            analysis_form.labelForField(self.q_max_edit).setText("q max" if english else "q 上限")
            analysis_form.labelForField(self.reference_axis_spin).setText(
                "Reference axis (deg)" if english else "图样参考轴（deg）"
            )
            evaluation_group = self.findChild(QtWidgets.QGroupBox, "butterflyEvaluationControls")
            evaluation_group.setTitle("Evaluation" if english else "评估设置")
            evaluation_form = evaluation_group.layout()
            evaluation_form.labelForField(self.evaluation_resamples_combo).setText(
                "Uncertainty resamples" if english else "不确定度重采样次数"
            )
            self.sensitivity_check.setText(
                "Run sensitivity checks" if english else "运行敏感性检查"
            )
            for index in range(self.evaluation_resamples_combo.count()):
                value = int(self.evaluation_resamples_combo.itemData(index))
                if value == 0:
                    preset = "Quick fit" if english else "快速拟合"
                    label = self._tr("evaluation.budget", preset=preset, count=value)
                elif value == 32:
                    preset = "Standard" if english else "标准"
                    label = self._tr("evaluation.budget", preset=preset, count=value)
                elif value == 128:
                    preset = "Extended" if english else "扩展"
                    label = self._tr("evaluation.budget", preset=preset, count=value)
                else:
                    label = self._tr("evaluation.custom", count=value)
                self.evaluation_resamples_combo.setItemText(index, label)
            self._sync_evaluation_controls()
            self.findChild(QtWidgets.QGroupBox, "butterflyFitOverlays").setTitle(
                self._tr("group.fit_overlays")
            )
            overlay_form = self.findChild(
                QtWidgets.QGroupBox, "butterflyFitOverlays"
            ).layout().itemAt(0).layout()
            overlay_form.labelForField(self.overlay_mode_combo).setText(
                self._tr("label.overlay_mode")
            )
            overlay_keys = (
                ("overlay.measured_only", "measured_only"),
                ("overlay.observed_ridges", "observed_ridges"),
                ("overlay.geometry_candidate", "geometry_candidate"),
                ("overlay.full2d_model", "full2d_model"),
                ("overlay.compare", "compare"),
            )
            for key, mode in overlay_keys:
                index = self.overlay_mode_combo.findData(mode)
                if index >= 0:
                    self.overlay_mode_combo.setItemText(index, self._tr(key))
            self.findChild(QtWidgets.QGroupBox, "butterflyPeakLandmarks").setTitle(
                self._tr("group.peak_landmarks")
            )
            self.global_max_check.setText(self._tr("check.raw_global_max"))
            self.supported_peaks_check.setText(self._tr("check.supported_peaks"))
            self.global_max_check.setToolTip(
                "Pixel raw-maximum diagnostic layer; it does not select the sector q*."
                if english
                else "像素原始最大值诊断图层；它不用于选择扇区 q*。"
            )
            self.supported_peaks_check.setToolTip(
                "Pixel-supported-peak diagnostic layer; it does not replace sector-integrated I(q)."
                if english
                else "像素支持峰诊断图层；它不替代扇区积分 I(q)。"
            )
            self._apply_trace_method_landmark_visibility()
            self._apply_trace_method_diagnostic_visibility()
            self.reset_peak_zoom_button.setText(self._tr("button.reset_peak_zoom"))
            previous_landmark = self._selected_landmark_id
            self._render_peak_table()
            if previous_landmark:
                selected_row = self._peak_table_row(previous_landmark)
                if selected_row is not None:
                    self.peak_table.setCurrentCell(selected_row, 0)
            self.normal_profile.set_language(english=english)
            self.ellipse_diagnostic.set_language(english=english)
            self.peak_angular_profile.set_language(english=english)
            self.peak_radial_profile.set_language(english=english)
            if self._selected_profile_point_id:
                selected_point = None
                for row in range(self.point_list.count()):
                    item = self.point_list.item(row)
                    candidate = item.data(QtCore.Qt.ItemDataRole.UserRole) if item else None
                    if isinstance(candidate, Mapping) and str(
                        candidate.get("point_id", "")
                    ) == self._selected_profile_point_id:
                        selected_point = candidate
                        break
                self._render_profile(self._selected_profile_point_id, selected_point)
            self.diagnostics_tabs.setTabText(
                0, "Point diagnostics" if english else "测量点诊断"
            )
            self.diagnostics_tabs.setTabText(
                1, "Peak profiles" if english else "峰位剖面"
            )
            self._render_peak_profiles(
                self._selected_landmark, self._selected_landmark_id
            ) if self._selected_landmark and self._selected_landmark_id else None
            self.findChild(QtWidgets.QGroupBox, "butterflyDisplayControls").setTitle(
                "Display" if english else "显示"
            )
            display_form = self.findChild(QtWidgets.QGroupBox, "butterflyDisplayControls").layout()
            display_form.labelForField(self.display_scale_combo).setText("Scale" if english else "对比度")
            display_form.labelForField(self.display_percentile_spin).setText(
                "Upper percentile" if english else "上分位数"
            )
            display_form.labelForField(self.uv_magnification_spin).setText(
                "u/v magnification" if english else "u/v 显示放大"
            )
            self.display_reset_button.setText("Reset contrast" if english else "重置对比度")
            self.findChild(QtWidgets.QGroupBox, "branchSideVisibility").setTitle(
                "Branch / side visibility" if english else "分支 / 侧边可见性"
            )
            labels = {
                (0, "upper"): "A · upper" if english else "A · 上侧",
                (0, "lower"): "A · lower" if english else "A · 下侧",
                (1, "upper"): "B · upper" if english else "B · 上侧",
                (1, "lower"): "B · lower" if english else "B · 下侧",
            }
            for key, check in self._branch_checks.items():
                if key in labels:
                    check.setText(labels[key])
                elif key == (-1, "unknown"):
                    check.setText("Unknown" if english else "未知")
            self.show_excluded_check.setText("Show excluded" if english else "显示排除点")
            self.show_excluded_check.setToolTip(self._tr("tooltip.show_excluded_points"))
            self.findChild(QtWidgets.QGroupBox, "butterflyQuantitativeParameters").setTitle(
                "Quantitative parameters" if english else "定量参数"
            )
            self.quantity_table.setHorizontalHeaderLabels(
                ("Param", "Value", "State", "Cand.", "CI", "Why")
                if english
                else ("参数", "数值", "状态", "候选", "区间", "原因")
            )
            self.quantity_table.setToolTip(
                "Candidate values are cached/unvalidated and are not quantitative values."
                if english
                else "候选值为缓存且未验证，仅供诊断，不等同于定量值。"
            )
            if self._result:
                self._render_quantities(
                    _read(self._result, ("quantitative_parameters",), {}) or {}
                )
            self.findChild(QtWidgets.QGroupBox, "butterflyCorrections").setTitle(
                "Corrections" if english else "校正 / 编辑"
            )
            self.findChild(QtWidgets.QGroupBox, "butterflyWorkflow").setTitle(
                "Workflow" if english else "工作流"
            )
            correction_text = (
                ("Seed point", "Exclude point", "Exclude rectangle", "Include rectangle", "Exclude polygon", "Include polygon")
                if english
                else ("种子点", "排除点", "排除矩形", "包含矩形", "排除多边形", "包含多边形")
            )
            for index, text in enumerate(correction_text):
                self.correction_mode_combo.setItemText(index, text)
            self.seed_branch_combo.setItemText(0, "Auto / unknown" if english else "自动 / 未知")
            self.seed_branch_combo.setItemText(1, "A" if english else "A 分支")
            self.seed_branch_combo.setItemText(2, "B" if english else "B 分支")
            self.seed_side_combo.setItemText(0, "Unknown" if english else "未知")
            self.seed_side_combo.setItemText(1, "Upper" if english else "上侧")
            self.seed_side_combo.setItemText(2, "Lower" if english else "下侧")
            self.correct_button.setText("Correct" if english else "校正 / 编辑")
            self.undo_button.setText("Undo" if english else "撤销")
            self.redo_button.setText("Redo" if english else "重做")
            self.numeric_edit_button.setText("Keyboard edit" if english else "键盘编辑")
            correction_form = self.findChild(QtWidgets.QGroupBox, "butterflyCorrections").layout()
            correction_form.labelForField(self.correction_mode_combo).setText(
                "Edit mode" if english else "编辑模式"
            )
            correction_form.labelForField(self.seed_branch_combo).setText(
                "Seed branch" if english else "种子分支"
            )
            correction_form.labelForField(self.seed_side_combo).setText(
                "Seed side" if english else "种子侧边"
            )
            self.identify_button.setText("Identify arcs" if english else "识别弧线")
            self.evaluate_button.setText("Evaluate" if english else "评估结果")
            self.apply_batch_button.setText("Apply to batch" if english else "应用到批处理")
            self.cancel_button.setText("Cancel" if english else "取消")
            self.export_button.setText("Export evidence" if english else "导出分析证据")
            self.figure_export_button.setText(self._tr("button.butterfly_figure"))
            self.figure_export_button.setToolTip(self._tr("tooltip.butterfly_figure"))
            self.evaluate_button.setToolTip(
                self._tr("tooltip.butterfly_evaluate", resamples=self._evaluation_resamples())
            )
            _, _, q_error = self._parse_analysis_q_range()
            self._q_range_error = q_error or ""
            self._render_q_range_feedback()
            self._render_fit_source_state()
            self._render_fit_assessment()
            self._render_legacy_banner()
            self._render_magnification_label()
            self._render_excluded_count()
            self._render_batch_feedback()
            self._render_page_status()
            self._retranslate_q_star_source_cells()
            self._sync_action_state()
            from .help import apply_help

            apply_help(self, self._language)

        def _retranslate_q_star_source_cells(self) -> None:
            for row in range(self.quantity_table.rowCount()):
                item = self.quantity_table.item(row, 2)
                if item is None or item.data(QtCore.Qt.ItemDataRole.UserRole + 1) != "q_star_source":
                    continue
                text = translate_q_star_source(
                    self._language,
                    item.data(QtCore.Qt.ItemDataRole.UserRole),
                )
                display = "" if text is None else str(text)
                item.setText(display)
                item.setToolTip(display)

        def eventFilter(self, watched: Any, event: Any) -> bool:  # noqa: N802 - Qt API
            if watched is getattr(self, "point_list", None) and event.type() == QtCore.QEvent.Type.KeyPress:
                if event.key() == QtCore.Qt.Key.Key_Delete:
                    self._exclude_selected_point()
                    return True
                if event.key() in (QtCore.Qt.Key.Key_Return, QtCore.Qt.Key.Key_Enter):
                    self._on_point_list_changed(self.point_list.currentRow())
                    return True
            return super().eventFilter(watched, event)

        def set_frames(self, frames: Sequence[Any] | None, *, current: Any = None) -> None:
            self._frames = list(frames or ())
            self.frame_list.blockSignals(True)
            self.frame_list.clear()
            for frame in self._frames:
                item = QtWidgets.QListWidgetItem(str(frame))
                item.setData(QtCore.Qt.ItemDataRole.UserRole, frame)
                self.frame_list.addItem(item)
            self.frame_list.blockSignals(False)
            self.frame_count_label.setText(str(len(self._frames)))
            if current is not None:
                for index, frame in enumerate(self._frames):
                    if str(frame) == str(current):
                        self.frame_list.setCurrentRow(index)
                        break
            elif self._frames and self.frame_list.currentRow() < 0:
                self.frame_list.setCurrentRow(0)

        def set_current_frame(self, frame: Any) -> None:
            self._current_frame = frame
            self._render_frame_source()
            if frame is None:
                return
            for index, value in enumerate(self._frames):
                if str(value) == str(frame):
                    self.frame_list.setCurrentRow(index)
                    return

        def _on_frame_changed(self, row: int) -> None:
            if row < 0 or row >= len(self._frames):
                return
            frame = self._frames[row]
            if str(frame) != str(self._current_frame):
                self.invalidate_result()
            self.set_current_frame(frame)
            self.frameSelected.emit(frame)

        def set_data(
            self,
            observed: Any = None,
            *,
            qx: Any = None,
            qy: Any = None,
            valid_mask: Any = None,
            q_unit: str | None = None,
            source: Any = None,
        ) -> None:
            self._frame_data = {
                "observed": observed,
                "qx": qx,
                "qy": qy,
                "valid_mask": valid_mask,
                "q_unit": q_unit,
            }
            self.set_current_frame(source)
            self.qspace.set_data(observed, qx=qx, qy=qy, valid_mask=valid_mask, q_unit=q_unit)
            self.qspace.set_q_window(self._requested_q_window)
            self._has_loaded_data = getattr(self.qspace, "observed", None) is not None
            # A frame is a new scientific input.  Remove every result/selection
            # that could otherwise be painted over the new q-map.
            self.clear_result(
                message=(
                    "Ready · identify arcs"
                    if self._has_loaded_data and self._language.lower().startswith("en")
                    else "就绪 · 请识别弧线"
                    if self._has_loaded_data
                    else "Load a frame to begin"
                    if self._language.lower().startswith("en")
                    else "请先载入一帧图像"
                )
            )
            self._page_status_state = "ready" if self._has_loaded_data else "empty"
            self._render_page_status()
            self._sync_action_state()

        set_observed_data = set_data

        def set_display_settings(self, scale: str, percentile: float) -> None:
            # Contrast is intentionally display-only and does not emit an edit
            # or an analysis request.
            mode = str(scale or "linear").strip().lower().replace("-", "_")
            index = self.display_scale_combo.findData(mode)
            self.display_scale_combo.blockSignals(True)
            self.display_percentile_spin.blockSignals(True)
            try:
                self.display_scale_combo.setCurrentIndex(max(0, index))
                self.display_percentile_spin.setValue(float(percentile))
            finally:
                self.display_scale_combo.blockSignals(False)
                self.display_percentile_spin.blockSignals(False)
            self.qspace.set_display_settings(scale, percentile)

        @property
        def result_fresh(self) -> bool:
            return bool(self._result_fresh)

        @property
        def export_context(self) -> dict[str, Any]:
            return deepcopy(self._export_context)

        @property
        def display_settings(self) -> dict[str, Any]:
            return {
                "scale": str(self.display_scale_combo.currentData() or "linear"),
                "percentile": float(self.display_percentile_spin.value()),
            }

        def set_export_context(self, context: Mapping[str, Any] | None) -> None:
            self._export_context = deepcopy(dict(context or {}))

        def export_analysis(self, path: str | Path) -> dict[str, Path]:
            return export_butterfly_analysis(self, path, context=self._export_context)

        def figure_export_snapshot(
            self,
            *,
            context: Mapping[str, Any] | None = None,
        ) -> dict[str, Any]:
            """Copy the current physical image/result inputs for a worker export."""

            if not self._result_fresh or not self._result:
                raise ValueError("a fresh butterfly result is required for figure export")
            if _np is None:
                raise RuntimeError("NumPy is required for measurement figure export")

            def frozen_array(name: str, value: Any, *, optional: bool = False) -> Any:
                if value is None:
                    if optional:
                        return None
                    raise ValueError(f"measurement figure export requires {name}")
                try:
                    array = _np.array(value, copy=True, subok=True)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"measurement figure input {name} is not an array") from exc
                if array.ndim != 2 or not array.size:
                    raise ValueError(f"measurement figure input {name} must be a non-empty 2D array")
                array.setflags(write=False)
                return array

            observed = frozen_array("observed", self._frame_data.get("observed"))
            qx = frozen_array("qx", self._frame_data.get("qx"))
            qy = frozen_array("qy", self._frame_data.get("qy"))
            mask = frozen_array("valid_mask", self._frame_data.get("valid_mask"), optional=True)
            if qx.shape != observed.shape or qy.shape != observed.shape:
                raise ValueError("observed, qx and qy arrays must have the same 2D shape")
            if mask is not None and mask.shape != observed.shape:
                raise ValueError("valid_mask must have the same 2D shape as observed")

            result = deepcopy(self._result)

            def freeze_nested(value: Any) -> None:
                if isinstance(value, _np.ndarray):
                    value.setflags(write=False)
                elif isinstance(value, Mapping):
                    for item in value.values():
                        freeze_nested(item)
                elif isinstance(value, (list, tuple)):
                    for item in value:
                        freeze_nested(item)

            freeze_nested(result)
            return {
                "observed": observed,
                "qx": qx,
                "qy": qy,
                "valid_mask": mask,
                "result": result,
                "q_unit": str(self._frame_data.get("q_unit") or "unknown"),
                "display_scale": self.display_settings["scale"],
                "context": deepcopy(dict(context or self._export_context)),
                "result_revision": self._result_revision,
            }

        def _sync_export_state(self) -> None:
            self._sync_action_state()

        def invalidate_result(self) -> None:
            """Remove every displayed measurement derived from a stale input."""

            self.clear_result()

        def _on_analysis_range_changed(self, *_: Any) -> None:
            q_min, q_max, error = self._parse_analysis_q_range()
            self._q_range_error = error or ""
            self._render_q_range_feedback()
            self._sync_action_state()
            if error:
                self._render_page_status()
                return
            values = (q_min, q_max, float(self.reference_axis_spin.value()))
            if values == self._last_valid_analysis_range:
                self._render_page_status()
                return
            self._last_valid_analysis_range = values
            self.invalidate_result()
            self.analysisChanged.emit(
                {
                    "q_min": q_min,
                    "q_max": q_max,
                    "draw_axis_deg": float(self.reference_axis_spin.value()) + 90.0,
                }
            )

        def _on_display_changed(self, *_: Any) -> None:
            scale = str(self.display_scale_combo.currentData() or "linear")
            percentile = float(self.display_percentile_spin.value())
            self.qspace.set_display_settings(scale, percentile)
            self.displayChanged.emit(scale, percentile)

        def _on_uv_magnification_changed(self, value: float) -> None:
            self._uv_display_magnification = float(value)
            self._render_ellipse_local()

        def _reset_display(self) -> None:
            self.set_display_settings("linear", 99.5)
            self.displayChanged.emit("linear", 99.5)

        def _parse_analysis_q_range(
            self,
        ) -> tuple[float | None, float | None, str | None]:
            values: list[float | None] = []
            for key, edit in (("label.q_min", self.q_min_edit), ("label.q_max", self.q_max_edit)):
                text = edit.text().strip()
                if text.lower() in {"", "auto", "自动"}:
                    values.append(None)
                    continue
                try:
                    value = float(text)
                except (TypeError, ValueError):
                    return None, None, self._tr("workflow.q_range_number", field=self._tr(key))
                if not math.isfinite(value):
                    return None, None, self._tr("workflow.q_range_number", field=self._tr(key))
                values.append(value)
            q_min, q_max = values
            if q_min is not None and q_max is not None and q_min >= q_max:
                return q_min, q_max, self._tr("workflow.q_range_order")
            return q_min, q_max, None

        def _render_q_range_feedback(self) -> None:
            error = str(self._q_range_error or "")
            self.q_range_error_label.setText(error)
            self.q_range_error_label.setVisible(bool(error))
            for edit in (self.q_min_edit, self.q_max_edit):
                edit.setProperty("invalid", bool(error))
                edit.setToolTip(error)
                edit.style().unpolish(edit)
                edit.style().polish(edit)

        def _evaluation_resamples(self) -> int:
            try:
                return int(self._settings.get("evaluation_resamples", 32))
            except (TypeError, ValueError):
                return 32

        def _sync_evaluation_controls(self) -> None:
            value = self._evaluation_resamples()
            index = self.evaluation_resamples_combo.findData(value)
            if index < 0:
                label = self._tr("evaluation.custom", count=value)
                self.evaluation_resamples_combo.addItem(label, value)
                index = self.evaluation_resamples_combo.findData(value)
            self.evaluation_resamples_combo.blockSignals(True)
            self.sensitivity_check.blockSignals(True)
            try:
                self.evaluation_resamples_combo.setCurrentIndex(max(0, index))
                self.sensitivity_check.setChecked(bool(self._settings.get("sensitivity", True)))
            finally:
                self.evaluation_resamples_combo.blockSignals(False)
                self.sensitivity_check.blockSignals(False)

        def _on_evaluation_settings_changed(self, *_: Any) -> None:
            selected = self.evaluation_resamples_combo.currentData()
            if selected is None:
                return
            value = int(selected)
            changed = (
                value != self._evaluation_resamples()
                or bool(self.sensitivity_check.isChecked())
                != bool(self._settings.get("sensitivity", True))
            )
            if not changed:
                return
            self._settings["evaluation_resamples"] = value
            if self._settings.get("stage") == "evaluate":
                self._settings["resamples"] = value
            self._settings["sensitivity"] = bool(self.sensitivity_check.isChecked())
            self.clear_result()
            self.analysisChanged.emit({"butterfly": self.butterfly_settings})
            self._sync_action_state()

            from .help import refresh_evaluation_help

            refresh_evaluation_help(self, self._language)

        def set_q_window(self, q_window: Sequence[Any] | None = None) -> None:
            if q_window is None:
                self._requested_q_window = None
            else:
                try:
                    low, high = (float(value) for value in q_window)
                    if math.isfinite(low) and math.isfinite(high) and high > low:
                        self._requested_q_window = (low, high)
                except (TypeError, ValueError):
                    return
            self._landmark_zoomed = False
            self.reset_peak_zoom_button.setEnabled(False)
            self.qspace.set_q_window(q_window)

        def set_analysis_settings(
            self,
            settings: Mapping[str, Any] | None,
            *,
            replace: bool = False,
        ) -> None:
            if not isinstance(settings, Mapping):
                return
            previous_range = self._last_valid_analysis_range
            if any(key in settings for key in ("q_min", "q_max", "draw_axis_deg")):
                widgets = (self.q_min_edit, self.q_max_edit, self.reference_axis_spin)
                for widget in widgets:
                    widget.blockSignals(True)
                try:
                    for edit, key in ((self.q_min_edit, "q_min"), (self.q_max_edit, "q_max")):
                        if key in settings:
                            value = settings.get(key)
                            edit.setText("Auto" if value in (None, "") else str(value))
                    if settings.get("draw_axis_deg") is not None:
                        self.reference_axis_spin.setValue(float(settings["draw_axis_deg"]) - 90.0)
                except (TypeError, ValueError):
                    pass
                finally:
                    for widget in widgets:
                        widget.blockSignals(False)
            q_min, q_max, q_error = self._parse_analysis_q_range()
            self._q_range_error = q_error or ""
            q_changed = False
            if not q_error:
                values = (q_min, q_max, float(self.reference_axis_spin.value()))
                q_changed = values != previous_range
                self._last_valid_analysis_range = values
            self._render_q_range_feedback()
            recipe_keys = {
                "stage",
                "trace_method",
                "sector_width_deg",
                "sector_step_deg",
                "annular_radial_bins",
                "annular_angle_bins",
                "resamples",
                "evaluation_resamples",
                "seed",
                "edits",
                "sensitivity",
                "max_nfev",
            }
            nested = settings.get("butterfly")
            if not isinstance(nested, Mapping):
                analysis = settings.get("analysis")
                if isinstance(analysis, Mapping) and isinstance(analysis.get("butterfly"), Mapping):
                    nested = analysis["butterfly"]
                elif isinstance(analysis, Mapping):
                    nested = {key: analysis[key] for key in recipe_keys if key in analysis}
                else:
                    nested = {key: settings[key] for key in recipe_keys if key in settings}
            if not nested:
                if q_changed:
                    self.invalidate_result()
                self._sync_action_state()
                self._render_page_status()
                return
            if not isinstance(nested, Mapping):
                return
            recipe_source = {} if replace else deepcopy(self._settings)
            recipe_source.update(deepcopy(dict(nested)))
            explicit_method = nested.get("trace_method")
            if explicit_method in (None, ""):
                explicit_method = settings.get("trace_method")
            if explicit_method in (None, ""):
                explicit_method = settings.get("ridge_method")
            if explicit_method not in (None, ""):
                recipe_source["trace_method"] = _canonical_trace_method(
                    explicit_method,
                    default=_TRACE_METHOD_CURVATURE,
                )
            elif replace and "trace_method" not in nested:
                # A replaced recipe without the new field is an old project
                # recipe.  Keep its curvature semantics instead of silently
                # upgrading it to the new annular-trajectory default.
                recipe_source["trace_method"] = _TRACE_METHOD_CURVATURE
            if "evaluation_resamples" in nested:
                recipe_source["evaluation_resamples"] = nested["evaluation_resamples"]
            elif recipe_source.get("stage") == "evaluate" and "resamples" in nested:
                # An Evaluate recipe's explicit resampling count is its
                # effective uncertainty budget unless the separate saved
                # budget was explicitly supplied. Do not let a default or
                # previous-stage cached budget override that project value.
                recipe_source["evaluation_resamples"] = recipe_source.get("resamples")
            elif "evaluation_resamples" not in recipe_source:
                recipe_source["evaluation_resamples"] = (
                    32 if replace else self._evaluation_resamples()
                )
            try:
                recipe_source["evaluation_resamples"] = normalize_butterfly_settings(
                    {"resamples": recipe_source["evaluation_resamples"]}
                )["resamples"]
            except ValueError as exc:
                raise ValueError(
                    "butterfly evaluation_resamples must be a non-negative integer"
                ) from exc
            normalized = normalize_butterfly_settings(recipe_source)
            normalized["trace_method"] = _canonical_trace_method(
                normalized.get("trace_method"),
                default=_TRACE_METHOD_CURVATURE if replace else self._trace_method(),
            )
            normalized["sector_width_deg"] = min(
                180.0,
                max(0.5, _finite_positive(normalized.get("sector_width_deg"), 10.0)),
            )
            normalized["sector_step_deg"] = min(
                180.0,
                max(0.5, _finite_positive(normalized.get("sector_step_deg"), 5.0)),
            )
            normalized["annular_radial_bins"] = _bounded_int(
                normalized.get("annular_radial_bins"),
                40,
                minimum=4,
                maximum=192,
            )
            normalized["annular_angle_bins"] = _bounded_int(
                normalized.get("annular_angle_bins"),
                72,
                minimum=16,
                maximum=720,
            )
            old_settings = normalize_butterfly_settings(self._settings)
            old_settings["trace_method"] = _canonical_trace_method(
                old_settings.get("trace_method"),
                default=_TRACE_METHOD_ANNULAR_PEAK,
            )
            old_settings["sector_width_deg"] = min(
                180.0,
                max(0.5, _finite_positive(old_settings.get("sector_width_deg"), 10.0)),
            )
            old_settings["sector_step_deg"] = min(
                180.0,
                max(0.5, _finite_positive(old_settings.get("sector_step_deg"), 5.0)),
            )
            old_settings["annular_radial_bins"] = _bounded_int(
                old_settings.get("annular_radial_bins"),
                40,
                minimum=4,
                maximum=192,
            )
            old_settings["annular_angle_bins"] = _bounded_int(
                old_settings.get("annular_angle_bins"),
                72,
                minimum=16,
                maximum=720,
            )
            self._settings = normalized
            edits = normalized.get("edits", [])
            self._edits = [dict(edit) for edit in edits if isinstance(edit, Mapping)]
            if replace or "edits" in nested or normalized.get("edits") != old_settings.get("edits"):
                self._redo_edits.clear()
            self.qspace.set_edits(self._edits)
            self._update_edit_buttons()
            self._sync_evaluation_controls()
            self._sync_trace_method_controls()
            self._render_trace_method_label()
            self._apply_trace_method_landmark_visibility()
            self._apply_trace_method_diagnostic_visibility()
            if normalized != old_settings:
                self.clear_result(
                    message=(
                        "Settings changed · rerun Identify or Evaluate"
                        if self._language.lower().startswith("en")
                        else "设置已更改 · 请重新识别或评估"
                    )
                )
            elif q_changed:
                self.invalidate_result()
            self._sync_action_state()
            self._render_page_status()

        def reset_analysis_settings(self) -> None:
            """Reset the page recipe when a legacy project has no recipe."""

            legacy_recipe = deepcopy(DEFAULT_BUTTERFLY_SETTINGS)
            legacy_recipe["trace_method"] = _TRACE_METHOD_CURVATURE
            self.set_analysis_settings(legacy_recipe, replace=True)

        def set_legacy_method(self, method: Any) -> None:
            value = str(method or "radial_peak")
            self._legacy_method = None if value == "butterfly_curvature" else value
            self._render_legacy_banner()

        def _render_legacy_banner(self) -> None:
            if self._legacy_method:
                english = self._language.lower().startswith("en")
                if english:
                    self.legacy_banner.setText(
                        f"Loaded legacy ridge method '{self._legacy_method}'. "
                        "Choose Identify arcs to explicitly use butterfly_curvature; the project method is preserved."
                    )
                else:
                    self.legacy_banner.setText(
                        f"已载入旧版脊线方法“{self._legacy_method}”。"
                        "点击“识别弧线”将显式使用 butterfly_curvature；原项目方法保持不变。"
                    )
            self.legacy_banner.setVisible(bool(self._legacy_method))

        def _point_list_label(
            self,
            point: Mapping[str, Any],
            index: int,
            *,
            marker: str | None = None,
        ) -> str:
            accepted = _point_is_accepted(point)
            marker_text = marker or ("✓" if accepted else "×")
            if self._trace_method() == _TRACE_METHOD_ANNULAR_PEAK:
                q_center = _read(point, ("q_center", "q"), None)
                q_min = _read(point, ("q_min",), None)
                q_max = _read(point, ("q_max",), None)
                selected = _read(point, ("selected_peaks", "peaks"), ())
                selected_count = (
                    len(selected)
                    if isinstance(selected, Sequence) and not isinstance(selected, (str, bytes))
                    else 0
                )
                reason = _read(point, ("reason", "status", "failure_reason"), None)
                if q_min not in (None, "") and q_max not in (None, ""):
                    q_label = f"q=[{_fmt(q_min)}, {_fmt(q_max)}]"
                elif q_center not in (None, ""):
                    q_label = f"q={_fmt(q_center)}"
                else:
                    q_label = f"ring {index + 1}"
                peak_label = (
                    f"{selected_count} peaks"
                    if self._language.lower().startswith("en")
                    else f"{selected_count} 个峰"
                )
                if not selected_count and reason not in (None, ""):
                    peak_label = str(reason)
                return f"{marker_text} {q_label} · {peak_label}"
            if self._trace_method() == _TRACE_METHOD_RADIAL_SECTOR:
                center = _read(
                    point,
                    ("sector_center_deg", "chi_deg", "angular_peak_deg"),
                    None,
                )
                q_star = _read(
                    point,
                    ("q_star", "selected_peak_q", "q"),
                    None,
                )
                if center in (None, ""):
                    center = f"sector {index + 1}"
                else:
                    center = f"χ={_fmt(center)}°"
                if q_star in (None, ""):
                    reason = _read(point, ("failure_reason", "reason", "status"), None)
                    q_label = str(reason or ("unlocated" if self._language.lower().startswith("en") else "未定位"))
                else:
                    q_label = f"q*={_fmt(q_star)}"
                return f"{marker_text} {center} · {q_label}"
            point_id = str(_read(point, ("point_id",), "") or "")
            qx = _fmt(_read(point, ("qx",), None))
            qy = _fmt(_read(point, ("qy",), None))
            return f"{marker_text} {point_id}  ({qx}, {qy})"

        def set_result(self, result: Any = None) -> None:
            butterfly = result if isinstance(result, Mapping) else {}
            self._result = deepcopy(dict(butterfly))
            self._result_fresh = bool(self._result)
            if self._result:
                self._adopt_result_trace_method(self._result)
            self._model_parameters = None
            self._model_reference_axis_deg = None
            self._model_status = None
            self._model_diagnostics = {}
            self._result_revision += 1
            self._manual_review = {
                "manual_status": "unreviewed",
                "reviewed_by": "",
                "reviewed_at": None,
                "review_notes": "",
                "result_revision": self._result_revision if self._result_fresh else None,
            }
            points = self._result_points()
            list_points = self._point_list_points()
            profiles = _read(self._result, ("profiles",), {})
            profiles = dict(profiles) if isinstance(profiles, Mapping) else {}
            sector_bundle = _read(self._result, ("sector_peaks",), {})
            nested_profiles = _read(sector_bundle, ("profiles",), {})
            if isinstance(nested_profiles, Mapping):
                profiles.update(dict(nested_profiles))
            annular_bundle = _read(self._result, ("annular_peaks",), {})
            nested_profiles = _read(annular_bundle, ("profiles",), {})
            if isinstance(nested_profiles, Mapping):
                profiles.update(dict(nested_profiles))
            display_result = dict(self._result)
            display_result["points"] = points
            display_result["profiles"] = profiles
            self.qspace.set_butterfly(display_result)
            self._peak_landmarks = dict(
                _read(self._result, ("peak_landmarks",), {}) or {}
            )
            self._selected_landmark = {}
            self._selected_landmark_id = None
            self._selected_profile_point_id = None
            self._reset_landmark_zoom()
            self.qspace.set_peak_landmarks(self._peak_landmarks)
            self._render_peak_table()
            self.peak_angular_profile.clear(message_key="landmark.no_profiles")
            self.peak_radial_profile.clear(message_key="landmark.no_profiles")
            self._refresh_fit_layers()
            self.point_list.blockSignals(True)
            self.point_list.clear()
            for index, point in enumerate(list_points):
                if not isinstance(point, Mapping):
                    continue
                accepted = _point_is_accepted(point)
                marker = "✓" if accepted else "×"
                item = QtWidgets.QListWidgetItem(
                    self._point_list_label(point, index, marker=marker)
                )
                source_reason = _read(
                    point,
                    ("failure_reason", "reason", "status"),
                    None,
                )
                confidence = _read(point, ("confidence",), None)
                tooltip_parts = [str(value) for value in (source_reason, confidence) if value not in (None, "")]
                if tooltip_parts:
                    item.setToolTip(" · ".join(tooltip_parts))
                item.setData(QtCore.Qt.ItemDataRole.UserRole, dict(point))
                self.point_list.addItem(item)
            self.point_list.blockSignals(False)
            self._excluded_count = sum(
                1
                for point in points
                if isinstance(point, Mapping)
                and not _point_is_accepted(point)
            )
            self._render_excluded_count()
            self._profiles = dict(profiles or {}) if isinstance(profiles, Mapping) else {}
            self._ellipse_local = dict(_read(self._result, ("ellipse_local",), {}) or {})
            self._render_quantities(_read(self._result, ("quantitative_parameters",), {}) or {})
            diagnostics = _read(self._result, ("diagnostics",), {}) or {}
            magnification = _read(diagnostics, ("display_magnification", "magnification"), 1.0)
            if _finite(magnification) is not None:
                self._diagnostic_magnification = float(magnification)
                if self._ellipse_local:
                    self.uv_magnification_spin.blockSignals(True)
                    self.uv_magnification_spin.setValue(max(0.1, min(100.0, float(magnification))))
                    self.uv_magnification_spin.blockSignals(False)
                    self._uv_display_magnification = float(self.uv_magnification_spin.value())
            self._render_magnification_label()
            self._page_status_state = (
                "failed"
                if self._result and self._result_is_failed(self._result)
                else "result"
                if self._result
                else "ready"
                if self._data_ready()
                else "empty"
            )
            self._page_status_kind = str(self._settings.get("stage", "trace"))
            self._page_status_error = self._result.get("error")
            self._render_page_status()
            self._sync_export_state()

        set_butterfly_result = set_result

        def clear_result(self, *, message: str | None = None, state: str | None = None) -> None:
            self._result = {}
            self._result_fresh = False
            self._result_revision += 1
            self._manual_review = {
                "manual_status": "unreviewed",
                "reviewed_by": "",
                "reviewed_at": None,
                "review_notes": "",
                "result_revision": None,
            }
            self._profiles = {}
            self._ellipse_local = {}
            self.qspace.set_butterfly({})
            self.qspace.set_selected_point(None)
            self._clear_diagnostic_layers()
            self.point_list.blockSignals(True)
            self.point_list.clear()
            self.point_list.blockSignals(False)
            self._selected_ellipse_point = {}
            self._excluded_count = 0
            self._render_excluded_count()
            self.quantity_table.setRowCount(0)
            self.normal_profile.clear()
            self.ellipse_diagnostic.clear()
            self._selected_profile_point_id = None
            del message  # The structured state is rendered afresh on language changes.
            self._page_status_state = str(state or ("ready" if self._data_ready() else "empty"))
            self._page_status_kind = ""
            self._page_status_error = None
            self._render_page_status()
            self._sync_export_state()

        def _set_quantity_cell(self, row: int, column: int, text: Any) -> None:
            """Keep the complete value available when a narrow cell elides it."""

            value = str(text)
            item = QtWidgets.QTableWidgetItem(value)
            item.setToolTip(value)
            self.quantity_table.setItem(row, column, item)

        def _render_quantities(self, quantities: Mapping[str, Any]) -> None:
            self.quantity_table.setRowCount(0)
            for name, payload in quantities.items():
                if not isinstance(payload, Mapping):
                    payload = {"value": payload}
                row = self.quantity_table.rowCount()
                self.quantity_table.insertRow(row)
                value = _read(payload, ("value",), None)
                candidate_value = _read(payload, ("candidate_value", "candidate"), None)
                status = _read(payload, ("status",), "unknown")
                if (
                    value not in (None, "")
                    and candidate_value not in (None, "")
                    and _finite(value) is not None
                    and _finite(value) == _finite(candidate_value)
                ):
                    candidate_value = None
                interval = _read(payload, ("interval", "ci", "confidence_interval"), None)
                if isinstance(interval, Sequence) and not isinstance(interval, (str, bytes)):
                    interval_text = "[" + ", ".join(_fmt(item) for item in interval) + "]"
                else:
                    interval_text = _fmt(interval)
                reason = _read(payload, ("reason",), "")
                display_name = {
                    "axis_ratio": "b/a",
                    "theta_deg": "θ",
                    "theta": "θ",
                }.get(str(name), str(name))
                status_text = str(status)
                reason_text = str(reason or "")
                english = self._language.lower().startswith("en")
                status_key = status_text.strip().lower()
                if english:
                    status_text = {
                        "ok": "available",
                        "pass": "available",
                        "warn": "warning",
                        "warning": "warning",
                        "estimate": "estimate",
                        "candidate": "candidate",
                        "available": "available",
                        "unavailable": "unavailable",
                        "undetermined": "undetermined",
                        "not_evaluated": "not evaluated",
                        "not assessed": "not assessed",
                    }.get(status_key, status_text)
                    if status_text.lower() in {
                        "not_evaluated",
                        "not evaluated",
                        "pending",
                        "pending_evaluation",
                    }:
                        status_text = "not evaluated"
                        if not reason_text:
                            reason_text = self._tr(
                                "workflow.trace_evaluate_reason",
                                resamples=self._evaluation_resamples(),
                            )
                else:
                    status_text = {
                        "ok": "通过",
                        "pass": "通过",
                        "warn": "警告",
                        "warning": "警告",
                        "estimate": "估计值",
                        "candidate": "候选值",
                        "available": "可用",
                        "unavailable": "不可用",
                        "not_evaluated": "待评估",
                        "not evaluated": "待评估",
                        "pending": "待评估",
                        "pending_evaluation": "待评估",
                        "undetermined": "未确定",
                        "fail": "失败",
                        "failed": "失败",
                    }.get(status_text.lower(), status_text)
                    if status_text == "待评估" and not reason_text:
                        reason_text = self._tr(
                            "workflow.trace_evaluate_reason",
                            resamples=self._evaluation_resamples(),
                        )
                confidence = str(_read(payload, ("confidence",), "") or "").strip()
                publication_status = str(
                    _read(payload, ("publication_status",), "") or ""
                ).strip()
                if confidence:
                    confidence_label = confidence
                    if not english:
                        confidence_label = {
                            "empirical": "经验支持",
                            "limited": "有限支持",
                            "unavailable": "不可用",
                        }.get(confidence.lower(), confidence)
                    status_text = f"{status_text} · {confidence_label}"
                if publication_status:
                    publication_label = publication_status
                    if not english:
                        publication_label = {
                            "available": "可报告",
                            "not_assessed": "未评估发表状态",
                        }.get(publication_status.lower(), publication_status)
                    elif publication_status.lower() == "not_assessed":
                        publication_label = "publication not assessed"
                    reason_text = " · ".join(
                        part for part in (publication_label, reason_text) if part
                    )
                state_tip = " · ".join(
                    part
                    for part in (
                        str(status),
                        f"confidence: {confidence}" if confidence else "",
                        f"publication_status: {publication_status}" if publication_status else "",
                    )
                    if part
                )
                for column, text in enumerate(
                    (display_name, _fmt(value), status_text, _fmt(candidate_value), interval_text, reason_text)
                ):
                    self._set_quantity_cell(row, column, text)
                state_cell = self.quantity_table.item(row, 2)
                if state_cell is not None and state_tip:
                    state_cell.setToolTip(state_tip)
            self._render_review_observables()

        def _render_review_observables(self) -> None:
            """Show observed period and coverage even when Ln/b/a stay unpublished."""

            result = self._result if isinstance(self._result, Mapping) else {}
            candidate = result.get("candidate_fit")
            if not isinstance(candidate, Mapping):
                candidate = {}
            quality = result.get("quality")
            if not isinstance(quality, Mapping):
                quality = {}
            metrics = quality.get("metrics")
            side_counts = metrics.get("side_counts") if isinstance(metrics, Mapping) else None
            sides = None
            if isinstance(side_counts, Mapping) and side_counts:
                present = sum(
                    1
                    for value in side_counts.values()
                    if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0
                )
                sides = f"{present}/4"
            english = self._language.lower().startswith("en")
            bound_flags = candidate.get("bound_flags", candidate.get("bound_status"))
            flag_names = _flag_values(candidate.get("flags"))
            flag_names.extend(_flag_values(quality.get("flags")))
            if isinstance(bound_flags, Mapping):
                if bound_flags.get("axis_ratio"):
                    flag_names.append("axis_ratio_at_bound")
                if any(value is True for value in bound_flags.values()):
                    flag_names.append("parameter_at_explicit_bound")
            ellipse_limited = ellipse_candidate_is_limited(
                {"flags": flag_names, "bound_flags": bound_flags}
            )
            kind = classify_ellipse_publication(
                quality_status=quality.get("status"),
                axis_ratio=candidate.get("axis_ratio"),
                flags=flag_names,
            )
            unpublished_shape = unpublished_ellipse_shape(
                quality_status=quality.get("status"),
                axis_ratio=candidate.get("axis_ratio"),
                flags=flag_names,
            )
            if kind == "ring":
                reading = "ring only" if english else "仅一阶环"
            elif kind == "fail":
                reading = "fail" if english else "失败"
            elif ellipse_limited:
                reading = "constrained ellipse candidate" if english else "受约束椭圆候选"
            else:
                reading = "ellipse" if english else "椭圆"
            sector_result = self._is_sector_result()
            annular_result = self._is_annular_result()
            q_star_label = (
                "q* sector median (unassigned order)"
                if english
                else "主峰 q*中位数（未定级）"
            ) if sector_result else ("q* (first-order)" if english else "一阶 q*")
            ring_length_label = (
                "2π/q* (apparent)"
                if english
                else "2π/q*（表观）"
            ) if sector_result else ("L ring (nm)" if english else "环 L（nm）")
            sector_summary = result.get("measurement_summary")
            sector_summary = sector_summary if isinstance(sector_summary, Mapping) else {}
            quality_display = quality.get("status")
            if ellipse_limited and kind == "ellipse" and not annular_result:
                qualifier = "constrained ellipse candidate" if english else "受约束椭圆候选"
                quality_display = (
                    f"{quality_display} · {qualifier}"
                    if quality_display not in (None, "")
                    else qualifier
                )
            if sector_result:
                # The selected-sector statistic is the authoritative radial
                # readout for this method. Never populate it from the legacy
                # arc-radius aliases, which describe a different observable.
                sector_q_star = sector_summary.get(
                    "q_star_sector_median", result.get("q_star_sector_median")
                )
                sector_q_source = sector_summary.get(
                    "aggregation", "median of selected sector-profile peaks"
                )
                sector_period = sector_summary.get(
                    "apparent_period_from_sector_median_nm",
                    result.get("apparent_period_from_sector_median_nm"),
                )
                sector_q_unit = sector_summary.get(
                    "q_star_sector_median_unit", result.get("q_unit", "unknown")
                )
                if canonical_q_unit(sector_q_unit) not in {"nm⁻¹", "Å⁻¹"}:
                    sector_period = None
            else:
                sector_q_star = None
                sector_q_source = None
                sector_period = None
            annular_points = [
                point
                for point in (result.get("points") or ())
                if isinstance(point, Mapping)
                and bool(point.get("accepted", point.get("valid", False)))
                and bool(point.get("valid", True))
            ] if annular_result else []
            trajectory_ids = {
                str(point.get("trajectory_id"))
                for point in annular_points
                if point.get("trajectory_id") not in (None, "")
            }
            annular_bundle = result.get("annular_peaks")
            annuli = annular_bundle.get("annuli", ()) if isinstance(annular_bundle, Mapping) else ()
            annular_rows = (
                (
                    "reading" if english else "判读",
                    "annular I(χ) tracks" if english else "q 环 I(χ) 四瓣轨迹",
                    quality.get("status"),
                    None,
                    None,
                    "angular maxima are linked across q; no q* median or spacing is inferred"
                    if english
                    else "沿 q 连接每个环的方位峰；不由此推导 q* 中位数或周期",
                ),
                (
                    "quality" if english else "质量",
                    quality.get("status"),
                    quality.get("status"),
                    None,
                    None,
                    ", ".join(_flag_values(quality.get("flags"))),
                ),
                (
                    "track support" if english else "轨迹支持",
                    f"{len(trajectory_ids)} tracks" if trajectory_ids else "—",
                    "pending evaluation" if str(self._settings.get("stage", "trace")) == "trace" else None,
                    None,
                    None,
                    "multiple q rings; up to four peaks per ring"
                    if english
                    else "多个 q 环；每环最多四个方位峰",
                ),
                (
                    "annuli" if english else "q 环",
                    len(annuli) if isinstance(annuli, Sequence) and not isinstance(annuli, (str, bytes)) else "—",
                    None,
                    None,
                    None,
                    "profiles include raw counts and coverage"
                    if english
                    else "剖面保留原始像素数和覆盖率",
                ),
            )
            review_rows = annular_rows if annular_result else (
                (
                    "reading" if english else "判读",
                    reading,
                    quality_display,
                    None,
                    None,
                    None,
                ),
                (
                    "quality" if english else "质量",
                    quality_display,
                    quality_display,
                    None,
                    None,
                    ", ".join(_flag_values(quality.get("flags"))),
                ),
                (
                    "arcs" if english else "弧",
                    sides,
                    None,
                    None,
                    None,
                    None,
                ),
                (
                    q_star_label,
                    sector_q_star
                    if sector_result
                    else candidate.get("q_star_from_arcs", result.get("q_star_from_arcs")),
                    sector_q_source
                    if sector_result
                    else candidate.get("q_star_source", result.get("q_star_source")),
                    None,
                    None,
                    None,
                ),
                (
                    ring_length_label,
                    sector_period
                    if sector_result
                    else candidate.get(
                        "L_from_observed_radius_nm", result.get("L_from_observed_radius_nm")
                    ),
                    None,
                    None,
                    None,
                    None,
                ),
            )
            if not annular_result:
                extra = []
                quantitative = result.get("quantitative_parameters")
                quantitative = quantitative if isinstance(quantitative, Mapping) else {}
                shape_issue = (
                    unpublished_shape
                    or ellipse_limited
                    or kind in {"ring", "fail", "undetermined"}
                )
                default_reason = (
                    "Candidate derived from the apparent ellipse; review the fit flags before interpretation."
                    if english
                    else "基于表观椭圆的候选值；解释前请结合拟合标记判读。"
                )
                flag_reason = ", ".join(dict.fromkeys(flag_names))
                if flag_reason:
                    default_reason = f"{default_reason} {flag_reason}"
                if shape_issue:
                    default_reason = (
                        "Ellipse shape is not supported as a measured result; the fitted value is retained as a candidate."
                        if english
                        else "椭圆形状尚不支持作为测量结果；保留拟合值供判读。"
                    )
                    if flag_reason:
                        default_reason = f"{default_reason} {flag_reason}"

                def candidate_parameter(
                    names: tuple[str, ...],
                    quantity_names: tuple[str, ...],
                ) -> tuple[Any, str, str, str]:
                    value = next(
                        (
                            candidate.get(name)
                            for name in names
                            if candidate.get(name) not in (None, "")
                        ),
                        None,
                    )
                    state = "candidate"
                    confidence = "limited"
                    reason = default_reason
                    parameter = next(
                        (quantitative.get(name) for name in quantity_names if name in quantitative),
                        None,
                    )
                    if isinstance(parameter, Mapping):
                        parameter_value = next(
                            (
                                parameter.get(name)
                                for name in ("candidate_value", "candidate", "value")
                                if parameter.get(name) not in (None, "")
                            ),
                            None,
                        )
                        if parameter_value is not None:
                            value = parameter_value
                        if shape_issue:
                            state = "candidate"
                            confidence = "limited"
                            reason = default_reason
                        else:
                            state = str(_read(parameter, ("status",), state) or state)
                            confidence = str(_read(parameter, ("confidence",), confidence) or confidence)
                            reason = str(_read(parameter, ("reason",), reason) or reason)
                    return value, state, confidence, reason

                ln, ln_state, ln_confidence, ln_reason = candidate_parameter(
                    (
                        "Ln_candidate_from_minor_axis_nm",
                        "Ln_from_minor_axis_nm",
                        "L_N",
                    ),
                    ("Ln_from_minor_axis_nm", "Ln", "L_N"),
                )
                lz, lz_state, lz_confidence, lz_reason = candidate_parameter(
                    (
                        "Lz_candidate_from_draw_axis_nm",
                        "Lz_from_draw_axis_nm",
                        "L_z",
                    ),
                    ("Lz_from_draw_axis_nm", "Lz", "L_z"),
                )
                l_major, major_state, major_confidence, major_reason = candidate_parameter(
                    (
                        "L_candidate_from_major_axis_nm",
                        "L_from_major_axis_nm",
                    ),
                    ("L_from_major_axis_nm", "L_major"),
                )
                if ln not in (None, ""):
                    extra.append(
                        (
                            "Ln candidate (nm)" if english else "Ln 候选（nm）",
                            None,
                            f"{ln_state} · {ln_confidence}",
                            ln,
                            None,
                            ln_reason,
                        )
                    )
                if lz not in (None, ""):
                    extra.append(
                        (
                            "Lz candidate (nm)" if english else "Lz 候选（nm）",
                            None,
                            f"{lz_state} · {lz_confidence}",
                            lz,
                            None,
                            lz_reason,
                        )
                    )
                if l_major not in (None, ""):
                    extra.append(
                        (
                            "L major candidate (nm)" if english else "长轴 L 候选（nm）",
                            None,
                            f"{major_state} · {major_confidence}",
                            l_major,
                            None,
                            major_reason,
                        )
                    )
                if extra:
                    review_rows = review_rows + tuple(extra)
            if not any(row[1] not in (None, "", []) for row in review_rows):
                return
            for name, value, status, candidate_value, interval, reason in review_rows:
                if value in (None, "") and not status and not reason:
                    continue
                row = self.quantity_table.rowCount()
                self.quantity_table.insertRow(row)
                is_q_star_source = str(name) == str(q_star_label) and status not in (None, "")
                display_status = (
                    translate_q_star_source(self._language, status)
                    if is_q_star_source
                    else status
                )
                for column, text in enumerate(
                    (
                        str(name),
                        _fmt(value),
                        "" if display_status is None else str(display_status),
                        _fmt(candidate_value),
                        "" if interval is None else str(interval),
                        "" if reason is None else str(reason),
                    )
                ):
                    self._set_quantity_cell(row, column, text)
                if is_q_star_source:
                    source_item = self.quantity_table.item(row, 2)
                    source_item.setData(QtCore.Qt.ItemDataRole.UserRole, status)
                    source_item.setData(QtCore.Qt.ItemDataRole.UserRole + 1, "q_star_source")

        def _on_point_selected(self, point: Any) -> None:
            if not isinstance(point, Mapping):
                return
            point_id = self._profile_id_for_entry(point)
            self.point_list.blockSignals(True)
            for row in range(self.point_list.count()):
                item = self.point_list.item(row)
                data = item.data(QtCore.Qt.ItemDataRole.UserRole)
                if isinstance(data, Mapping) and self._profile_id_for_entry(data) == point_id:
                    self.point_list.setCurrentRow(row)
                    break
            self.point_list.blockSignals(False)
            self._render_profile(point_id, point)
            self.pointSelected.emit(dict(point))

        def _on_point_list_changed(self, row: int) -> None:
            if row < 0:
                return
            item = self.point_list.item(row)
            point = item.data(QtCore.Qt.ItemDataRole.UserRole) if item is not None else None
            if not isinstance(point, Mapping):
                return
            if self._trace_method() == _TRACE_METHOD_ANNULAR_PEAK:
                # An annulus row is an angular profile, not one editable
                # q-space point.  Its selected peaks are shown in the profile.
                self.qspace.set_selected_point(None)
            elif bool(_read(point, ("profile_only",), False)):
                self.qspace.set_selected_point(None)
            else:
                self.qspace.set_selected_point(_read(point, ("point_id",), None))
            self._on_point_selected(point)

        @staticmethod
        def _profile_id_for_entry(entry: Mapping[str, Any] | None) -> str:
            if not isinstance(entry, Mapping):
                return ""
            value = _read(entry, ("profile_id", "point_id", "annulus_id"), "")
            return str(value or "")

        def _exclude_selected_point(self) -> None:
            _exclude_selected_point_impl(self)

        @staticmethod
        def _series(
            profile: Mapping[str, Any],
            x_names: tuple[str, ...],
            y_names: tuple[str, ...],
        ) -> tuple[list[Any], list[Any]]:
            return _extract_profile_series(profile, x_names, y_names)

        def _sector_failure_text(
            self,
            point: Mapping[str, Any] | None,
            profile: Mapping[str, Any] | None = None,
        ) -> str:
            return _sector_failure_text_impl(self._language, point, profile)

        @staticmethod
        def _sector_failure_code(
            point: Mapping[str, Any] | None,
            profile: Mapping[str, Any] | None = None,
        ) -> str:
            return _sector_failure_code_impl(point, profile)

        def _radial_profile_title(
            self,
            point_id: str,
            point: Mapping[str, Any] | None,
            profile: Mapping[str, Any] | None,
        ) -> str:
            return _radial_profile_title_impl(self, point_id, point, profile)

        def _radial_profile_tooltip(
            self,
            point_id: str,
            point: Mapping[str, Any] | None,
            profile: Mapping[str, Any] | None,
            title: str,
        ) -> str:
            return _radial_profile_tooltip_impl(point_id, point, profile, title)

        def _annular_profile_title(
            self,
            profile_id: str,
            annulus: Mapping[str, Any] | None,
            profile: Mapping[str, Any] | None,
        ) -> str:
            return _annular_profile_title_impl(self, profile_id, annulus, profile)

        def _annular_profile_tooltip(
            self,
            profile_id: str,
            annulus: Mapping[str, Any] | None,
            profile: Mapping[str, Any] | None,
            title: str,
        ) -> str:
            return _annular_profile_tooltip_impl(profile_id, annulus, profile, title)

        def _render_annular_profile(
            self,
            profile_id: str,
            annulus: Mapping[str, Any] | None,
            profile: Mapping[str, Any] | None,
        ) -> None:
            _render_annular_profile_impl(self, profile_id, annulus, profile)

        def _render_radial_profile(
            self,
            point_id: str,
            point: Mapping[str, Any] | None,
            profile: Mapping[str, Any] | None,
        ) -> None:
            _render_radial_profile_impl(self, point_id, point, profile)

        def _render_profile(self, point_id: str, point: Mapping[str, Any] | None = None) -> None:
            _render_profile_impl(self, point_id, point)

        def _render_ellipse_local(self) -> None:
            _render_ellipse_local_impl(self)

        def _on_edit_requested(self, edit: Any) -> None:
            _apply_edit_request(self, edit)

        def start_correction(self) -> None:
            _start_correction_impl(self)

        def _open_numeric_edit_dialog(self) -> None:
            _open_numeric_edit_dialog_impl(self)

        def snapshot_state(self) -> dict[str, Any]:
            return _snapshot_workbench_state(self)

        def restore_state(self, snapshot: Mapping[str, Any]) -> None:
            _restore_workbench_state(self, snapshot)

        def undo_edit(self) -> None:
            _undo_edit_impl(self)

        def redo_edit(self) -> None:
            _redo_edit_impl(self)

        def _update_edit_buttons(self) -> None:
            _update_edit_buttons_impl(self)

        def _request_payload(self, *, stage: str, resamples: int) -> dict[str, Any]:
            self._settings["stage"] = stage
            if stage == "evaluate":
                selected_resamples = self._evaluation_resamples()
                self._settings["evaluation_resamples"] = selected_resamples
                self._settings["resamples"] = selected_resamples
            else:
                # Identify is a zero-resample trace request. Keep the user's
                # selected evaluation budget separately for the next stage.
                self._settings["resamples"] = int(resamples)
            self._settings["edits"] = deepcopy(self._edits)
            self._page_status_state = "ready"
            self._page_status_kind = stage
            self._page_status_error = None
            self._render_page_status()
            return self.butterfly_settings

        def request_identify(self) -> None:
            if self._q_range_error:
                self._render_q_range_feedback()
                self._render_page_status()
                return
            if not self._data_ready():
                self.set_job_status("empty")
                return
            self.identifyRequested.emit(self._request_payload(stage="trace", resamples=0))

        def request_evaluate(self) -> None:
            if self._q_range_error:
                self._render_q_range_feedback()
                self._render_page_status()
                return
            if not self._data_ready():
                self.set_job_status("empty")
                return
            self.evaluateRequested.emit(
                self._request_payload(
                    stage="evaluate",
                    resamples=self._evaluation_resamples(),
                )
            )

        def apply_to_batch(self) -> None:
            if self._q_range_error:
                self._render_q_range_feedback()
                self._render_page_status()
                return
            self.applyToBatchRequested.emit({"analysis": self.analysis_settings, "edits": self.edits})

        def set_batch_feedback(
            self,
            successes: Sequence[Any] = (),
            failures: Sequence[Any] = (),
            *,
            limited: Sequence[Any] = (),
        ) -> None:
            self._batch_success_count = len(list(successes))
            self._batch_limited_count = len(list(limited))
            self._batch_failure_items = list(failures)
            self._render_batch_feedback()

        def set_busy(self, busy: bool) -> None:
            self._busy = bool(busy)
            self._sync_action_state()
            if busy:
                self.set_job_status("running")

        def set_job_status(
            self,
            state: str,
            kind: str = "",
            *,
            error: Any = None,
            result_ok: bool | None = None,
            elapsed_s: float | None = None,
            progress_percent: int | None = None,
            progress_phase: str | None = None,
        ) -> None:
            """Route asynchronous worker outcomes into the page-local status."""

            state = str(state or "ready").lower()
            lifecycle_state = state
            label = str(kind or "analysis")
            retain_error = state in {"error", "failed"} or result_ok is False
            if state == "canceled":
                state = "cancelled"
            if result_ok is False:
                state = "failed"
            # A completed worker may have submitted a fresh diagnostic
            # payload whose engineering quality is FAIL.  Preserve that
            # payload; only an exception/error lifecycle invalidates the old
            # measurement.
            diagnostic_failure = bool(
                result_ok is False
                and lifecycle_state in {"completed", "complete", "result"}
            )
            detached_job = label in _DETACHED_JOB_KINDS or (
                state in {"cancelled", "canceled"}
                and self._detached_status_restore is not None
            )
            analysis_job = (
                label in _ANALYSIS_JOB_KINDS
                or (
                    state in {"cancelled", "ignored", "stale"}
                    and label in {"cancelled", "canceled", "ignored", "stale"}
                )
            ) and not detached_job
            if detached_job and state in {"running", "cancelling"} and self._result_fresh:
                # MainWindow first sends a generic ``running`` update from
                # set_busy(), then the detached export kind.  Reconstruct the
                # measurement state here so an export error can restore it.
                self._detached_status_restore = (
                    "result",
                    str(self._settings.get("stage", "trace")),
                    self._page_status_error,
                )
            elif detached_job and state in {
                "cancelled",
                "failed",
                "error",
                "completed",
                "complete",
                "ready",
            } and self._result_fresh and self._detached_status_restore is None:
                # A detached exporter may report an error without a preceding
                # page-local running callback (for example in a direct test or
                # a fast worker failure). Preserve the measurement in that
                # case as well.
                self._detached_status_restore = (
                    "result",
                    str(self._settings.get("stage", "trace")),
                    self._page_status_error,
                )
            restoring_detached_page = bool(
                self._detached_status_restore
                and not detached_job
                and label == self._detached_status_restore[1]
                and state
                in {"cancelled", "failed", "error", "completed", "complete", "result", "ready"}
            )
            terminal_analysis = analysis_job and state in {
                "cancelled",
                "ignored",
                "stale",
                "failed",
                "error",
            } and not restoring_detached_page and not diagnostic_failure
            if terminal_analysis:
                # A failed/cancelled analysis invalidates the displayed
                # measurement itself.  Do not leave q*/L or export controls
                # backed by a result that no longer belongs to the current job.
                self.clear_result(state=state)
                self._sync_action_state()
            if state in {"running", "cancelling"}:
                if elapsed_s is not None:
                    self._job_elapsed_s = max(0.0, float(elapsed_s))
                if progress_percent is not None:
                    self._job_progress_percent = max(0, min(100, int(progress_percent)))
                if progress_phase is not None:
                    self._job_progress_phase = str(progress_phase)
            else:
                self._job_elapsed_s = None
                self._job_progress_percent = None
                self._job_progress_phase = ""
            restored_detached = (
                detached_job
                and state in {"cancelled", "failed", "error", "completed", "complete", "ready"}
                and self._detached_status_restore
            )
            if restored_detached and self._result_fresh:
                self._page_status_state, self._page_status_kind, restored_error = (
                    self._detached_status_restore
                )
                self._page_status_error = restored_error
                self._detached_status_restore = None
            else:
                self._page_status_state = {
                    "error": "failed",
                    "complete": "completed",
                }.get(state, state)
                self._page_status_kind = label
                self._page_status_error = error if retain_error else None
                if detached_job and state not in {"running", "cancelling"}:
                    self._detached_status_restore = None
                if restoring_detached_page:
                    self._detached_status_restore = None
            self._render_page_status()
            try:
                QtGui.QAccessible.updateAccessibility(
                    QtGui.QAccessibleEvent(self.status_label, QtGui.QAccessible.Event.NameChanged)
                )
            except (AttributeError, TypeError):
                pass

        def finish_detached_job_if_active(self, kind: str) -> None:
            """Clear a detached worker label without replacing a newer page state."""

            if (
                self._page_status_kind == str(kind)
                and self._page_status_state in {"running", "cancelling"}
            ):
                self.set_job_status("ready")

        def save_screenshot(self, path: str | Path) -> Path:
            target = Path(path).expanduser().resolve()
            target.parent.mkdir(parents=True, exist_ok=True)
            if not bool(self.grab().save(str(target))):
                raise OSError(f"could not save screenshot: {target}")
            return target


else:

    class ButterflyWorkbench:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs
            require_qt()


__all__ = ["ButterflyWorkbench", "DEFAULT_BUTTERFLY_SETTINGS"]
