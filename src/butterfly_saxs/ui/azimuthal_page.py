"""Interactive q-annulus azimuthal profile and peak fitting page."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..azimuthal_analysis import (
    AzimuthalFitResult,
    AzimuthalProfile,
    export_azimuthal_csv_bundle,
    fit_azimuthal_peaks,
    measure_azimuthal_profile,
)
from .qt_compat import QT_AVAILABLE, QtCore, QtWidgets, require_qt
from .workers import AnalysisWorker, GenerationGuard


def _analyze_frame(parameters: dict[str, Any], payload: dict[str, Any]) -> tuple[Any, Any]:
    """Measure and fit a plain frame snapshot without accessing Qt widgets."""
    profile = measure_azimuthal_profile(**payload, **parameters["profile"])
    return profile, fit_azimuthal_peaks(profile, **parameters["fit"])


def _q_unit_key(value: Any) -> str:
    text = str(value or "unknown").strip().lower()
    return (text.replace(" ", "").replace("−", "-").replace("⁻", "-")
            .replace("¹", "1").replace("^", ""))


def _same_q_unit(first: Any, second: Any) -> bool:
    return _q_unit_key(first) == _q_unit_key(second)


def _display_q_unit(value: Any) -> str:
    text = str(value or "unknown").replace("−", "-")
    for superscript, replacement in (
        ("⁻¹", "^-1"), ("⁻²", "^-2"), ("⁻³", "^-3"),
        ("¹", "^1"), ("²", "^2"), ("³", "^3"),
    ):
        text = text.replace(superscript, replacement)
    return text


if QT_AVAILABLE:
    from matplotlib.colors import ListedColormap
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.lines import Line2D
    from matplotlib.figure import Figure
    from ..font_support import font_properties

    class AzimuthalPage(QtWidgets.QWidget):
        """Analyze an angular profile measured from an explicitly chosen q band."""

        profileChanged = QtCore.Signal(object, object)

        def __init__(self, parent: Any = None, *, language: str = "zh_CN") -> None:
            super().__init__(parent)
            self.setObjectName("azimuthalPage")
            self.language = str(language)
            self.data: np.ndarray | None = None
            self.qx: np.ndarray | None = None
            self.qy: np.ndarray | None = None
            self.q_map: np.ndarray | None = None
            self.valid_mask: np.ndarray | None = None
            self.q_unit = "unknown"
            self.source: str | None = None
            self.profile: AzimuthalProfile | None = None
            self.fit_result: AzimuthalFitResult | None = None
            self._generation = GenerationGuard()
            self._worker: AnalysisWorker | None = None
            self._closed = False
            self._thread_pool = QtCore.QThreadPool(self)
            self._thread_pool.setMaxThreadCount(1)
            self._build_ui()
            self.set_language(self.language)
            self._show_empty()

        def _build_ui(self) -> None:
            root = QtWidgets.QVBoxLayout(self)
            root.setContentsMargins(12, 10, 12, 10)
            root.setSpacing(8)

            self.heading = QtWidgets.QLabel()
            self.heading.setObjectName("azimuthalHeading")
            heading_font = self.heading.font()
            heading_font.setPointSize(15)
            heading_font.setBold(True)
            self.heading.setFont(heading_font)
            root.addWidget(self.heading)

            self.description = QtWidgets.QLabel()
            self.description.setWordWrap(True)
            self.description.setObjectName("azimuthalDescription")
            root.addWidget(self.description)

            self.settings_group = QtWidgets.QGroupBox()
            self.settings_group.setObjectName("azimuthalSettings")
            controls = QtWidgets.QGridLayout(self.settings_group)
            controls.setHorizontalSpacing(8)
            controls.setVerticalSpacing(6)
            self.q_min_spin = self._double_spin()
            self.q_max_spin = self._double_spin()
            self.q_min_spin.setObjectName("azimuthalQMin")
            self.q_max_spin.setObjectName("azimuthalQMax")
            self.bins_spin = QtWidgets.QSpinBox()
            self.bins_spin.setRange(36, 1440)
            self.bins_spin.setSingleStep(36)
            self.bins_spin.setValue(360)
            self.bins_spin.setObjectName("azimuthalBins")
            self.model_combo = QtWidgets.QComboBox()
            for name, label in (
                ("pseudo_voigt", "pseudo-Voigt"), ("gaussian", "Gaussian"),
                ("lorentzian", "Lorentzian"), ("von_mises", "von Mises"),
            ):
                self.model_combo.addItem(label, name)
            self.model_combo.setObjectName("azimuthalModel")
            self.max_peaks_spin = QtWidgets.QSpinBox()
            self.max_peaks_spin.setRange(1, 12)
            self.max_peaks_spin.setValue(4)
            self.max_peaks_spin.setObjectName("azimuthalMaxPeaks")
            self.min_separation_spin = self._double_spin(1.0, 180.0, 20.0, 1)
            self.min_separation_spin.setObjectName("azimuthalMinSeparation")
            self.height_fraction_spin = self._double_spin(0.0, 1.0, 0.08, 2)
            self.height_fraction_spin.setSingleStep(0.02)
            self.height_fraction_spin.setObjectName("azimuthalMinHeight")
            self.initial_width_spin = self._double_spin(2.0, 180.0, 35.0, 1)
            self.initial_width_spin.setObjectName("azimuthalInitialFwhm")
            self.eta_spin = self._double_spin(0.0, 1.0, 0.5, 2)
            self.eta_spin.setSingleStep(0.05)
            self.eta_spin.setObjectName("azimuthalEta")
            self._labels: dict[str, Any] = {}
            control_items = [
                ("q_min", self.q_min_spin), ("q_max", self.q_max_spin),
                ("bins", self.bins_spin), ("model", self.model_combo),
                ("max_peaks", self.max_peaks_spin), ("separation", self.min_separation_spin),
                ("height_fraction", self.height_fraction_spin), ("initial_width", self.initial_width_spin),
                ("eta", self.eta_spin),
            ]
            for index, (key, widget) in enumerate(control_items):
                row, col = divmod(index, 3)
                label = QtWidgets.QLabel()
                label.setWordWrap(True)
                label.setBuddy(widget)
                self._labels[key] = label
                controls.addWidget(label, row * 2, col)
                controls.addWidget(widget, row * 2 + 1, col)
                controls.setColumnStretch(col, 1)
                widget.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
            self.settings_scroll = QtWidgets.QScrollArea()
            self.settings_scroll.setObjectName("azimuthalSettingsScroll")
            self.settings_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
            self.settings_scroll.setWidgetResizable(True)
            self.settings_scroll.setMinimumHeight(85)
            self.settings_scroll.setMaximumHeight(240)
            self.settings_scroll.setWidget(self.settings_group)
            root.addWidget(self.settings_scroll)

            actions = QtWidgets.QHBoxLayout()
            self.analyze_button = QtWidgets.QPushButton()
            self.analyze_button.setObjectName("azimuthalAnalyze")
            self.analyze_button.setProperty("role", "primary")
            self.analyze_button.setEnabled(False)
            self.export_button = QtWidgets.QPushButton()
            self.export_button.setObjectName("azimuthalExport")
            self.export_button.setEnabled(False)
            self.status_label = QtWidgets.QLabel()
            self.status_label.setObjectName("azimuthalStatus")
            self.status_label.setWordWrap(True)
            self.status_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop)
            self.status_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
            self.status_label.setMargin(4)
            actions.addWidget(self.analyze_button)
            actions.addWidget(self.export_button)
            actions.addStretch(1)
            root.addLayout(actions)
            self.progress = QtWidgets.QProgressBar()
            self.progress.setObjectName("azimuthalProgress")
            self.progress.setRange(0, 0)
            self.progress.setTextVisible(False)
            self.progress.setFixedHeight(4)
            self.progress.hide()
            root.addWidget(self.progress)
            self.feedback_area = QtWidgets.QScrollArea()
            self.feedback_area.setObjectName("azimuthalFeedback")
            self.feedback_area.setWidgetResizable(True)
            self.feedback_area.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
            self.feedback_area.setFixedHeight(42)
            self.feedback_area.setWidget(self.status_label)
            root.addWidget(self.feedback_area)

            self.figure = Figure(figsize=(12.0, 6.5), constrained_layout=True)
            layout = self.figure.add_gridspec(
                2, 2, width_ratios=(1.0, 1.45), height_ratios=(3.0, 1.0)
            )
            self.image_axes = self.figure.add_subplot(layout[:, 0])
            self.profile_axes = self.figure.add_subplot(layout[0, 1])
            self.residual_axes = self.figure.add_subplot(layout[1, 1], sharex=self.profile_axes)
            self.canvas = FigureCanvasQTAgg(self.figure)
            self.canvas.setObjectName("azimuthalCanvas")
            self.canvas.setMinimumHeight(240)
            root.addWidget(self.canvas, 1)

            self.analyze_button.clicked.connect(self.start_analysis)
            self.export_button.clicked.connect(self._choose_export_path)
            self.model_combo.currentIndexChanged.connect(self._sync_fit_controls)
            for control, signal_name in (
                (self.q_min_spin, "valueChanged"), (self.q_max_spin, "valueChanged"),
                (self.bins_spin, "valueChanged"), (self.model_combo, "currentIndexChanged"),
                (self.max_peaks_spin, "valueChanged"),
                (self.min_separation_spin, "valueChanged"),
                (self.height_fraction_spin, "valueChanged"),
                (self.initial_width_spin, "valueChanged"), (self.eta_spin, "valueChanged"),
            ):
                getattr(control, signal_name).connect(self._invalidate_measurement)
            self._sync_fit_controls()

        @staticmethod
        def _double_spin(minimum: float = 0.0, maximum: float = 1.0e12,
                         value: float = 0.0, decimals: int = 4) -> Any:
            spin = QtWidgets.QDoubleSpinBox()
            spin.setRange(float(minimum), float(maximum))
            spin.setDecimals(int(decimals))
            spin.setValue(float(value))
            spin.setKeyboardTracking(False)
            return spin

        def _tr(self, key: str) -> str:
            english = str(self.language).lower().startswith("en")
            strings = {
                "heading": ("方位角峰拟合", "Azimuthal peak fitting"),
                "description": (
                    "从二维散射图像中选择 q 环带，检查角向强度分布并拟合周期峰。拟合只使用实际测得的有效角度点。",
                    "Choose a q annulus in the 2-D scattering image, inspect its angular intensity profile, and fit periodic peaks. Fits use measured valid angles only.",
                ),
                "q_min": ("q 下限", "q minimum"), "q_max": ("q 上限", "q maximum"),
                "bins": ("角度分箱", "Angular bins"), "model": ("峰模型", "Peak model"),
                "max_peaks": ("最多峰数", "Maximum peaks"),
                "separation": ("最小峰间距 (°)", "Minimum separation (°)"),
                "height_fraction": ("最小相对高度", "Minimum relative height"),
                "initial_width": ("初始 FWHM (°)", "Initial FWHM (°)"),
                "eta": ("pseudo-Voigt η", "pseudo-Voigt η"),
                "analyze": ("分析环带", "Analyze annulus"),
                "settings": ("环带与峰拟合设置", "Annulus and peak fit settings"),
                "ready": ("二维图像和 q 坐标已就绪；设置环带后点击分析。", "Image and q coordinates ready; choose an annulus, then analyze."),
                "running": ("正在提取角向强度并拟合峰…", "Measuring angular intensity and fitting peaks…"),
                "changed_running": ("数据或设置已变化；等待当前计算结束后重新分析。", "Data or settings changed; analyze again when the current calculation finishes."),
                "export": ("导出 CSV", "Export CSV"),
                "no_data": ("请先加载二维图像并完成 q 标定。", "Load a 2-D image and q calibration to begin."),
                "profile_title": ("环带角向强度", "Azimuthal intensity in q annulus"),
                "preview_title": ("二维图像与 q 环带", "2-D image and q annulus"),
                "detector_x": ("探测器 x (px)", "Detector x (px)"),
                "detector_y": ("探测器 y (px)", "Detector y (px)"),
                "angle": ("方位角 (°)", "Azimuth (°)"),
                "intensity": ("强度 (输入单位)", "Intensity (input units)"),
                "residual": ("残差", "Residual"),
                "measured": ("实测均值", "Measured mean"),
                "fit": ("联合拟合", "Joint fit"),
                "baseline": ("背景", "Baseline"),
                "no_profile": ("尚未分析环带", "No annulus analyzed"),
                "analyzed": ("q = {q0:.6g}–{q1:.6g} {unit}；有效角度点 {n}；平均覆盖率 {coverage:.1%}。",
                             "q = {q0:.6g}–{q1:.6g} {unit}; {n} measured angular bins; mean coverage {coverage:.1%}."),
                "fit_status": ("拟合成功：{peaks} 个峰，RMSE {rmse:.4g}。", "Fit succeeded: {peaks} peaks, RMSE {rmse:.4g}."),
                "fit_failed": ("未得到有效峰拟合：{message}", "No valid peak fit: {message}"),
                "export_done": ("已导出：{profile} 和 {peaks}", "Exported {profile} and {peaks}"),
                "export_exists": ("文件已存在，是否覆盖？", "The files already exist. Overwrite both files?"),
                "export_error": ("导出失败：{error}", "Export failed: {error}"),
                "analysis_error": ("分析失败：{error}", "Analysis failed: {error}"),
                "file_filter": ("CSV 文件 (*.csv)", "CSV files (*.csv)"),
            }
            return strings[key][0 if not english else 1]

        def _apply_language(self) -> None:
            self.heading.setText(self._tr("heading"))
            self.description.setText(self._tr("description"))
            self.settings_group.setTitle(self._tr("settings"))
            self.settings_scroll.setAccessibleName(self._tr("settings"))
            self.progress.setAccessibleName(self._tr("running"))
            for key, label in self._labels.items():
                label.setText(self._tr(key))
                label.buddy().setAccessibleName(self._tr(key))
            self.analyze_button.setText(self._tr("analyze"))
            self.export_button.setText(self._tr("export"))
            if self.jobs_running():
                key = "changed_running" if self._worker and not self._generation.is_current(self._worker.generation) else "running"
                self._set_feedback(self._tr(key), "running")
            elif self.data is None:
                self._set_feedback(self._tr("no_data"), "empty")
            elif self.profile is None:
                self._set_feedback(self._tr("ready"), "ready")
            else:
                self._present_result(self.profile, self.fit_result)
                return
            self._draw()

        def set_language(self, language: str) -> None:
            self.language = str(language)
            self._apply_language()
            from .help import apply_help

            apply_help(self, self.language)

        def set_data(
            self,
            data: Any,
            *,
            qx: Any,
            qy: Any,
            valid_mask: Any,
            q_unit: str,
            source: Any,
        ) -> None:
            """Set one frame; ``valid_mask=True`` marks usable detector pixels."""

            self._generation.next()
            image = np.asarray(data, dtype=float)
            if image.ndim != 2 or image.size == 0:
                self.invalidate(self._tr("analysis_error").format(error="data must be a non-empty 2-D image"))
                raise ValueError("data must be a non-empty two-dimensional image")
            try:
                qx_array, qy_array = np.broadcast_arrays(np.asarray(qx, dtype=float), np.asarray(qy, dtype=float))
                qx_array = np.broadcast_to(qx_array, image.shape)
                qy_array = np.broadcast_to(qy_array, image.shape)
                if valid_mask is None:
                    valid_array = np.ones(image.shape, dtype=bool)
                else:
                    valid_array = np.broadcast_to(np.asarray(valid_mask, dtype=bool), image.shape)
            except (TypeError, ValueError) as exc:
                self.invalidate(self._tr("analysis_error").format(error="qx, qy, and valid_mask must match image shape"))
                raise ValueError("qx, qy, and valid_mask must broadcast to the image shape") from exc
            finite_q = np.hypot(qx_array, qy_array)
            # q-window limits describe calibration geometry. They must not
            # move merely because a detector mask changes between frames.
            valid_q = finite_q[np.isfinite(finite_q)]
            if not valid_q.size:
                self.invalidate(self._tr("analysis_error").format(error="no finite q coordinates"))
                raise ValueError("no finite q coordinates")

            previous_window = None
            previous_unit = self.q_unit
            if self.data is not None:
                previous_window = (self.q_min_spin.value(), self.q_max_spin.value())
            self.profile = None
            self.fit_result = None
            self.data = image
            self.qx = qx_array
            self.qy = qy_array
            self.q_map = finite_q
            self.valid_mask = valid_array
            self.q_unit = str(q_unit or "unknown")
            self.source = str(source) if source is not None else None
            lower, upper = float(np.min(valid_q)), float(np.max(valid_q))
            if not upper > lower:
                self.invalidate(self._tr("analysis_error").format(error="q map has no radial range"))
                raise ValueError("q map has no radial range")
            self.q_min_spin.setRange(lower, upper)
            self.q_max_spin.setRange(lower, upper)
            span = upper - lower
            self.q_min_spin.setSingleStep(max(span / 100.0, np.finfo(float).eps))
            self.q_max_spin.setSingleStep(max(span / 100.0, np.finfo(float).eps))
            self.q_min_spin.setDecimals(10)
            self.q_max_spin.setDecimals(10)
            unit_suffix = f" {_display_q_unit(self.q_unit)}"
            self.q_min_spin.setSuffix(unit_suffix)
            self.q_max_spin.setSuffix(unit_suffix)
            if (previous_window and _same_q_unit(previous_unit, self.q_unit)
                    and previous_window[1] > previous_window[0]):
                q_min = float(np.clip(previous_window[0], lower, upper))
                q_max = float(np.clip(previous_window[1], lower, upper))
                if q_max <= q_min:
                    q_min, q_max = lower, upper
            else:
                q_min, q_max = lower, upper
            self.q_min_spin.setValue(q_min)
            self.q_max_spin.setValue(q_max)
            self.profile = None
            self.fit_result = None
            self.analyze_button.setEnabled(not self.jobs_running())
            self.export_button.setEnabled(False)
            running = self.jobs_running()
            self._set_feedback(self._tr("changed_running") if running else self._tr("ready"), "running" if running else "ready")
            self._draw()

        def invalidate(self, message: str | None = None) -> None:
            """Clear current measurements after image, calibration, or mask changes."""

            self._generation.next()
            self.data = None
            self.qx = None
            self.qy = None
            self.q_map = None
            self.valid_mask = None
            self.q_unit = "unknown"
            self.source = None
            self.profile = None
            self.fit_result = None
            self.analyze_button.setEnabled(False)
            self.export_button.setEnabled(False)
            self._set_feedback(message or self._tr("no_data"), "empty")
            self._show_empty()

        def _sync_fit_controls(self, *_args: Any) -> None:
            is_pseudo = self.model_combo.currentData() == "pseudo_voigt"
            self.eta_spin.setEnabled(is_pseudo)

        def _invalidate_measurement(self, *_args: Any) -> None:
            """Drop a result as soon as one of its analysis settings changes."""

            self._generation.next()
            if self.jobs_running():
                self._set_feedback(self._tr("changed_running"), "running")
            if self.profile is None and self.fit_result is None:
                if self.sender() in (self.q_min_spin, self.q_max_spin) and self.data is not None:
                    self._draw()
                return
            self.profile = None
            self.fit_result = None
            self.export_button.setEnabled(False)
            if not self.jobs_running():
                self._set_feedback(self._tr("ready"), "ready")
            self._draw()

        def _analysis_inputs(self, *, snapshot: bool) -> tuple[dict[str, Any], dict[str, Any]]:
            payload = {
                "data": self.data, "qx": self.qx, "qy": self.qy,
                "valid_mask": self.valid_mask, "q_unit": self.q_unit, "source": self.source,
            }
            if snapshot:
                for key in ("data", "qx", "qy", "valid_mask"):
                    payload[key] = np.array(payload[key], copy=True)
            parameters = {
                "profile": {
                    "q_window": (self.q_min_spin.value(), self.q_max_spin.value()),
                    "n_bins": self.bins_spin.value(), "statistic": "mean",
                },
                "fit": {
                    "model": str(self.model_combo.currentData()),
                    "max_peaks": self.max_peaks_spin.value(),
                    "min_separation_deg": self.min_separation_spin.value(),
                    "min_height_fraction": self.height_fraction_spin.value(),
                    "initial_fwhm_deg": self.initial_width_spin.value(),
                    "eta": self.eta_spin.value(),
                },
            }
            return parameters, payload

        def _set_feedback(self, text: str, state: str) -> None:
            self.status_label.setText(text)
            self.status_label.setProperty("state", state)
            self.status_label.style().unpolish(self.status_label)
            self.status_label.style().polish(self.status_label)

        def _prepare_analysis(self) -> None:
            self.profile = None
            self.fit_result = None
            self.export_button.setEnabled(False)
            self.analyze_button.setEnabled(False)
            self._set_feedback(self._tr("running"), "running")
            self._draw()

        def analyze(self) -> AzimuthalFitResult | None:
            """Synchronous entry point for integrations; buttons use ``start_analysis``."""
            if self.data is None or self.jobs_running():
                return None
            self._prepare_analysis()
            try:
                profile, fit_result = _analyze_frame(*self._analysis_inputs(snapshot=False))
            except (TypeError, ValueError, RuntimeError) as exc:
                self._set_feedback(self._tr("analysis_error").format(error=exc), "error")
                return None
            finally:
                self.analyze_button.setEnabled(self.data is not None)
            self._present_result(profile, fit_result)
            self.profileChanged.emit(profile, fit_result)
            return fit_result

        def start_analysis(self) -> None:
            """Run one immutable frame/settings snapshot without blocking the UI."""
            if self.data is None or self.jobs_running():
                return
            self._closed = False
            parameters, payload = self._analysis_inputs(snapshot=True)
            self._prepare_analysis()
            worker = AnalysisWorker(
                _analyze_frame, generation=self._generation.next(), kind="azimuthal",
                parameters=parameters, payload=payload,
            )
            worker.signals.finished.connect(self._analysis_finished, QtCore.Qt.ConnectionType.QueuedConnection)
            worker.signals.error.connect(self._analysis_failed, QtCore.Qt.ConnectionType.QueuedConnection)
            self._worker = worker
            self.progress.show()
            self._thread_pool.start(worker)

        @QtCore.Slot(int, str, object)
        def _analysis_finished(self, generation: int, _kind: str, result: Any) -> None:
            self._worker = None
            self.progress.hide()
            if self._closed:
                return
            self.analyze_button.setEnabled(self.data is not None)
            if not self._generation.is_current(generation):
                if self.data is not None:
                    self._set_feedback(self._tr("ready"), "ready")
                return
            self._present_result(*result)
            self.profileChanged.emit(*result)

        @QtCore.Slot(int, str, object)
        def _analysis_failed(self, generation: int, _kind: str, error: Any) -> None:
            self._worker = None
            self.progress.hide()
            if self._closed:
                return
            self.analyze_button.setEnabled(self.data is not None)
            if self._generation.is_current(generation):
                self._set_feedback(self._tr("analysis_error").format(error=error), "error")
            elif self.data is not None:
                self._set_feedback(self._tr("ready"), "ready")

        def jobs_running(self) -> bool:
            return self._worker is not None or self._thread_pool.activeThreadCount() > 0

        def shutdown(self) -> None:
            """Discard pending output; the owner can await ``jobs_running`` asynchronously."""
            self._closed = True
            self._generation.next()

        def closeEvent(self, event: Any) -> None:  # noqa: N802 - Qt API
            self.shutdown()
            if self.jobs_running():
                event.ignore()
                QtCore.QTimer.singleShot(50, self.close)
                return
            event.accept()

        def showEvent(self, event: Any) -> None:  # noqa: N802 - Qt API
            super().showEvent(event)
            self.canvas.draw_idle()

        def _present_result(self, profile: AzimuthalProfile, fit_result: AzimuthalFitResult) -> None:
            self.profile = profile
            self.fit_result = fit_result
            self.export_button.setEnabled(True)
            n_observed = profile.n_supported_bins
            average_coverage = (
                float(np.mean(profile.coverage[profile.geometry_counts > 0]))
                if np.any(profile.geometry_counts > 0) else 0.0
            )
            summary = self._tr("analyzed").format(
                q0=profile.q_min,
                q1=profile.q_max,
                unit=_display_q_unit(profile.q_unit),
                n=n_observed,
                coverage=average_coverage,
            )
            if fit_result.success:
                summary += "\n" + self._tr("fit_status").format(
                    peaks=len(fit_result.peaks), rmse=fit_result.rmse
                )
            else:
                summary += "\n" + self._tr("fit_failed").format(message=fit_result.message)
            self._set_feedback(summary, "complete" if fit_result.success else "warning")
            self._draw()

        def _draw(self) -> None:
            if not hasattr(self, "profile_axes"):
                return
            font = font_properties(8, language=self.language)
            title_font = font_properties(11, weight="bold", language=self.language)
            previous_colorbar = getattr(self, "_image_colorbar", None)
            if previous_colorbar is not None:
                previous_colorbar.remove()
            self._image_colorbar = None
            self._q_contours = None
            self.image_axes.clear()
            self.profile_axes.clear()
            self.residual_axes.clear()
            for axes in (self.image_axes, self.profile_axes, self.residual_axes):
                axes.spines["top"].set_visible(False)
                axes.spines["right"].set_visible(False)
            for axes in (self.profile_axes, self.residual_axes):
                axes.set_xlim(0.0, 360.0)
                axes.set_xticks(np.arange(0.0, 361.0, 45.0))
                axes.grid(True, color="#d9e0e8", linewidth=0.7, alpha=0.7)
            self.image_axes.set_title(self._tr("preview_title"), fontproperties=title_font)
            self.image_axes.set_xlabel(self._tr("detector_x"), fontproperties=font)
            self.image_axes.set_ylabel(self._tr("detector_y"), fontproperties=font)
            self.image_axes.set_aspect("equal", adjustable="box")
            self.profile_axes.set_title(
                self._tr("profile_title"), loc="left", fontsize=11, fontproperties=title_font
            )
            self.profile_axes.set_ylabel(self._tr("intensity"), fontproperties=font)
            self.residual_axes.set_ylabel(self._tr("residual"), fontproperties=font)
            self.residual_axes.set_xlabel(self._tr("angle"), fontproperties=font)

            if self.data is not None and self.q_map is not None:
                finite_intensity = np.isfinite(self.data)
                finite_geometry = np.isfinite(self.q_map)
                selected = (
                    finite_intensity & finite_geometry & (self.q_map > 0.0) & self.valid_mask
                    & (self.q_map >= self.q_min_spin.value())
                    & (self.q_map <= self.q_max_spin.value())
                )
                if np.any(finite_intensity):
                    intensity_values = self.data[finite_intensity]
                    vmin, vmax = (float(value) for value in np.percentile(intensity_values, (1.0, 99.7)))
                    if not vmax > vmin:
                        pad = max(abs(vmin) * 0.01, 0.5)
                        vmin, vmax = vmin - pad, vmax + pad
                    alpha = np.where(selected, 1.0, 0.20).astype(np.float32)
                    alpha[~finite_intensity] = 0.0
                    image_artist = self.image_axes.imshow(
                        self.data,
                        origin="upper",
                        cmap="magma",
                        vmin=vmin,
                        vmax=vmax,
                        interpolation="nearest",
                        alpha=alpha,
                    )
                    self._image_colorbar = self.figure.colorbar(
                        image_artist, ax=self.image_axes, fraction=0.046, pad=0.035
                    )
                    self._image_colorbar.set_label(self._tr("intensity"), fontproperties=font)
                    for label in self._image_colorbar.ax.get_yticklabels():
                        label.set_fontproperties(font)

                invalid = ~self.valid_mask | ~finite_intensity | ~finite_geometry
                if np.any(invalid):
                    mask_overlay = np.ma.masked_where(
                        ~invalid, np.ones(self.data.shape, dtype=np.uint8)
                    )
                    self.image_axes.imshow(
                        mask_overlay,
                        origin="upper",
                        cmap=ListedColormap(["#62a5e5"]),
                        vmin=0,
                        vmax=1,
                        alpha=0.55,
                        interpolation="nearest",
                    )

                q_values = self.q_map[finite_geometry]
                if q_values.size:
                    q_low, q_high = float(np.min(q_values)), float(np.max(q_values))
                    q_min, q_max = self.q_min_spin.value(), self.q_max_spin.value()
                    levels = sorted({
                        float(level) for level in (q_min, q_max)
                        if np.isfinite(level) and q_low < level < q_high
                    })
                    if len(levels) > 0:
                        stride = max(1, int(np.ceil(max(self.q_map.shape) / 600.0)))
                        rows = np.arange(self.q_map.shape[0])[::stride]
                        cols = np.arange(self.q_map.shape[1])[::stride]
                        detector_x, detector_y = np.meshgrid(cols, rows)
                        q_sample = self.q_map[::stride, ::stride]
                        if q_sample.ndim == 2 and min(q_sample.shape) >= 2:
                            colors = ["#f7f7f7", "#45ead0"][:len(levels)]
                            self._q_contours = self.image_axes.contour(
                                detector_x,
                                detector_y,
                                q_sample,
                                levels=levels,
                                colors=colors,
                                linewidths=1.3,
                                linestyles="--",
                            )
                            contour_handles = [
                                Line2D([0], [0], color=color, linewidth=1.3, linestyle="--",
                                       label=f"q = {level:.4g} {_display_q_unit(self.q_unit)}")
                                for level, color in zip(levels, colors)
                            ]
                            self.image_axes.legend(
                                handles=contour_handles, loc="lower left", frameon=True, prop=font
                            )
                self.image_axes.set_title(
                    f"{self._tr('preview_title')} · {self.q_min_spin.value():.5g}–{self.q_max_spin.value():.5g} {_display_q_unit(self.q_unit)}",
                    fontproperties=title_font,
                )
                self.image_axes.set_xlim(-0.5, self.data.shape[1] - 0.5)
                self.image_axes.set_ylim(self.data.shape[0] - 0.5, -0.5)
            else:
                self.image_axes.text(
                    0.5, 0.5, self._tr("no_data"), transform=self.image_axes.transAxes,
                    ha="center", va="center", color="#687789", fontproperties=font,
                )

            if self.profile is not None:
                profile = self.profile
                observed = np.isfinite(profile.intensity)
                self.profile_axes.plot(
                    profile.angle_deg[observed], profile.intensity[observed],
                    linestyle="none", marker="o", markersize=3.2,
                    color="#2b6f9f", markeredgewidth=0, label=self._tr("measured"),
                )
                fit = self.fit_result
                if fit is not None:
                    self.profile_axes.plot(
                        profile.angle_deg, fit.model, color="#d55e00", linewidth=1.8,
                        label=self._tr("fit"),
                    )
                    if np.isfinite(fit.baseline):
                        self.profile_axes.axhline(
                            fit.baseline, color="#777777", linewidth=0.9,
                            linestyle=":", label=f"{self._tr('baseline')}: {fit.baseline:.4g}",
                        )
                    self.residual_axes.axhline(0.0, color="#444444", linewidth=0.8)
                    residual_valid = np.isfinite(fit.residual)
                    self.residual_axes.plot(
                        profile.angle_deg[residual_valid], fit.residual[residual_valid],
                        linestyle="none", marker="o", markersize=2.8, color="#7a5195",
                    )
                    if fit.peaks:
                        self.profile_axes.legend(loc="best", frameon=False, prop=font)
                    self.profile_axes.set_title(
                        f"{self._tr('profile_title')} · q = {profile.q_min:.5g}–{profile.q_max:.5g} {_display_q_unit(profile.q_unit)}",
                        loc="left", fontsize=11, fontproperties=title_font,
                    )
                else:
                    self.residual_axes.axhline(0.0, color="#444444", linewidth=0.8)
            else:
                self.profile_axes.text(
                    0.5, 0.5, self._tr("no_data"), transform=self.profile_axes.transAxes,
                    ha="center", va="center", color="#687789", fontproperties=font,
                )
                self.residual_axes.axhline(0.0, color="#444444", linewidth=0.8)
            for axes in (self.image_axes, self.profile_axes, self.residual_axes):
                for label in (*axes.get_xticklabels(), *axes.get_yticklabels()):
                    label.set_fontproperties(font)
            if self.canvas.isVisible():
                self.canvas.draw_idle()

        def _show_empty(self) -> None:
            self._draw()

        def export_to(self, profile_path: str | Path, *, overwrite: bool = False) -> tuple[Path, Path]:
            if self.profile is None or self.fit_result is None:
                raise ValueError("analyzed azimuthal profile is required for export")
            target = Path(profile_path)
            if target.suffix.lower() != ".csv":
                target = target.with_suffix(".csv")
            peaks_target = target.with_name(f"{target.stem}_peaks.csv")
            if not overwrite and (target.exists() or peaks_target.exists()):
                raise FileExistsError("one or both azimuthal CSV outputs already exist")
            return export_azimuthal_csv_bundle(
                self.profile, self.fit_result, target, overwrite=overwrite
            )

        def _choose_export_path(self) -> None:
            if self.profile is None or self.fit_result is None:
                return
            chosen, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, self._tr("export"), "azimuthal_profile.csv", self._tr("file_filter")
            )
            if not chosen:
                return
            target = Path(chosen)
            if target.suffix.lower() != ".csv":
                target = target.with_suffix(".csv")
            peaks_target = target.with_name(f"{target.stem}_peaks.csv")
            overwrite = target.exists() or peaks_target.exists()
            if overwrite:
                answer = QtWidgets.QMessageBox.question(
                    self, self._tr("export"), self._tr("export_exists"),
                    QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
                    QtWidgets.QMessageBox.StandardButton.No,
                )
                if answer != QtWidgets.QMessageBox.StandardButton.Yes:
                    return
            try:
                paths = self.export_to(target, overwrite=overwrite)
                self.status_label.setText(self._tr("export_done").format(
                    profile=paths[0].name, peaks=paths[1].name
                ))
            except (OSError, ValueError) as exc:
                self.status_label.setText(self._tr("export_error").format(error=exc))

        def document(self) -> dict[str, Any]:
            """Return restorable view settings without embedding detector arrays."""

            return {
                "schema": "butterfly_saxs.azimuthal_page.v1",
                "settings": {
                    "n_bins": self.bins_spin.value(),
                    "model": str(self.model_combo.currentData()),
                    "max_peaks": self.max_peaks_spin.value(),
                    "min_separation_deg": self.min_separation_spin.value(),
                    "min_height_fraction": self.height_fraction_spin.value(),
                    "initial_fwhm_deg": self.initial_width_spin.value(),
                    "eta": self.eta_spin.value(),
                },
                "q_unit": self.q_unit,
                "q_window": [self.q_min_spin.value(), self.q_max_spin.value()] if self.data is not None else None,
            }

        def restore_document(self, document: Any) -> None:
            if not isinstance(document, dict) or document.get("schema") != "butterfly_saxs.azimuthal_page.v1":
                raise ValueError("unsupported azimuthal page document")
            settings = document.get("settings", {})
            self.bins_spin.setValue(int(settings.get("n_bins", self.bins_spin.value())))
            model = str(settings.get("model", self.model_combo.currentData()))
            index = self.model_combo.findData(model)
            if index >= 0:
                self.model_combo.setCurrentIndex(index)
            self.max_peaks_spin.setValue(int(settings.get("max_peaks", self.max_peaks_spin.value())))
            self.min_separation_spin.setValue(float(settings.get("min_separation_deg", self.min_separation_spin.value())))
            self.height_fraction_spin.setValue(float(settings.get("min_height_fraction", self.height_fraction_spin.value())))
            self.initial_width_spin.setValue(float(settings.get("initial_fwhm_deg", self.initial_width_spin.value())))
            self.eta_spin.setValue(float(settings.get("eta", self.eta_spin.value())))
            q_window = document.get("q_window")
            if (self.data is not None and _same_q_unit(document.get("q_unit"), self.q_unit)
                    and isinstance(q_window, (list, tuple)) and len(q_window) == 2):
                lower = self.q_min_spin.minimum()
                upper = self.q_max_spin.maximum()
                q0 = float(np.clip(float(q_window[0]), lower, upper))
                q1 = float(np.clip(float(q_window[1]), lower, upper))
                if q1 > q0:
                    self.q_min_spin.setValue(q0)
                    self.q_max_spin.setValue(q1)

else:
    class AzimuthalPage:  # pragma: no cover - only used without optional UI dependencies
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            require_qt()


__all__ = ["AzimuthalPage"]
