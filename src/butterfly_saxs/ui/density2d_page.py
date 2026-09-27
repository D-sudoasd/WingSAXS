"""Interactive sector-resolved low-q analysis page for 2-D SAXS images."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..density2d import (
    Density2DFit,
    Density2DProfile,
    analyze_density2d,
    evaluate_density2d_model,
    export_density2d_bundle,
)
from ..settings import canonical_q_unit
from .qt_compat import QT_AVAILABLE, QtCore, QtWidgets, require_qt

if QT_AVAILABLE:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.colors import ListedColormap
    from matplotlib.figure import Figure

    from ..font_support import font_properties

    class Density2DPage(QtWidgets.QWidget):
        """Measure and fit a selected angular sector from the loaded 2-D image."""

        profileChanged = QtCore.Signal(object, object)

        def __init__(self, parent: Any = None, *, language: str = "zh_CN") -> None:
            super().__init__(parent)
            self.setObjectName("density2dPage")
            self.language = str(language)
            self.data: np.ndarray | None = None
            self.qx: np.ndarray | None = None
            self.qy: np.ndarray | None = None
            self.valid_mask: np.ndarray | None = None
            self.q_unit = "unknown"
            self.source: str | None = None
            self.profile: Density2DProfile | None = None
            self.fits: tuple[Density2DFit, ...] = ()
            self._window_is_explicit = False
            self._pending_q_window: tuple[float, float] | None = None
            self._pending_q_window_unit: str | None = None
            self._build_ui()
            self._apply_language()
            self._draw()

        def _build_ui(self) -> None:
            root = QtWidgets.QVBoxLayout(self)
            root.setContentsMargins(12, 10, 12, 10)
            root.setSpacing(8)

            self.heading = QtWidgets.QLabel()
            self.heading.setObjectName("density2dHeading")
            heading_font = self.heading.font()
            heading_font.setPointSize(15)
            heading_font.setBold(True)
            self.heading.setFont(heading_font)
            root.addWidget(self.heading)

            self.description = QtWidgets.QLabel()
            self.description.setWordWrap(True)
            self.description.setObjectName("density2dDescription")
            root.addWidget(self.description)

            controls = QtWidgets.QGridLayout()
            controls.setHorizontalSpacing(8)
            controls.setVerticalSpacing(6)
            self.q_min_spin = self._double_spin()
            self.q_min_spin.setObjectName("density2dQMin")
            self.q_max_spin = self._double_spin(0.0, 1.0e12, 1.0, 10)
            self.q_max_spin.setObjectName("density2dQMax")
            self.azimuth_spin = self._double_spin(-360.0, 360.0, 0.0, 1)
            self.azimuth_spin.setObjectName("density2dAzimuthCenter")
            self.half_width_spin = self._double_spin(0.1, 180.0, 18.0, 1)
            self.half_width_spin.setObjectName("density2dAzimuthHalfWidth")
            self.n_q_spin = QtWidgets.QSpinBox()
            self.n_q_spin.setRange(12, 1000)
            self.n_q_spin.setSingleStep(12)
            self.n_q_spin.setValue(120)
            self.n_q_spin.setObjectName("density2dRadialBins")

            self.power_law_check = QtWidgets.QCheckBox()
            self.power_law_check.setObjectName("density2dPowerLawEnabled")
            self.power_law_check.setChecked(True)
            self.ornstein_zernike_check = QtWidgets.QCheckBox()
            self.ornstein_zernike_check.setObjectName("density2dOrnsteinZernikeEnabled")
            self.ornstein_zernike_check.setChecked(True)

            self._labels: dict[str, Any] = {}
            control_items = [
                ("q_min", self.q_min_spin), ("q_max", self.q_max_spin),
                ("azimuth", self.azimuth_spin), ("half_width", self.half_width_spin),
                ("bins", self.n_q_spin),
            ]
            for index, (key, widget) in enumerate(control_items):
                row, column = divmod(index, 3)
                label = QtWidgets.QLabel()
                self._labels[key] = label
                controls.addWidget(label, row, column * 2)
                controls.addWidget(widget, row, column * 2 + 1)
            model_label = QtWidgets.QLabel()
            self._labels["models"] = model_label
            controls.addWidget(model_label, 2, 0)
            controls.addWidget(self.power_law_check, 2, 1, 1, 2)
            controls.addWidget(self.ornstein_zernike_check, 2, 3, 1, 3)
            root.addLayout(controls)

            actions = QtWidgets.QHBoxLayout()
            self.analyze_button = QtWidgets.QPushButton()
            self.analyze_button.setObjectName("density2dAnalyze")
            self.export_button = QtWidgets.QPushButton()
            self.export_button.setObjectName("density2dExport")
            self.export_button.setEnabled(False)
            self.status_label = QtWidgets.QLabel()
            self.status_label.setObjectName("density2dStatus")
            self.status_label.setWordWrap(True)
            actions.addWidget(self.analyze_button)
            actions.addWidget(self.export_button)
            actions.addWidget(self.status_label, 1)
            root.addLayout(actions)

            self.figure = Figure(figsize=(10.2, 5.8), constrained_layout=True)
            grid = self.figure.add_gridspec(2, 2, width_ratios=(1.1, 1.5), height_ratios=(3.0, 1.2))
            self.image_axes = self.figure.add_subplot(grid[:, 0])
            self.profile_axes = self.figure.add_subplot(grid[0, 1])
            self.residual_axes = self.figure.add_subplot(grid[1, 1], sharex=self.profile_axes)
            self.canvas = FigureCanvasQTAgg(self.figure)
            self.canvas.setObjectName("density2dCanvas")
            root.addWidget(self.canvas, 1)

            self.analyze_button.clicked.connect(self.analyze)
            self.export_button.clicked.connect(self._choose_export_path)
            self.azimuth_spin.valueChanged.connect(self._invalidate_analysis_result)
            self.half_width_spin.valueChanged.connect(self._invalidate_analysis_result)
            self.q_min_spin.valueChanged.connect(self._on_window_changed)
            self.q_max_spin.valueChanged.connect(self._on_window_changed)
            self.n_q_spin.valueChanged.connect(self._invalidate_analysis_result)
            self.power_law_check.toggled.connect(self._invalidate_analysis_result)
            self.ornstein_zernike_check.toggled.connect(self._invalidate_analysis_result)

        @staticmethod
        def _double_spin(
            minimum: float = 0.0,
            maximum: float = 1.0e12,
            value: float = 0.0,
            decimals: int = 8,
        ) -> Any:
            spin = QtWidgets.QDoubleSpinBox()
            spin.setRange(float(minimum), float(maximum))
            spin.setDecimals(int(decimals))
            spin.setValue(float(value))
            spin.setKeyboardTracking(False)
            return spin

        def _tr(self, key: str) -> str:
            english = str(self.language).lower().startswith("en")
            strings = {
                "heading": ("二维低 q 扇区分析", "2-D low-q sector analysis"),
                "description": (
                    "直接从二维图像提取所选 q 范围与方位角扇区的实测平均强度，并独立拟合经验幂律和 Ornstein–Zernike 模型。遮罩与空 q 分箱保持可见；只有物理 q 标定才报告实空间相关长度。",
                    "Measure mean intensity directly from the selected q range and angular sector in the 2-D image, then fit empirical power-law and Ornstein–Zernike candidates independently. Masked pixels and empty q bins remain visible; a real-space correlation length is reported only for calibrated physical q.",
                ),
                "q_min": ("q 下限", "q minimum"),
                "q_max": ("q 上限", "q maximum"),
                "azimuth": ("扇区中心角 (°)", "Sector centre (°)"),
                "half_width": ("扇区半宽 (°)", "Sector half-width (°)"),
                "bins": ("径向分箱", "Radial bins"),
                "models": ("拟合模型", "Fit models"),
                "power_law": ("幂律", "Power law"),
                "ornstein_zernike": ("Ornstein–Zernike", "Ornstein–Zernike"),
                "analyze": ("分析所选扇区", "Analyze selected sector"),
                "export": ("导出 JSON + CSV", "Export JSON + CSV"),
                "no_data": ("请先载入二维图像及其 q 坐标。", "Load a 2-D image and its q coordinates to begin."),
                "region_title": ("图像与所选 q 扇区", "Image and selected q sector"),
                "profile_title": ("扇区实测强度与经验拟合", "Measured sector profile and empirical fits"),
                "residual_title": ("拟合残差", "Fit residuals"),
                "pixel_x": ("像素 x", "Pixel x"),
                "pixel_y": ("像素 y", "Pixel y"),
                "q_axis": ("q ({unit})", "q ({unit})"),
                "intensity": ("平均强度 (输入单位)", "Mean intensity (input units)"),
                "residual": ("实测 − 拟合", "Observed − fit"),
                "measured": ("实测均值", "Measured mean"),
                "no_profile": ("尚未分析二维扇区", "No 2-D sector analyzed"),
                "analyzed": (
                    "q = {q0:.5g}–{q1:.5g} {unit}；角度 {angle:.1f}° ± {width:.1f}°；有效分箱 {bins}/{total}；平均覆盖 {coverage:.1%}。",
                    "q = {q0:.5g}–{q1:.5g} {unit}; sector {angle:.1f}° ± {width:.1f}°; measured bins {bins}/{total}; mean coverage {coverage:.1%}.",
                ),
                "fit_ok": (
                    "{name} 候选：{details}；RMSE {rmse:.4g}",
                    "{name} candidate: {details}; RMSE {rmse:.4g}",
                ),
                "fit_failed": ("{name} 未拟合：{message}", "{name} fit failed: {message}"),
                "weak_identification": ("辨识度有限", "weakly identified"),
                "export_done": ("已导出 {csv} 和 {json}", "Exported {csv} and {json}"),
                "export_exists": ("CSV 或 JSON 文件已存在，是否覆盖？", "A CSV or JSON output already exists. Overwrite both?"),
                "export_error": ("导出失败：{error}", "Export failed: {error}"),
                "analysis_error": ("分析失败：{error}", "Analysis failed: {error}"),
                "model_required": ("至少选择一个模型。", "Select at least one model."),
                "file_filter": ("CSV 文件 (*.csv)", "CSV files (*.csv)"),
            }
            if key == "power_law_check":
                return strings["power_law"][1 if english else 0]
            if key == "ornstein_zernike_check":
                return strings["ornstein_zernike"][1 if english else 0]
            return strings[key][1 if english else 0]

        def _apply_language(self) -> None:
            self.heading.setText(self._tr("heading"))
            self.description.setText(self._tr("description"))
            for key, label in self._labels.items():
                label.setText(self._tr(key))
            self.power_law_check.setText(self._tr("power_law_check"))
            self.ornstein_zernike_check.setText(self._tr("ornstein_zernike_check"))
            self.analyze_button.setText(self._tr("analyze"))
            self.export_button.setText(self._tr("export"))
            self._draw()

        def set_language(self, language: str) -> None:
            self.language = str(language)
            self._apply_language()

        def _on_window_changed(self, *_args: Any) -> None:
            if self.data is not None:
                self._window_is_explicit = True
            self._invalidate_analysis_result()

        def _invalidate_analysis_result(self, *_args: Any) -> None:
            """Drop fitted outputs when a control changes their source settings."""

            if self.profile is not None or self.fits:
                self.profile = None
                self.fits = ()
                self.export_button.setEnabled(False)
                self.status_label.setText(self._tr("no_profile"))
            self._draw()

        def set_data(
            self,
            data: Any,
            *,
            qx: Any,
            qy: Any,
            valid_mask: Any = None,
            q_unit: str = "unknown",
            source: Any = None,
        ) -> None:
            """Set one image; ``valid_mask=True`` marks usable detector pixels."""

            image = np.asarray(data, dtype=float)
            if image.ndim != 2 or image.size == 0:
                self.invalidate(self._tr("analysis_error").format(error="data must be a non-empty 2-D image"))
                raise ValueError("data must be a non-empty two-dimensional image")
            try:
                qx_map, qy_map = np.broadcast_arrays(np.asarray(qx, dtype=float), np.asarray(qy, dtype=float))
                qx_map = np.broadcast_to(qx_map, image.shape)
                qy_map = np.broadcast_to(qy_map, image.shape)
                if valid_mask is None:
                    usable = np.ones(image.shape, dtype=bool)
                else:
                    usable = np.broadcast_to(np.asarray(valid_mask, dtype=bool), image.shape)
            except (TypeError, ValueError) as exc:
                self.invalidate(self._tr("analysis_error").format(error="qx, qy, and valid_mask must match image shape"))
                raise ValueError("qx, qy, and valid_mask must broadcast to the image shape") from exc

            q_radius = np.hypot(qx_map, qy_map)
            finite_q = q_radius[np.isfinite(q_radius)]
            if not finite_q.size:
                self.invalidate(self._tr("analysis_error").format(error="no finite q coordinates"))
                raise ValueError("q maps contain no finite coordinates")
            q_lower, q_upper = float(np.min(finite_q)), float(np.max(finite_q))
            if q_upper <= q_lower:
                self.invalidate(self._tr("analysis_error").format(error="q map has no radial range"))
                raise ValueError("q map has no radial range")

            preserve = (
                self.data is not None
                and self._window_is_explicit
                and canonical_q_unit(self.q_unit) == canonical_q_unit(q_unit)
            )
            pending_matches_unit = (
                self._pending_q_window is not None
                and self._pending_q_window_unit is not None
                and canonical_q_unit(self._pending_q_window_unit) == canonical_q_unit(q_unit)
            )
            if pending_matches_unit:
                previous_window = self._pending_q_window
            elif self._pending_q_window is None and preserve:
                previous_window = self.q_min_spin.value(), self.q_max_spin.value()
            else:
                previous_window = None
            self.data = image
            self.qx = qx_map
            self.qy = qy_map
            self.valid_mask = usable
            self.q_unit = str(q_unit or "unknown")
            self.source = str(source) if source is not None else None
            self.profile = None
            self.fits = ()

            span = q_upper - q_lower
            self.q_min_spin.blockSignals(True)
            self.q_max_spin.blockSignals(True)
            self.q_min_spin.setRange(q_lower, q_upper)
            self.q_max_spin.setRange(q_lower, q_upper)
            self.q_min_spin.setDecimals(10)
            self.q_max_spin.setDecimals(10)
            step = max(span / 100.0, np.finfo(float).eps)
            self.q_min_spin.setSingleStep(step)
            self.q_max_spin.setSingleStep(step)
            if previous_window and previous_window[1] > previous_window[0]:
                q0 = float(np.clip(previous_window[0], q_lower, q_upper))
                q1 = float(np.clip(previous_window[1], q_lower, q_upper))
                if q1 <= q0:
                    q0 = q_lower
                    q1 = q_lower + 0.35 * span
            else:
                q0 = q_lower
                q1 = q_lower + 0.35 * span
            self.q_min_spin.setValue(q0)
            self.q_max_spin.setValue(q1)
            self.q_min_spin.blockSignals(False)
            self.q_max_spin.blockSignals(False)
            self._window_is_explicit = previous_window is not None
            self._pending_q_window = None
            self._pending_q_window_unit = None
            self.analyze_button.setEnabled(True)
            self.export_button.setEnabled(False)
            self.status_label.setText(self._tr("no_profile"))
            self._draw()

        def invalidate(self, message: str | None = None) -> None:
            """Clear image measurements after image, calibration, or mask changes."""

            self.data = None
            self.qx = None
            self.qy = None
            self.valid_mask = None
            self.q_unit = "unknown"
            self.source = None
            self.profile = None
            self.fits = ()
            self._window_is_explicit = False
            self.analyze_button.setEnabled(False)
            self.export_button.setEnabled(False)
            self.status_label.setText(message or self._tr("no_data"))
            self._draw()

        def _selected_models(self) -> tuple[str, ...]:
            models = []
            if self.power_law_check.isChecked():
                models.append("power_law")
            if self.ornstein_zernike_check.isChecked():
                models.append("ornstein_zernike")
            return tuple(models)

        def analyze(self) -> tuple[Density2DFit, ...] | None:
            if self.data is None or self.qx is None or self.qy is None or self.valid_mask is None:
                return None
            models = self._selected_models()
            if not models:
                self.status_label.setText(self._tr("model_required"))
                return None
            self.profile = None
            self.fits = ()
            self.export_button.setEnabled(False)
            self.status_label.setText(self._tr("no_profile"))
            try:
                result = analyze_density2d(
                    self.data,
                    qx=self.qx,
                    qy=self.qy,
                    q_window=(self.q_min_spin.value(), self.q_max_spin.value()),
                    azimuth_center_deg=self.azimuth_spin.value(),
                    azimuth_half_width_deg=self.half_width_spin.value(),
                    n_q=self.n_q_spin.value(),
                    valid_mask=self.valid_mask,
                    q_unit=self.q_unit,
                    source=self.source,
                    models=models,
                )
            except (TypeError, ValueError, RuntimeError) as exc:
                self.status_label.setText(self._tr("analysis_error").format(error=exc))
                self._draw()
                return None
            self.profile = result.profile
            self.fits = result.fits
            self.export_button.setEnabled(True)
            geometry_bins = np.asarray(result.profile.geometry_counts) > 0
            average_coverage = (
                float(np.mean(result.profile.coverage[geometry_bins]))
                if np.any(geometry_bins) else 0.0
            )
            summary = self._tr("analyzed").format(
                q0=result.profile.q_min,
                q1=result.profile.q_max,
                unit=result.profile.q_unit,
                angle=result.profile.azimuth_center_deg,
                width=result.profile.azimuth_half_width_deg,
                bins=result.profile.n_supported_bins,
                total=result.profile.q.size,
                coverage=average_coverage,
            )
            details = [summary]
            for fit in self.fits:
                if not fit.success:
                    name = self._model_label(fit.model_name)
                    details.append(self._tr("fit_failed").format(name=name, message=fit.message))
                    continue
                if fit.model_name == "power_law":
                    fit_details = f"α={float(fit.parameters['alpha']):.4g}; R²(log I)={float(fit.r_squared):.4g}"
                elif "xi" in fit.parameters:
                    fit_details = (
                        f"ξ={float(fit.parameters['xi']):.4g} {fit.parameters['xi_unit']}; "
                        f"R²(I)={float(fit.r_squared):.4g}"
                    )
                elif "xi_candidate" in fit.parameters:
                    fit_details = (
                        f"ξ candidate={float(fit.parameters['xi_candidate']):.4g} {fit.parameters['xi_unit']} "
                        f"({self._tr('weak_identification')}); R²(I)={float(fit.r_squared):.4g}"
                    )
                else:
                    fit_details = (
                        f"qref·ξ={float(fit.parameters['q_reference_times_xi']):.4g}; "
                        f"R²(I)={float(fit.r_squared):.4g}"
                    )
                details.append(self._tr("fit_ok").format(
                    name=self._model_label(fit.model_name),
                    details=fit_details,
                    rmse=float(fit.rmse or 0.0),
                ))
            self.status_label.setText("\n".join(details))
            self._draw()
            self.profileChanged.emit(self.profile, self.fits)
            return self.fits

        def _model_label(self, model_name: str) -> str:
            return self._tr("power_law_check") if model_name == "power_law" else self._tr("ornstein_zernike_check")

        def _selection(self, qx_map: np.ndarray, qy_map: np.ndarray) -> np.ndarray:
            q = np.hypot(qx_map, qy_map)
            angle = np.mod(np.degrees(np.arctan2(qy_map, qx_map)), 360.0)
            delta = (angle - self.azimuth_spin.value() + 180.0) % 360.0 - 180.0
            return (
                np.isfinite(qx_map) & np.isfinite(qy_map) & np.isfinite(q)
                & (q > 0.0)
                & (q >= self.q_min_spin.value()) & (q <= self.q_max_spin.value())
                & (np.abs(delta) <= self.half_width_spin.value())
            )

        def _draw_preview(self, font: Any, title_font: Any) -> None:
            axes = self.image_axes
            axes.clear()
            axes.spines["top"].set_visible(False)
            axes.spines["right"].set_visible(False)
            axes.set_title(self._tr("region_title"), loc="left", fontproperties=title_font)
            axes.set_xlabel(self._tr("pixel_x"), fontproperties=font)
            axes.set_ylabel(self._tr("pixel_y"), fontproperties=font)
            if self.data is None or self.qx is None or self.qy is None or self.valid_mask is None:
                axes.set_facecolor("#f1f4f7")
                axes.text(
                    0.5, 0.5, self._tr("no_data"), transform=axes.transAxes,
                    ha="center", va="center", color="#687789", fontproperties=font,
                )
                return

            height, width = self.data.shape
            stride = max(1, int(np.ceil(np.sqrt(self.data.size / 1_100_000))))
            image = self.data[::stride, ::stride]
            qx_map = self.qx[::stride, ::stride]
            qy_map = self.qy[::stride, ::stride]
            usable = self.valid_mask[::stride, ::stride] & np.isfinite(image)
            selected = self._selection(qx_map, qy_map) & usable
            visible = image[usable]
            if visible.size:
                lower, upper = np.percentile(visible, (1.0, 99.0))
                if not upper > lower:
                    lower, upper = float(np.min(visible)), float(np.max(visible) + 1.0)
                shown = np.ma.masked_where(~usable, image)
                axes.imshow(
                    shown, cmap="gray", interpolation="nearest", origin="upper",
                    extent=(-0.5, width - 0.5, height - 0.5, -0.5),
                    vmin=float(lower), vmax=float(upper), rasterized=True,
                )
                dim = usable & ~selected
                dim_overlay = np.ma.masked_where(~dim, np.ones(image.shape, dtype=float))
                axes.imshow(
                    dim_overlay,
                    cmap=ListedColormap(["#101923"]), vmin=0.0, vmax=1.0,
                    alpha=0.58, interpolation="nearest", origin="upper",
                    extent=(-0.5, width - 0.5, height - 0.5, -0.5), rasterized=True,
                )
                if np.any(selected) and np.any(~selected):
                    try:
                        axes.contour(
                            selected.astype(float), levels=(0.5,), colors=("#f0a43a",),
                            linewidths=0.9, origin="upper",
                            extent=(-0.5, width - 0.5, height - 0.5, -0.5),
                        )
                    except ValueError:
                        pass
                axes.set_xlim(-0.5, width - 0.5)
                axes.set_ylim(height - 0.5, -0.5)
                axes.set_aspect("equal", adjustable="box")
            else:
                axes.set_facecolor("#f1f4f7")
                axes.text(
                    0.5, 0.5, self._tr("no_data"), transform=axes.transAxes,
                    ha="center", va="center", color="#687789", fontproperties=font,
                )
            for label in (*axes.get_xticklabels(), *axes.get_yticklabels()):
                label.set_fontproperties(font)

        def _draw(self) -> None:
            if not hasattr(self, "profile_axes"):
                return
            font = font_properties(8, language=self.language)
            title_font = font_properties(10, weight="bold", language=self.language)
            self.profile_axes.set_xscale("linear")
            self.profile_axes.set_yscale("linear")
            self.residual_axes.set_xscale("linear")
            self.profile_axes.clear()
            self.residual_axes.clear()
            self._draw_preview(font, title_font)
            for axes in (self.profile_axes, self.residual_axes):
                axes.grid(True, which="both", color="#d9e0e8", linewidth=0.7, alpha=0.75)
                axes.spines["top"].set_visible(False)
                axes.spines["right"].set_visible(False)
            self.profile_axes.set_title(
                self._tr("profile_title"), loc="left", fontproperties=title_font
            )
            self.profile_axes.set_ylabel(self._tr("intensity"), fontproperties=font)
            self.residual_axes.set_title(
                self._tr("residual_title"), loc="left", fontproperties=font
            )
            self.residual_axes.set_ylabel(self._tr("residual"), fontproperties=font)
            unit = self.q_unit
            self.residual_axes.set_xlabel(self._tr("q_axis").format(unit=unit), fontproperties=font)
            self.profile_axes.tick_params(labelbottom=False)

            if self.profile is not None:
                profile = self.profile
                supported = np.isfinite(profile.q) & np.isfinite(profile.intensity) & (profile.counts > 0)
                self.profile_axes.plot(
                    profile.q[supported], profile.intensity[supported], linestyle="none",
                    marker="o", markersize=3.3, color="#276b8e", markeredgewidth=0,
                    label=self._tr("measured"), zorder=3,
                )
                colors = {"power_law": "#d45d00", "ornstein_zernike": "#6f55a3"}
                labels: list[str] = []
                for fit in self.fits:
                    if not fit.success or fit.model_name not in self._selected_models():
                        continue
                    color = colors[fit.model_name]
                    q_grid = np.geomspace(float(fit.q_min), float(fit.q_max), 180)
                    fitted_grid = evaluate_density2d_model(q_grid, fit.model_name, fit.parameters)
                    visible = np.isfinite(fitted_grid)
                    self.profile_axes.plot(
                        q_grid[visible], fitted_grid[visible], color=color, linewidth=1.8,
                        label=self._fit_legend(fit), zorder=2,
                    )
                    residual_valid = np.isfinite(fit.residual) & np.isfinite(profile.q)
                    self.residual_axes.plot(
                        profile.q[residual_valid], fit.residual[residual_valid], linestyle="none",
                        marker="o", markersize=2.8, color=color, label=self._model_label(fit.model_name),
                    )
                    labels.append(fit.model_name)
                self.residual_axes.axhline(0.0, color="#414b55", linewidth=0.8)
                if labels:
                    self.residual_axes.legend(loc="best", frameon=False, prop=font)
                positive_q = profile.q[supported & (profile.q > 0.0)]
                if positive_q.size:
                    self.profile_axes.set_xscale("log")
                    self.residual_axes.set_xscale("log")
                    self.profile_axes.set_xlim(float(np.min(positive_q)), float(np.max(positive_q)))
                observed = profile.intensity[supported]
                selected_fit_values = [
                    fit.fitted_intensity[np.isfinite(fit.fitted_intensity)]
                    for fit in self.fits if fit.success
                ]
                log_values = [observed, *selected_fit_values]
                if log_values and all(values.size and np.all(values > 0.0) for values in log_values):
                    self.profile_axes.set_yscale("log")
                if self.fits or np.any(supported):
                    self.profile_axes.legend(loc="best", frameon=False, prop=font)
                for axes in (self.profile_axes, self.residual_axes):
                    for label in (*axes.get_xticklabels(), *axes.get_yticklabels()):
                        label.set_fontproperties(font)
            else:
                self.profile_axes.text(
                    0.5, 0.5, self._tr("no_profile"), transform=self.profile_axes.transAxes,
                    ha="center", va="center", color="#687789", fontproperties=font,
                )
                self.residual_axes.axhline(0.0, color="#414b55", linewidth=0.8)
            self.canvas.draw_idle()

        def _fit_legend(self, fit: Density2DFit) -> str:
            if fit.model_name == "power_law":
                return f"Power law · α={float(fit.parameters['alpha']):.3g}"
            if "xi" in fit.parameters:
                return f"OZ · ξ={float(fit.parameters['xi']):.3g} {fit.parameters['xi_unit']}"
            if "xi_candidate" in fit.parameters:
                return f"OZ · ξ*={float(fit.parameters['xi_candidate']):.3g} {fit.parameters['xi_unit']}"
            return "OZ empirical fit"

        def export_to(self, csv_path: str | Path, *, overwrite: bool = False) -> tuple[Path, Path]:
            if self.profile is None or not self.fits:
                raise ValueError("analyzed 2-D sector profile is required for export")
            return export_density2d_bundle(self.profile, self.fits, csv_path, overwrite=overwrite)

        def _choose_export_path(self) -> None:
            if self.profile is None or not self.fits:
                return
            chosen, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, self._tr("export"), "density_2d_sector.csv", self._tr("file_filter")
            )
            if not chosen:
                return
            target = Path(chosen)
            if target.suffix.lower() != ".csv":
                target = target.with_suffix(".csv")
            json_target = target.with_suffix(".json")
            overwrite = target.exists() or json_target.exists()
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
                    csv=paths[0].name, json=paths[1].name
                ))
            except (OSError, ValueError) as exc:
                self.status_label.setText(self._tr("export_error").format(error=exc))

        def document(self) -> dict[str, Any]:
            """Return restorable page settings without detector arrays."""

            return {
                "schema": "butterfly_saxs.density2d_page.v1",
                "settings": {
                    "q_window": [self.q_min_spin.value(), self.q_max_spin.value()],
                    "azimuth_center_deg": self.azimuth_spin.value(),
                    "azimuth_half_width_deg": self.half_width_spin.value(),
                    "n_q": self.n_q_spin.value(),
                    "models": list(self._selected_models()),
                },
                "q_unit": self.q_unit,
            }

        def restore_document(self, document: Any) -> None:
            if not isinstance(document, dict) or document.get("schema") != "butterfly_saxs.density2d_page.v1":
                raise ValueError("unsupported 2-D density page document")
            self._pending_q_window = None
            self._pending_q_window_unit = None
            settings = document.get("settings", {})
            self.azimuth_spin.setValue(float(settings.get("azimuth_center_deg", self.azimuth_spin.value())))
            self.half_width_spin.setValue(float(settings.get("azimuth_half_width_deg", self.half_width_spin.value())))
            self.n_q_spin.setValue(int(settings.get("n_q", self.n_q_spin.value())))
            models = set(settings.get("models", ("power_law", "ornstein_zernike")))
            self.power_law_check.setChecked("power_law" in models)
            self.ornstein_zernike_check.setChecked("ornstein_zernike" in models)
            q_window = settings.get("q_window")
            if isinstance(q_window, (tuple, list)) and len(q_window) == 2:
                parsed = float(q_window[0]), float(q_window[1])
                if np.all(np.isfinite(parsed)) and parsed[0] < parsed[1]:
                    if self.data is not None:
                        saved_unit = document.get("q_unit")
                        if (
                            saved_unit is not None
                            and canonical_q_unit(saved_unit) == canonical_q_unit(self.q_unit)
                        ):
                            lower = self.q_min_spin.minimum()
                            upper = self.q_max_spin.maximum()
                            q0 = float(np.clip(parsed[0], lower, upper))
                            q1 = float(np.clip(parsed[1], lower, upper))
                            if q1 > q0:
                                self.q_min_spin.setValue(q0)
                                self.q_max_spin.setValue(q1)
                                self._window_is_explicit = True
                    else:
                        self._pending_q_window = parsed
                        saved_unit = document.get("q_unit")
                        self._pending_q_window_unit = str(saved_unit) if saved_unit is not None else None

else:
    class Density2DPage:  # pragma: no cover - only used without optional UI dependencies
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            require_qt()


__all__ = ["Density2DPage"]
