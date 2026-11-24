"""Interactive q-annulus azimuthal profile and peak fitting page."""

from __future__ import annotations

from collections.abc import Callable
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
from ._common import (
    display_q_unit as _display_q_unit,
    normalize_save_path,
    same_q_unit as _same_q_unit,
)
from ._measurement_base import _MeasurementCanvasPage
from .qt_compat import QT_AVAILABLE, QtWidgets, require_qt


def _analyze_frame(parameters: dict[str, Any], payload: dict[str, Any]) -> tuple[Any, Any]:
    """Measure and fit a plain frame snapshot without accessing Qt widgets."""

    profile = measure_azimuthal_profile(**payload, **parameters["profile"])
    return profile, fit_azimuthal_peaks(profile, **parameters["fit"])


if QT_AVAILABLE:
    from matplotlib.colors import ListedColormap
    from matplotlib.lines import Line2D

    from ..font_support import font_properties

    class AzimuthalPage(_MeasurementCanvasPage):
        """Analyze an angular profile measured from an explicitly chosen q band."""

        _worker_kind = "azimuthal"

        def __init__(self, parent: Any = None, *, language: str = "zh_CN") -> None:
            super().__init__(parent, page_prefix="azimuthal", language=language)
            self.q_map: np.ndarray | None = None
            self.fit_result: AzimuthalFitResult | None = None
            self._build_ui()
            self.set_language(self.language)
            self._show_empty()

        def _build_ui(self) -> None:
            controls = self._build_page_scaffold(
                prefix="azimuthal",
                settings_max_height=240,
                figsize=(12.0, 6.5),
                width_ratios=(1.0, 1.45),
                height_ratios=(3.0, 1.0),
            )
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
                ("pseudo_voigt", "pseudo-Voigt"),
                ("gaussian", "Gaussian"),
                ("lorentzian", "Lorentzian"),
                ("von_mises", "von Mises"),
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
            self._populate_control_grid(
                controls,
                (
                    ("q_min", self.q_min_spin),
                    ("q_max", self.q_max_spin),
                    ("bins", self.bins_spin),
                    ("model", self.model_combo),
                    ("max_peaks", self.max_peaks_spin),
                    ("separation", self.min_separation_spin),
                    ("height_fraction", self.height_fraction_spin),
                    ("initial_width", self.initial_width_spin),
                    ("eta", self.eta_spin),
                ),
            )
            self.model_combo.currentIndexChanged.connect(self._sync_fit_controls)
            for control, signal_name in (
                (self.q_min_spin, "valueChanged"),
                (self.q_max_spin, "valueChanged"),
                (self.bins_spin, "valueChanged"),
                (self.model_combo, "currentIndexChanged"),
                (self.max_peaks_spin, "valueChanged"),
                (self.min_separation_spin, "valueChanged"),
                (self.height_fraction_spin, "valueChanged"),
                (self.initial_width_spin, "valueChanged"),
                (self.eta_spin, "valueChanged"),
            ):
                getattr(control, signal_name).connect(self._invalidate_measurement)
            self._sync_fit_controls()

        def _tr(self, key: str) -> str:
            english = str(self.language).lower().startswith("en")
            strings = {
                "heading": ("方位角峰拟合", "Azimuthal peak fitting"),
                "description": (
                    "从二维散射图像中选择 q 环带，检查角向强度分布并拟合周期峰。拟合只使用实际测得的有效角度点。",
                    "Choose a q annulus in the 2-D scattering image, inspect its angular intensity profile, and fit periodic peaks. Fits use measured valid angles only.",
                ),
                "q_min": ("q 下限", "q minimum"),
                "q_max": ("q 上限", "q maximum"),
                "bins": ("角度分箱", "Angular bins"),
                "model": ("峰模型", "Peak model"),
                "max_peaks": ("最多峰数", "Maximum peaks"),
                "separation": ("最小峰间距 (°)", "Minimum separation (°)"),
                "height_fraction": ("最小相对高度", "Minimum relative height"),
                "initial_width": ("初始 FWHM (°)", "Initial FWHM (°)"),
                "eta": ("pseudo-Voigt η", "pseudo-Voigt η"),
                "analyze": ("分析环带", "Analyze annulus"),
                "settings": ("环带与峰拟合设置", "Annulus and peak fit settings"),
                "ready": (
                    "二维图像和 q 坐标已就绪；设置环带后点击分析。",
                    "Image and q coordinates ready; choose an annulus, then analyze.",
                ),
                "running": ("正在提取角向强度并拟合峰…", "Measuring angular intensity and fitting peaks…"),
                "changed_running": (
                    "数据或设置已变化；等待当前计算结束后重新分析。",
                    "Data or settings changed; analyze again when the current calculation finishes.",
                ),
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
                "analyzed": (
                    "q = {q0:.6g}–{q1:.6g} {unit}；有效角度点 {n}；平均覆盖率 {coverage:.1%}。",
                    "q = {q0:.6g}–{q1:.6g} {unit}; {n} measured angular bins; mean coverage {coverage:.1%}.",
                ),
                "fit_status": ("拟合成功：{peaks} 个峰，RMSE {rmse:.4g}。", "Fit succeeded: {peaks} peaks, RMSE {rmse:.4g}."),
                "fit_failed": ("未得到有效峰拟合：{message}", "No valid peak fit: {message}"),
                "export_done": ("已导出：{profile} 和 {peaks}", "Exported {profile} and {peaks}"),
                "export_exists": ("文件已存在，是否覆盖？", "The files already exist. Overwrite both files?"),
                "export_error": ("导出失败：{error}", "Export failed: {error}"),
                "analysis_error": ("分析失败：{error}", "Analysis failed: {error}"),
                "file_filter": ("CSV 文件 (*.csv)", "CSV files (*.csv)"),
            }
            return strings[key][0 if not english else 1]

        def _clear_results(self) -> None:
            self.profile = None
            self.fit_result = None

        def _present_current_result(self) -> None:
            assert self.profile is not None and self.fit_result is not None
            self._present_result(self.profile, self.fit_result)

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
            image, qx_array, qy_array, valid_array, finite_q, lower, upper = (
                self._validate_frame_arrays(
                    data,
                    qx=qx,
                    qy=qy,
                    valid_mask=valid_mask,
                    empty_q_error="no finite q coordinates",
                )
            )
            previous_window = (
                (self.q_min_spin.value(), self.q_max_spin.value())
                if self.data is not None
                else None
            )
            previous_unit = self.q_unit
            self._clear_results()
            self.data = image
            self.qx = qx_array
            self.qy = qy_array
            self.q_map = finite_q
            self.valid_mask = valid_array
            self.q_unit = str(q_unit or "unknown")
            self.source = str(source) if source is not None else None
            if (
                previous_window
                and _same_q_unit(previous_unit, self.q_unit)
                and previous_window[1] > previous_window[0]
            ):
                q_min = float(np.clip(previous_window[0], lower, upper))
                q_max = float(np.clip(previous_window[1], lower, upper))
                if q_max <= q_min:
                    q_min, q_max = lower, upper
            else:
                q_min, q_max = lower, upper
            self._configure_q_spin_bounds(lower, upper, q_min, q_max)
            self._clear_results()
            self._finish_set_data()

        def invalidate(self, message: str | None = None) -> None:
            """Clear current measurements after image, calibration, or mask changes."""

            self.q_map = None
            self._reset_base_state(message)
            self._show_empty()

        def _sync_fit_controls(self, *_args: Any) -> None:
            self.eta_spin.setEnabled(self.model_combo.currentData() == "pseudo_voigt")

        def _invalidate_measurement(self, *_args: Any) -> None:
            """Drop a result as soon as one of its analysis settings changes."""

            self._generation.next()
            if self.jobs_running():
                self._set_feedback(self._tr("changed_running"), "running")
            if self.profile is None and self.fit_result is None:
                if self.sender() in (self.q_min_spin, self.q_max_spin) and self.data is not None:
                    self._draw()
                return
            self._clear_results()
            self.export_button.setEnabled(False)
            if not self.jobs_running():
                self._set_feedback(self._tr("ready"), "ready")
            self._draw()

        def _analysis_inputs(self, *, snapshot: bool) -> tuple[dict[str, Any], dict[str, Any]]:
            payload = {
                "data": self.data,
                "qx": self.qx,
                "qy": self.qy,
                "valid_mask": self.valid_mask,
                "q_unit": self.q_unit,
                "source": self.source,
            }
            if snapshot:
                for key in ("data", "qx", "qy", "valid_mask"):
                    payload[key] = np.array(payload[key], copy=True)
            parameters = {
                "profile": {
                    "q_window": (self.q_min_spin.value(), self.q_max_spin.value()),
                    "n_bins": self.bins_spin.value(),
                    "statistic": "mean",
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

        def _worker_job(self) -> Callable[..., Any]:
            return _analyze_frame

        def _worker_request_args(self) -> tuple[dict[str, Any], Any]:
            return self._analysis_inputs(snapshot=True)

        def _run_sync_job(self) -> Any:
            return _analyze_frame(*self._analysis_inputs(snapshot=False))

        def _apply_worker_result(self, result: Any) -> AzimuthalFitResult:
            profile, fit_result = result
            self._present_result(profile, fit_result)
            self.profileChanged.emit(profile, fit_result)
            return fit_result

        def _present_result(self, profile: AzimuthalProfile, fit_result: AzimuthalFitResult) -> None:
            self.profile = profile
            self.fit_result = fit_result
            self.export_button.setEnabled(True)
            n_observed = profile.n_supported_bins
            average_coverage = (
                float(np.mean(profile.coverage[profile.geometry_counts > 0]))
                if np.any(profile.geometry_counts > 0)
                else 0.0
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
                    finite_intensity
                    & finite_geometry
                    & (self.q_map > 0.0)
                    & self.valid_mask
                    & (self.q_map >= self.q_min_spin.value())
                    & (self.q_map <= self.q_max_spin.value())
                )
                if np.any(finite_intensity):
                    intensity_values = self.data[finite_intensity]
                    vmin, vmax = (
                        float(value) for value in np.percentile(intensity_values, (1.0, 99.7))
                    )
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
                    levels = sorted(
                        {
                            float(level)
                            for level in (q_min, q_max)
                            if np.isfinite(level) and q_low < level < q_high
                        }
                    )
                    if len(levels) > 0:
                        stride = max(1, int(np.ceil(max(self.q_map.shape) / 600.0)))
                        rows = np.arange(self.q_map.shape[0])[::stride]
                        cols = np.arange(self.q_map.shape[1])[::stride]
                        detector_x, detector_y = np.meshgrid(cols, rows)
                        q_sample = self.q_map[::stride, ::stride]
                        if q_sample.ndim == 2 and min(q_sample.shape) >= 2:
                            colors = ["#f7f7f7", "#45ead0"][: len(levels)]
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
                                Line2D(
                                    [0],
                                    [0],
                                    color=color,
                                    linewidth=1.3,
                                    linestyle="--",
                                    label=f"q = {level:.4g} {_display_q_unit(self.q_unit)}",
                                )
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
                    0.5,
                    0.5,
                    self._tr("no_data"),
                    transform=self.image_axes.transAxes,
                    ha="center",
                    va="center",
                    color="#687789",
                    fontproperties=font,
                )

            if self.profile is not None:
                profile = self.profile
                observed = np.isfinite(profile.intensity)
                self.profile_axes.plot(
                    profile.angle_deg[observed],
                    profile.intensity[observed],
                    linestyle="none",
                    marker="o",
                    markersize=3.2,
                    color="#2b6f9f",
                    markeredgewidth=0,
                    label=self._tr("measured"),
                )
                fit = self.fit_result
                if fit is not None:
                    self.profile_axes.plot(
                        profile.angle_deg,
                        fit.model,
                        color="#d55e00",
                        linewidth=1.8,
                        label=self._tr("fit"),
                    )
                    if np.isfinite(fit.baseline):
                        self.profile_axes.axhline(
                            fit.baseline,
                            color="#777777",
                            linewidth=0.9,
                            linestyle=":",
                            label=f"{self._tr('baseline')}: {fit.baseline:.4g}",
                        )
                    self.residual_axes.axhline(0.0, color="#444444", linewidth=0.8)
                    residual_valid = np.isfinite(fit.residual)
                    self.residual_axes.plot(
                        profile.angle_deg[residual_valid],
                        fit.residual[residual_valid],
                        linestyle="none",
                        marker="o",
                        markersize=2.8,
                        color="#7a5195",
                    )
                    if fit.peaks:
                        self.profile_axes.legend(loc="best", frameon=False, prop=font)
                    self.profile_axes.set_title(
                        f"{self._tr('profile_title')} · q = {profile.q_min:.5g}–{profile.q_max:.5g} {_display_q_unit(profile.q_unit)}",
                        loc="left",
                        fontsize=11,
                        fontproperties=title_font,
                    )
                else:
                    self.residual_axes.axhline(0.0, color="#444444", linewidth=0.8)
            else:
                self.profile_axes.text(
                    0.5,
                    0.5,
                    self._tr("no_data"),
                    transform=self.profile_axes.transAxes,
                    ha="center",
                    va="center",
                    color="#687789",
                    fontproperties=font,
                )
                self.residual_axes.axhline(0.0, color="#444444", linewidth=0.8)
            for axes in (self.image_axes, self.profile_axes, self.residual_axes):
                for label in (*axes.get_xticklabels(), *axes.get_yticklabels()):
                    label.set_fontproperties(font)
            if self.canvas.isVisible():
                self.canvas.draw_idle()

        def export_to(self, profile_path: str | Path, *, overwrite: bool = False) -> tuple[Path, Path]:
            if self.profile is None or self.fit_result is None:
                raise ValueError("analyzed azimuthal profile is required for export")
            target = normalize_save_path(profile_path, ".csv")
            peaks_target = target.with_name(f"{target.stem}_peaks.csv")
            if not overwrite and (target.exists() or peaks_target.exists()):
                raise FileExistsError("one or both azimuthal CSV outputs already exist")
            return export_azimuthal_csv_bundle(
                self.profile, self.fit_result, target, overwrite=overwrite
            )

        def _choose_export_path(self) -> None:
            if self.profile is None or self.fit_result is None:
                return
            self._prompt_and_export(
                default_name="azimuthal_profile.csv",
                extra_targets=lambda target: (target.with_name(f"{target.stem}_peaks.csv"),),
                format_done=lambda paths: self._tr("export_done").format(
                    profile=paths[0].name, peaks=paths[1].name
                ),
            )

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
                "q_window": (
                    [self.q_min_spin.value(), self.q_max_spin.value()]
                    if self.data is not None
                    else None
                ),
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
            self.min_separation_spin.setValue(
                float(settings.get("min_separation_deg", self.min_separation_spin.value()))
            )
            self.height_fraction_spin.setValue(
                float(settings.get("min_height_fraction", self.height_fraction_spin.value()))
            )
            self.initial_width_spin.setValue(
                float(settings.get("initial_fwhm_deg", self.initial_width_spin.value()))
            )
            self.eta_spin.setValue(float(settings.get("eta", self.eta_spin.value())))
            q_window = document.get("q_window")
            if (
                self.data is not None
                and _same_q_unit(document.get("q_unit"), self.q_unit)
                and isinstance(q_window, (list, tuple))
                and len(q_window) == 2
            ):
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
