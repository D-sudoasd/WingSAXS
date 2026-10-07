"""Standalone viewer for the illustrative lamellar projection FFT."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..lamellar_scattering import (
    DEFAULT_GRID_SHAPE,
    LamellarScatteringResult,
    export_lamellar_scattering,
    render_lamellar_scattering,
    simulate_projected_density_fft,
)
from .qt_compat import QT_AVAILABLE, QtCore, QtGui, QtWidgets


_TEXT = {
    "title": ("片层投影与二维 FFT", "Lamellar projection and 2D FFT"),
    "scope": (
        "示意性正向计算：将当前有限片层沿 z 方向投影，再计算二维 FFT。结果不是实测散射强度，也不代表唯一的三维重建。",
        "Illustrative forward calculation: project the current finite slabs along z, then compute a 2D FFT. This is not measured scattering intensity or a unique 3D reconstruction.",
    ),
    "rows": ("行数", "Rows"),
    "columns": ("列数", "Columns"),
    "margin": ("边缘真空比例", "Vacuum margin"),
    "run": ("重新计算", "Recalculate"),
    "export": ("导出结果", "Export results"),
    "close": ("关闭", "Close"),
    "ready": ("二维投影与 FFT 已计算。", "2D projection and FFT calculated."),
    "failed": ("计算失败", "Calculation failed"),
    "saved": ("结果已保存：\n{paths}", "Results saved:\n{paths}"),
    "save_title": ("导出投影 FFT", "Export projection FFT"),
    "save_filter": ("压缩数组 (*.npz)", "Compressed arrays (*.npz)"),
    "notice_title": ("投影 FFT 结果", "Projection FFT results"),
    "canvas_hint": (
        "左：投影密度；右：归一化 FFT 功率（对数色标）",
        "Left: projected density; right: normalized FFT power (log color scale)",
    ),
}


def _t(key: str, language: str) -> str:
    return _TEXT[key][0 if str(language).lower().startswith("zh") else 1]


if QT_AVAILABLE:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.figure import Figure

    class LamellarScatteringDialog(QtWidgets.QDialog):
        """Display and export one scene's conditional 2D projection FFT."""

        def __init__(
            self,
            scene: Any,
            parent: Any = None,
            *,
            language: str = "zh_CN",
        ) -> None:
            super().__init__(parent)
            self.setObjectName("lamellarScatteringDialog")
            self.setWindowTitle(_t("title", language))
            self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, False)
            self._scene = scene
            self._language = language
            self._result: LamellarScatteringResult | None = None
            self.figure = Figure(figsize=(11.2, 5.0), facecolor="#fbfcfd")
            self.canvas = FigureCanvasQTAgg(self.figure)
            self.canvas.setMinimumSize(600, 320)
            self._build_ui()
            screen = QtGui.QGuiApplication.primaryScreen()
            available = screen.availableGeometry() if screen else QtCore.QRect(0, 0, 1366, 900)
            self.resize(min(1180, max(760, available.width() - 40)), min(760, max(540, available.height() - 60)))
            self._calculate()

            from .help import apply_help

            apply_help(self, self._language)

        def _build_ui(self) -> None:
            self.setStyleSheet(
                "#lamellarScatteringDialog { background: #f5f7f8; color: #253744; }"
                "#lamellarScatteringDialog QPushButton { padding: 6px 12px; }"
                "#lamellarScatteringDialog QSpinBox, #lamellarScatteringDialog QDoubleSpinBox { min-height: 25px; }"
            )
            outer = QtWidgets.QVBoxLayout(self)
            outer.setContentsMargins(16, 12, 16, 12)
            heading = QtWidgets.QLabel(_t("scope", self._language))
            heading.setWordWrap(True)
            heading.setStyleSheet(
                "background: #e8eef1; color: #354b57; border-left: 3px solid #668b9b; "
                "padding: 9px 11px; font-size: 12px;"
            )
            outer.addWidget(heading)

            settings = QtWidgets.QHBoxLayout()
            self.rows = QtWidgets.QSpinBox()
            self.rows.setObjectName("lamellarScatteringRows")
            self.rows.setRange(16, 2048)
            self.rows.setValue(DEFAULT_GRID_SHAPE[0])
            self.rows.setAccessibleName(_t("rows", self._language))
            self.columns = QtWidgets.QSpinBox()
            self.columns.setObjectName("lamellarScatteringColumns")
            self.columns.setRange(16, 2048)
            self.columns.setValue(DEFAULT_GRID_SHAPE[1])
            self.columns.setAccessibleName(_t("columns", self._language))
            self.margin = QtWidgets.QDoubleSpinBox()
            self.margin.setObjectName("lamellarScatteringMargin")
            self.margin.setRange(0.0, 1.0)
            self.margin.setDecimals(2)
            self.margin.setSingleStep(0.05)
            self.margin.setValue(0.1)
            self.margin.setSuffix(" × span")
            self.margin.setAccessibleName(_t("margin", self._language))
            for label, widget, key in (
                ("rowsLabel", self.rows, "rows"),
                ("columnsLabel", self.columns, "columns"),
                ("marginLabel", self.margin, "margin"),
            ):
                settings.addWidget(QtWidgets.QLabel(_t(key, self._language)))
                settings.addWidget(widget)
            self.calculate_button = QtWidgets.QPushButton(_t("run", self._language))
            self.calculate_button.setObjectName("lamellarScatteringCalculate")
            self.calculate_button.clicked.connect(self._calculate)
            settings.addWidget(self.calculate_button)
            settings.addStretch(1)
            outer.addLayout(settings)

            self.canvas_hint = QtWidgets.QLabel(_t("canvas_hint", self._language))
            self.canvas_hint.setStyleSheet("color: #596b75; padding: 3px 0;")
            outer.addWidget(self.canvas_hint)
            outer.addWidget(self.canvas, 1)
            self.status = QtWidgets.QLabel()
            self.status.setObjectName("lamellarScatteringStatus")
            self.status.setWordWrap(True)
            self.status.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
            self.status.setStyleSheet("color: #80612d; padding: 4px 0;")
            outer.addWidget(self.status)

            buttons = QtWidgets.QHBoxLayout()
            buttons.addStretch(1)
            self.export_button = QtWidgets.QPushButton(_t("export", self._language))
            self.export_button.setObjectName("lamellarScatteringExport")
            self.export_button.setEnabled(False)
            self.export_button.clicked.connect(self._export)
            buttons.addWidget(self.export_button)
            close_button = QtWidgets.QPushButton(_t("close", self._language))
            close_button.setObjectName("lamellarScatteringClose")
            close_button.clicked.connect(self.accept)
            buttons.addWidget(close_button)
            outer.addLayout(buttons)

        def _calculate(self) -> None:
            self.calculate_button.setEnabled(False)
            try:
                self._result = simulate_projected_density_fft(
                    self._scene,
                    shape=(self.rows.value(), self.columns.value()),
                    margin_fraction=self.margin.value(),
                )
                render_lamellar_scattering(
                    self._result, figure=self.figure, language=self._language
                )
                self.canvas.draw_idle()
                self.export_button.setEnabled(True)
                diagnostics = self._result.diagnostics
                warning_lines = diagnostics.get("warnings", [])
                resolution = " × ".join(
                    f"{value:.3g}" for value in diagnostics["q_resolution_xy"]
                )
                nyquist = " × ".join(
                    f"{value:.3g}" for value in diagnostics["nyquist_q_xy"]
                )
                unit = self._result.q_unit
                detail = (
                    f"Δq = {resolution} {unit}; q Nyquist = {nyquist} {unit}. "
                    f"Minimum slab dimension = "
                    f"{diagnostics['thinnest_slab_dimension_pixels']:.2f} px."
                )
                if warning_lines:
                    detail += "\n" + "\n".join(warning_lines)
                    self.status.setStyleSheet("color: #805722; padding: 4px 0;")
                else:
                    self.status.setStyleSheet("color: #456c5c; padding: 4px 0;")
                self.status.setText(_t("ready", self._language) + " " + detail)
            except Exception as exc:
                self._result = None
                self.export_button.setEnabled(False)
                self.status.setStyleSheet("color: #8b4040; padding: 4px 0;")
                self.status.setText(f"{_t('failed', self._language)}: {exc}")
            finally:
                self.calculate_button.setEnabled(True)

        def _export(self) -> None:
            if self._result is None:
                return
            chosen, _ = QtWidgets.QFileDialog.getSaveFileName(
                self,
                _t("save_title", self._language),
                str(Path.cwd() / "lamellar-projection.npz"),
                _t("save_filter", self._language),
            )
            if not chosen:
                return
            try:
                paths = export_lamellar_scattering(
                    self._result,
                    chosen,
                    overwrite=False,
                    language=self._language,
                )
            except (OSError, ValueError, TypeError) as exc:
                QtWidgets.QMessageBox.warning(
                    self,
                    _t("notice_title", self._language),
                    str(exc),
                )
                return
            QtWidgets.QMessageBox.information(
                self,
                _t("notice_title", self._language),
                _t("saved", self._language).format(
                    paths="\n".join(str(path) for path in paths)
                ),
            )

else:

    class LamellarScatteringDialog:  # pragma: no cover - depends on optional GUI runtime
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("LamellarScatteringDialog requires the desktop Qt runtime")


__all__ = ("LamellarScatteringDialog",)
