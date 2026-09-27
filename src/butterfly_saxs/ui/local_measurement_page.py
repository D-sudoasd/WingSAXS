"""Qt workbench page for manual 2D point, line-profile and ROI measurements."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..local_measurement import (
    LineProfile,
    PointMeasurement,
    ROIOrientation,
    export_local_measurements_csv,
    export_local_measurements_json,
    extract_line_profile,
    measure_point,
    measure_roi_orientation,
)
from .qt_compat import QT_AVAILABLE, QtCore, QtGui, QtWidgets, require_qt


_TEXT = {
    "title": ("二维局部测量", "2D local measurements"),
    "subtitle": ("在当前二维探测器图像上读取点、线剖面和局部取向。", "Measure points, line profiles and local orientation on the current 2D detector image."),
    "mode": ("选择方式", "Selection"),
    "point": ("点 / 邻域", "Point / patch"),
    "line": ("线剖面", "Line profile"),
    "roi": ("ROI 取向", "ROI orientation"),
    "patch_radius": ("邻域半径 (px)", "Patch radius (px)"),
    "roi_width": ("ROI 半宽 (px)", "ROI half-width (px)"),
    "scale": ("图像显示", "Image display"),
    "linear": ("线性", "Linear"),
    "log": ("对数压缩", "Log compression"),
    "asinh": ("Asinh", "Asinh"),
    "export": ("导出…", "Export…"),
    "clear": ("清除记录", "Clear records"),
    "measurements": ("测量记录", "Measurements"),
    "profile": ("观测线剖面", "Observed line profile"),
    "image": ("二维探测器图像", "2D detector image"),
    "empty": ("请先载入一帧二维数据和 q 标定图。", "Load a 2D frame and its q-coordinate maps."),
    "point_hint": ("单击图像读取点值；邻域统计只使用有效像素。", "Click the image to read a point; patch statistics use valid pixels only."),
    "line_hint": ("依次单击线剖面两端；只读取线上实际探测器像素。", "Click the two line endpoints; only observed detector pixels are sampled."),
    "roi_hint": ("依次单击局部条带两端，计算 q 空间强度加权主轴。", "Click the local stripe endpoints to estimate its intensity-weighted q-space axis."),
    "gap_hint": ("掩膜和非有限数据保留为空隙，不会插值补点。", "Masked and non-finite samples remain gaps; no interpolation is added."),
    "interpretation": ("ROI 主轴描述当前观测强度分布，不代表唯一片层倾角或结构机制。", "The ROI axis describes observed intensity; it is not a unique lamellar tilt or structural mechanism."),
    "no_data": ("当前没有可测量的二维图像。", "No measurable 2D frame is available."),
    "bad_point": ("该像素被掩膜，或强度 / q 坐标无效。", "That pixel is masked or has invalid intensity/q coordinates."),
    "click_start": ("已选第一端点；再选第二端点。", "First endpoint selected; choose the second endpoint."),
    "export_filter": ("CSV 文件 (*.csv);;JSON 文件 (*.json)", "CSV files (*.csv);;JSON files (*.json)"),
    "export_done": ("测量已导出：{path}", "Measurements exported: {path}"),
    "nothing_export": ("没有可导出的测量记录。", "There are no measurement records to export."),
    "point_name": ("点 / 邻域", "Point / patch"),
    "line_name": ("线剖面", "Line profile"),
    "roi_name": ("ROI 取向", "ROI orientation"),
    "source": ("数据帧", "Frame"),
    "result": ("结果", "Result"),
    "signal": ("强度 / 状态", "Intensity / status"),
    "no_record": ("尚无测量。", "No measurements yet."),
    "select_record": ("选择一条记录查看数值和观测证据。", "Select a record to inspect its values and observed evidence."),
    "cancel": ("取消", "Cancel"),
    "clear_confirm": ("删除当前工作区内全部手动测量记录？", "Delete all manual measurement records in this workspace?"),
    "invalid_data": ("数据无效：{reason}", "Invalid data: {reason}"),
    "physical_note": ("L = 2π/|q| 仅在 q 使用 nm^-1 或 A^-1 时给出。", "L = 2π/|q| is reported only for q in nm^-1 or A^-1."),
    "fwhm_note": ("FWHM 是所选剖面的宽度描述；仅在完整观测到两侧半高交点时计算。", "FWHM describes the selected profile and is computed only when both half-height crossings are observed."),
    "source_none": ("当前帧", "Current frame"),
}


def _tr(key: str, language: str) -> str:
    pair = _TEXT[key]
    return pair[1 if str(language).lower().startswith("en") else 0]


def _source_key(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":"))
    return str(value)


def _source_display(value: str | None, language: str) -> str:
    if not value:
        return _tr("source_none", language)
    if value.startswith("__anonymous_frame__:"):
        return "In-memory frame" if language.lower().startswith("en") else "内存帧（仅当前会话）"
    try:
        decoded = json.loads(value)
    except (ValueError, TypeError):
        return value
    if isinstance(decoded, Mapping):
        label = decoded.get("frame_id", decoded.get("frame", decoded.get("path", decoded.get("source"))))
        if label is not None:
            return str(label)
    return value


def _selection_key(selection: Mapping[str, Any]) -> str:
    return json.dumps(selection, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _display_unit(unit: str | None) -> str:
    return {"nm⁻¹": "nm^-1", "Å⁻¹": "A^-1", "Å": "A"}.get(str(unit), str(unit or "unknown"))


def _qt_cjk_font_family() -> str | None:
    if not QT_AVAILABLE:
        return None
    try:
        families = set(QtGui.QFontDatabase.families())
    except Exception:
        families = set()
    for name in ("Microsoft YaHei UI", "Microsoft YaHei", "Noto Sans SC", "SimHei"):
        if name in families:
            return name
    return None


if QT_AVAILABLE:
    import pyqtgraph as pg

    class LocalMeasurementPage(QtWidgets.QWidget):
        """Interactive manual measurements over an already loaded 2D frame.

        ``set_data`` receives the current detector image, qx/qy calibration
        maps and a positive validity mask (``True`` means usable).  Calibration
        and mask construction remain owned by the caller's established data
        path.  Saved documents retain pixel selections and are recomputed when
        matching source frames are supplied again.
        """

        documentChanged = QtCore.Signal(object)

        def __init__(self, parent: Any = None, *, language: str = "zh_CN") -> None:
            super().__init__(parent)
            self.setObjectName("localMeasurementPage")
            self.language = str(language or "zh_CN")
            self._data: np.ndarray | None = None
            self._qx: np.ndarray | None = None
            self._qy: np.ndarray | None = None
            self._valid: np.ndarray | None = None
            self._q_unit = "unknown"
            self._source: str | None = None
            self._source_label = _tr("source_none", self.language)
            self._records: list[PointMeasurement | LineProfile | ROIOrientation] = []
            self._selections: list[dict[str, Any]] = []
            self._pending_selections: list[dict[str, Any]] = []
            self._pending_anchor: tuple[int, int] | None = None
            self._overlay_items: list[Any] = []
            self._active_record_index: int | None = None
            self._build_ui()
            self._apply_plot_fonts()
            self.set_language(self.language)
            self._clear_image(_tr("empty", self.language))

        def _build_ui(self) -> None:
            self.setStyleSheet(
                "#localMeasurementPage { background: #f5f7f8; color: #253340; }"
                "#localMeasurementPage QPushButton, #localMeasurementPage QToolButton { padding: 5px 9px; }"
                "#localMeasurementPage QComboBox, #localMeasurementPage QSpinBox { min-height: 24px; }"
                "#localMeasurementPage QTableWidget { background: white; border: 1px solid #dce2e6; }"
            )
            root = QtWidgets.QVBoxLayout(self)
            root.setContentsMargins(14, 10, 14, 8)
            root.setSpacing(8)
            heading = QtWidgets.QHBoxLayout()
            text_layout = QtWidgets.QVBoxLayout()
            self.title = QtWidgets.QLabel()
            self.title.setStyleSheet("font-size: 22px; font-weight: 600;")
            self.subtitle = QtWidgets.QLabel()
            self.subtitle.setWordWrap(True)
            self.subtitle.setStyleSheet("color: #647381; font-size: 11px;")
            text_layout.addWidget(self.title)
            text_layout.addWidget(self.subtitle)
            heading.addLayout(text_layout, 1)
            self.export_button = QtWidgets.QPushButton()
            self.export_button.setObjectName("localMeasurement_export")
            self.export_button.clicked.connect(self._choose_export)
            heading.addWidget(self.export_button)
            self.clear_button = QtWidgets.QPushButton()
            self.clear_button.setObjectName("localMeasurement_clear")
            self.clear_button.clicked.connect(self._confirm_clear)
            heading.addWidget(self.clear_button)
            root.addLayout(heading)

            controls = QtWidgets.QHBoxLayout()
            self.mode_label = QtWidgets.QLabel()
            self.mode_combo = QtWidgets.QComboBox()
            self.mode_combo.setObjectName("localMeasurement_mode")
            self.mode_combo.addItem("", "point")
            self.mode_combo.addItem("", "line")
            self.mode_combo.addItem("", "roi")
            self.mode_combo.currentIndexChanged.connect(self._mode_changed)
            self.mode_label.setBuddy(self.mode_combo)
            controls.addWidget(self.mode_label)
            controls.addWidget(self.mode_combo)
            self.patch_label = QtWidgets.QLabel()
            self.patch_radius = QtWidgets.QSpinBox()
            self.patch_radius.setObjectName("localMeasurement_patchRadius")
            self.patch_radius.setRange(0, 30)
            self.patch_radius.setValue(1)
            self.patch_label.setBuddy(self.patch_radius)
            controls.addWidget(self.patch_label)
            controls.addWidget(self.patch_radius)
            self.roi_width_label = QtWidgets.QLabel()
            self.roi_half_width = QtWidgets.QDoubleSpinBox()
            self.roi_half_width.setObjectName("localMeasurement_roiHalfWidth")
            self.roi_half_width.setRange(0.5, 100.0)
            self.roi_half_width.setSingleStep(0.5)
            self.roi_half_width.setValue(3.0)
            self.roi_half_width.setDecimals(1)
            self.roi_width_label.setBuddy(self.roi_half_width)
            controls.addWidget(self.roi_width_label)
            controls.addWidget(self.roi_half_width)
            self.scale_label = QtWidgets.QLabel()
            self.scale_combo = QtWidgets.QComboBox()
            self.scale_combo.setObjectName("localMeasurement_scale")
            for mode in ("linear", "log", "asinh"):
                self.scale_combo.addItem("", mode)
            self.scale_combo.currentIndexChanged.connect(self._render_image)
            self.scale_label.setBuddy(self.scale_combo)
            controls.addWidget(self.scale_label)
            controls.addWidget(self.scale_combo)
            controls.addStretch(1)
            root.addLayout(controls)

            self.status = QtWidgets.QLabel()
            self.status.setWordWrap(True)
            self.status.setStyleSheet("color: #8a601f; padding: 2px 0;")
            root.addWidget(self.status)

            splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
            self.image_plot = pg.PlotWidget(background="#161b22")
            self.image_plot.setObjectName("localMeasurementImage")
            self.image_plot.setMinimumSize(360, 260)
            self.image_plot.setLabel("bottom", "Detector column", units="pixel")
            self.image_plot.setLabel("left", "Detector row", units="pixel")
            self.image_plot.getPlotItem().getViewBox().invertY(True)
            self.image_plot.getPlotItem().getViewBox().setAspectLocked(True, ratio=1.0)
            self.image_item = pg.ImageItem(axisOrder="row-major")
            try:
                cmap = pg.colormap.get("CET-L4")
                self.image_item.setLookupTable(cmap.getLookupTable(0.0, 1.0, 256))
            except Exception:  # pragma: no cover - depends on optional colormap registry
                pass
            self.image_plot.addItem(self.image_item)
            self.image_plot.scene().sigMouseClicked.connect(self._on_image_click)
            splitter.addWidget(self.image_plot)

            side = QtWidgets.QWidget()
            side_layout = QtWidgets.QVBoxLayout(side)
            side_layout.setContentsMargins(8, 0, 0, 0)
            self.profile_title = QtWidgets.QLabel()
            self.profile_title.setStyleSheet("font-weight: 600; color: #52616c;")
            side_layout.addWidget(self.profile_title)
            self.profile_plot = pg.PlotWidget(background="#ffffff")
            self.profile_plot.setObjectName("localMeasurementProfile")
            self.profile_plot.setMinimumHeight(190)
            self.profile_plot.showGrid(x=True, y=True, alpha=0.18)
            self.profile_plot.setLabel("bottom", "Path coordinate")
            self.profile_plot.setLabel("left", "Observed intensity")
            self.profile_plot.getPlotItem().getAxis("bottom").enableAutoSIPrefix(False)
            side_layout.addWidget(self.profile_plot, 2)
            self.measurements_title = QtWidgets.QLabel()
            self.measurements_title.setStyleSheet("font-weight: 600; color: #52616c;")
            side_layout.addWidget(self.measurements_title)
            self.table = QtWidgets.QTableWidget(0, 4)
            self.table.setObjectName("localMeasurementTable")
            self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
            self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
            self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
            self.table.horizontalHeader().setStretchLastSection(True)
            self.table.itemSelectionChanged.connect(self._table_selection_changed)
            side_layout.addWidget(self.table, 2)
            self.details = QtWidgets.QLabel()
            self.details.setWordWrap(True)
            self.details.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
            self.details.setStyleSheet("color: #536472; font-size: 11px; padding: 3px 0;")
            side_layout.addWidget(self.details)
            self.boundary_note = QtWidgets.QLabel()
            self.boundary_note.setWordWrap(True)
            self.boundary_note.setStyleSheet("color: #65717c; font-size: 10px; padding: 2px 0;")
            side_layout.addWidget(self.boundary_note)
            splitter.addWidget(side)
            splitter.setStretchFactor(0, 3)
            splitter.setStretchFactor(1, 2)
            splitter.setSizes([650, 420])
            splitter.setChildrenCollapsible(False)
            root.addWidget(splitter, 1)

        def _apply_plot_fonts(self) -> None:
            family = _qt_cjk_font_family()
            if not family:
                return
            font = QtGui.QFont(family)
            self.setFont(font)
            for plot in (self.image_plot, self.profile_plot):
                item = plot.getPlotItem()
                for name in ("bottom", "left"):
                    axis = item.getAxis(name)
                    axis.setStyle(tickFont=font)
                    label = getattr(axis, "label", None)
                    if label is not None and hasattr(label, "setFont"):
                        label.setFont(font)
                title_item = getattr(item.titleLabel, "item", None)
                if title_item is not None and hasattr(title_item, "setFont"):
                    title_item.setFont(font)

        def set_language(self, language: str) -> None:
            self.language = str(language or "zh_CN")
            self.title.setText(_tr("title", self.language))
            self.subtitle.setText(_tr("subtitle", self.language))
            self.export_button.setText(_tr("export", self.language))
            self.clear_button.setText(_tr("clear", self.language))
            self.mode_label.setText(_tr("mode", self.language))
            self.patch_label.setText(_tr("patch_radius", self.language))
            self.roi_width_label.setText(_tr("roi_width", self.language))
            self.scale_label.setText(_tr("scale", self.language))
            for index, key in enumerate(("point", "line", "roi")):
                self.mode_combo.setItemText(index, _tr(key, self.language))
            for index, key in enumerate(("linear", "log", "asinh")):
                self.scale_combo.setItemText(index, _tr(key, self.language))
            self.profile_title.setText(_tr("profile", self.language))
            self.measurements_title.setText(_tr("measurements", self.language))
            self.table.setHorizontalHeaderLabels([
                _tr("mode", self.language), _tr("source", self.language),
                _tr("result", self.language), _tr("signal", self.language),
            ])
            self.image_plot.setTitle(_tr("image", self.language))
            path_label = "路径距离" if not self.language.lower().startswith("en") else "Path distance"
            self.profile_plot.setLabel("bottom", path_label)
            self.profile_plot.setLabel("left", "I(q)" if self.language.lower().startswith("en") else "观测强度")
            self._update_visibility()
            self._refresh_table()
            self._render_image()
            if self._active_record_index is not None and self._active_record_index < len(self._records):
                self._show_record(self._active_record_index)
            elif self._data is None:
                self.status.setText(_tr("empty", self.language))
                self.details.setText(_tr("select_record", self.language))
            else:
                self.status.setText(self._current_hint())

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
            """Display one calibrated 2D frame and apply matching saved picks."""

            try:
                observed = np.asarray(data, dtype=float)
                qx_values = np.asarray(qx, dtype=float)
                qy_values = np.asarray(qy, dtype=float)
                if observed.ndim != 2 or observed.size == 0:
                    raise ValueError("data must be a non-empty two-dimensional array")
                if qx_values.shape != observed.shape or qy_values.shape != observed.shape:
                    raise ValueError("qx and qy must match the detector image shape")
                if valid_mask is None:
                    valid = np.ones(observed.shape, dtype=bool)
                else:
                    valid = np.asarray(valid_mask, dtype=bool).copy()
                    if valid.shape != observed.shape:
                        raise ValueError("valid_mask must match the detector image shape")
                valid &= np.isfinite(observed) & np.isfinite(qx_values) & np.isfinite(qy_values)
            except (TypeError, ValueError) as exc:
                self.invalidate()
                self.status.setText(_tr("invalid_data", self.language).format(reason=exc))
                return
            self._pending_selections.extend(self._selections)
            self._records.clear()
            self._selections.clear()
            self._active_record_index = None
            self._refresh_table()
            self._data = observed
            self._qx = qx_values
            self._qy = qy_values
            self._valid = valid
            self._q_unit = str(q_unit or "unknown")
            self._source = _source_key(source)
            if self._source is None:
                # Anonymous arrays can be reused within one session, but their
                # picks must never be rebound to another same-shaped frame.
                self._source = f"__anonymous_frame__:{id(observed)}"
            self._source_label = _source_display(self._source, self.language)
            self._pending_anchor = None
            self.status.setText("")
            self._render_image()
            self._refresh_table()
            self._apply_pending_selections()
            self._draw_measurement_overlays()

        def invalidate(self, reason: str | None = None) -> None:
            """Clear the displayed frame while keeping saved manual selections."""

            self._pending_selections.extend(self._selections)
            self._records.clear()
            self._selections.clear()
            self._active_record_index = None
            self._refresh_table()
            self._data = self._qx = self._qy = self._valid = None
            self._pending_anchor = None
            self._clear_image(reason or _tr("no_data", self.language))
            self._draw_measurement_overlays()

        def document(self) -> dict[str, Any]:
            """Persist user selections and settings; numerical results are recomputed on restore."""

            return {
                "schema": "butterfly-saxs/local-measurement-selections-v1",
                "settings": {
                    "patch_radius": self.patch_radius.value(),
                    "roi_half_width_pixels": self.roi_half_width.value(),
                    "mode": str(self.mode_combo.currentData() or "point"),
                },
                "selections": [dict(selection) for selection in self._selections],
                "pending_selections": [dict(selection) for selection in self._pending_selections],
            }

        def restore_document(self, document: Mapping[str, Any]) -> None:
            """Restore pixel picks, then recompute any that match the loaded frame."""

            if not isinstance(document, Mapping):
                raise ValueError("local measurement document must be a mapping")
            settings = document.get("settings", {})
            if isinstance(settings, Mapping):
                try:
                    self.patch_radius.setValue(int(settings.get("patch_radius", 1)))
                    self.roi_half_width.setValue(float(settings.get("roi_half_width_pixels", 3.0)))
                except (TypeError, ValueError):
                    pass
                mode = str(settings.get("mode", "point"))
                index = self.mode_combo.findData(mode)
                if index >= 0:
                    self.mode_combo.setCurrentIndex(index)
            selections = document.get("selections", [])
            pending = document.get("pending_selections", [])
            if not isinstance(selections, list) or not isinstance(pending, list):
                raise ValueError("local measurement selections must be lists")
            self._records.clear()
            self._selections.clear()
            self._pending_selections = [dict(item) for item in [*selections, *pending] if isinstance(item, Mapping)]
            self._refresh_table()
            self._apply_pending_selections()
            self.documentChanged.emit(self.document())

        @property
        def measurements(self) -> tuple[Any, ...]:
            return tuple(self._records)

        @property
        def image_data(self) -> np.ndarray | None:
            return self._data

        def _mode_changed(self, *_: Any) -> None:
            self._pending_anchor = None
            self._update_visibility()
            self.status.setText(self._current_hint())

        def _update_visibility(self) -> None:
            is_point = self.mode_combo.currentData() == "point"
            is_roi = self.mode_combo.currentData() == "roi"
            self.patch_label.setVisible(is_point)
            self.patch_radius.setVisible(is_point)
            self.roi_width_label.setVisible(is_roi)
            self.roi_half_width.setVisible(is_roi)

        def _current_hint(self) -> str:
            mode = str(self.mode_combo.currentData() or "point")
            key = {"point": "point_hint", "line": "line_hint", "roi": "roi_hint"}[mode]
            return _tr(key, self.language)

        def _clear_image(self, message: str) -> None:
            self.image_item.clear()
            self.profile_plot.clear()
            self.image_plot.setTitle(_tr("image", self.language))
            self.status.setText(message)
            self.details.setText(_tr("select_record", self.language))

        def _render_image(self, *_: Any) -> None:
            if self._data is None or self._valid is None:
                return
            values = self._data[self._valid]
            if values.size == 0:
                self._clear_image(_tr("no_data", self.language))
                return
            display = self._data.astype(float, copy=True)
            scale = str(self.scale_combo.currentData() or "linear")
            if scale == "log":
                display = np.sign(display) * np.log1p(np.abs(display))
            elif scale == "asinh":
                scale_value = float(np.median(np.abs(values))) or 1.0
                display = np.arcsinh(display / scale_value)
            display[~self._valid] = np.nan
            finite = display[self._valid & np.isfinite(display)]
            low, high = np.percentile(finite, (1.0, 99.5))
            if not np.isfinite(low) or not np.isfinite(high) or high <= low:
                low, high = float(np.min(finite)), float(np.max(finite))
                if high <= low:
                    high = low + 1.0
            self.image_item.setImage(display, autoLevels=False, levels=(float(low), float(high)))
            self.image_plot.setRange(xRange=(-0.5, self._data.shape[1] - 0.5),
                                     yRange=(-0.5, self._data.shape[0] - 0.5), padding=0.02)
            self.status.setText(self._current_hint())
            self._draw_measurement_overlays()

        def _on_image_click(self, event: Any) -> None:
            if self._data is None or self._valid is None:
                return
            if event.button() != QtCore.Qt.MouseButton.LeftButton:
                return
            point = self.image_plot.getPlotItem().getViewBox().mapSceneToView(event.scenePos())
            col, row = int(round(point.x())), int(round(point.y()))
            if row < 0 or col < 0 or row >= self._data.shape[0] or col >= self._data.shape[1]:
                return
            if self.mode_combo.currentData() == "point":
                self._add_point(row, col)
            elif self._pending_anchor is None:
                self._pending_anchor = (col, row)
                self.status.setText(_tr("click_start", self.language))
                self._draw_measurement_overlays()
            else:
                start = self._pending_anchor
                self._pending_anchor = None
                if self.mode_combo.currentData() == "line":
                    self._add_line(start, (col, row))
                else:
                    self._add_roi(start, (col, row))

        def _add_point(self, row: int, col: int) -> None:
            try:
                record = measure_point(
                    self._data, self._qx, self._qy, row, col, valid_mask=self._valid,
                    q_unit=self._q_unit, patch_radius=self.patch_radius.value(), source=self._source,
                )
            except ValueError:
                self.status.setText(_tr("bad_point", self.language))
                return
            selection = {
                "kind": "point", "row": row, "col": col,
                "patch_radius": self.patch_radius.value(), "source": self._source,
                "shape": list(self._data.shape),
            }
            self._append_record(record, selection)

        def _add_line(self, start: tuple[int, int], end: tuple[int, int]) -> None:
            try:
                record = extract_line_profile(
                    self._data, self._qx, self._qy, start, end, valid_mask=self._valid,
                    q_unit=self._q_unit, source=self._source,
                )
            except ValueError as exc:
                self.status.setText(_tr("invalid_data", self.language).format(reason=exc))
                return
            selection = {
                "kind": "line", "start": list(start), "end": list(end),
                "source": self._source, "shape": list(self._data.shape),
            }
            self._append_record(record, selection)

        def _add_roi(self, start: tuple[int, int], end: tuple[int, int]) -> None:
            try:
                record = measure_roi_orientation(
                    self._data, self._qx, self._qy, start, end, valid_mask=self._valid,
                    q_unit=self._q_unit, half_width_pixels=self.roi_half_width.value(),
                    source=self._source,
                )
            except ValueError as exc:
                self.status.setText(_tr("invalid_data", self.language).format(reason=exc))
                return
            selection = {
                "kind": "roi_orientation", "start": list(start), "end": list(end),
                "half_width_pixels": self.roi_half_width.value(), "source": self._source,
                "shape": list(self._data.shape),
            }
            self._append_record(record, selection)

        def _append_record(self, record: Any, selection: dict[str, Any]) -> None:
            self._records.append(record)
            self._selections.append(selection)
            self._active_record_index = len(self._records) - 1
            self._refresh_table(select_index=self._active_record_index)
            self._show_record(self._active_record_index)
            self._draw_measurement_overlays()
            self.documentChanged.emit(self.document())

        def _apply_pending_selections(self) -> None:
            if self._data is None:
                return
            keep: list[dict[str, Any]] = []
            for selection in self._pending_selections:
                if selection.get("source") != self._source:
                    keep.append(selection)
                    continue
                if selection.get("shape") != list(self._data.shape):
                    keep.append(selection)
                    continue
                if any(_selection_key(item) == _selection_key(selection) for item in self._selections):
                    continue
                kind = selection.get("kind")
                try:
                    if kind == "point":
                        record = measure_point(
                            self._data, self._qx, self._qy, selection["row"], selection["col"],
                            valid_mask=self._valid, q_unit=self._q_unit,
                            patch_radius=selection.get("patch_radius", 1), source=self._source,
                        )
                    elif kind == "line":
                        record = extract_line_profile(
                            self._data, self._qx, self._qy, tuple(selection["start"]), tuple(selection["end"]),
                            valid_mask=self._valid, q_unit=self._q_unit, source=self._source,
                        )
                    elif kind == "roi_orientation":
                        record = measure_roi_orientation(
                            self._data, self._qx, self._qy, tuple(selection["start"]), tuple(selection["end"]),
                            valid_mask=self._valid, q_unit=self._q_unit,
                            half_width_pixels=selection.get("half_width_pixels", 3.0), source=self._source,
                        )
                    else:
                        continue
                except (KeyError, TypeError, ValueError):
                    keep.append(selection)
                    continue
                self._records.append(record)
                self._selections.append(dict(selection))
            self._pending_selections = keep
            if keep and any(selection.get("source") == self._source for selection in keep):
                self.status.setText(
                    "Some saved pixel selections are unavailable under the current mask or calibration."
                    if self.language.lower().startswith("en")
                    else "部分已保存的像素选择在当前掩膜或标定下不可用，记录仍已保留。"
                )
            if self._records:
                self._active_record_index = len(self._records) - 1
            self._refresh_table(select_index=self._active_record_index)
            if self._active_record_index is not None:
                self._show_record(self._active_record_index)

        def _refresh_table(self, *, select_index: int | None = None) -> None:
            if not hasattr(self, "table"):
                return
            self.table.blockSignals(True)
            self.table.setRowCount(len(self._records))
            for index, record in enumerate(self._records):
                if isinstance(record, PointMeasurement):
                    kind, result, signal = _tr("point_name", self.language), f"q={record.q:.5g} {record.q_unit}", f"I={record.intensity:.5g}"
                elif isinstance(record, LineProfile):
                    kind = _tr("line_name", self.language)
                    result = (f"FWHM={record.fwhm.value:.5g} {_display_unit(record.fwhm.x_unit)}"
                              if record.fwhm.value is not None else f"FWHM: {record.fwhm.reason}")
                    signal = f"{int(np.count_nonzero(record.valid))}/{record.valid.size}"
                else:
                    kind = _tr("roi_name", self.language)
                    result = f"{record.orientation_deg:.4g}°" if record.orientation_deg is not None else str(record.reason)
                    signal = f"n={record.n_pixels}; {record.status}"
                source = record.source if hasattr(record, "source") else None
                values = (kind, _source_display(source, self.language), result, signal)
                for column, value in enumerate(values):
                    self.table.setItem(index, column, QtWidgets.QTableWidgetItem(str(value)))
            if select_index is not None and 0 <= select_index < len(self._records):
                self.table.selectRow(select_index)
            self.table.blockSignals(False)

        def _table_selection_changed(self) -> None:
            rows = self.table.selectionModel().selectedRows()
            if rows:
                self._active_record_index = rows[0].row()
                self._show_record(self._active_record_index)

        def _show_record(self, index: int) -> None:
            if not 0 <= index < len(self._records):
                return
            record = self._records[index]
            if isinstance(record, LineProfile):
                self._render_profile(record)
                width = record.fwhm.value
                if width is None:
                    fwhm_text = f"FWHM unavailable: {record.fwhm.reason}."
                else:
                    fwhm_text = f"FWHM = {width:.6g} {_display_unit(record.fwhm.x_unit)}; status: {record.fwhm.status}."
                    if record.fwhm.length_proxy is not None:
                        fwhm_text += f" 2π/FWHM = {record.fwhm.length_proxy:.6g} {_display_unit(record.fwhm.length_unit)} (profile-based proxy)."
                self.details.setText(
                    f"{fwhm_text}\n{_tr('fwhm_note', self.language)}\n"
                    f"Observed samples: {int(np.count_nonzero(record.valid))}/{record.valid.size}; "
                    f"q unit: {_display_unit(record.q_unit)}; profile axis: {_display_unit(record.axis_unit)}."
                )
            elif isinstance(record, PointMeasurement):
                self.profile_plot.clear()
                spacing = (f"L = 2π/|q| = {record.spacing:.6g} {_display_unit(record.spacing_unit)}"
                           if record.spacing is not None else "L = 2π/|q| unavailable for this q unit or q=0.")
                self.details.setText(
                    f"Pixel (column, row) = ({record.col}, {record.row}); I = {record.intensity:.6g}.\n"
                    f"qx = {record.qx:.6g}, qy = {record.qy:.6g}, |q| = {record.q:.6g} {_display_unit(record.q_unit)}.\n"
                    f"{spacing}\nPatch: n={record.patch_count}, mean={record.patch_mean:.6g}, "
                    f"median={record.patch_median:.6g}, SD={record.patch_std:.6g}.\n{_tr('physical_note', self.language)}"
                )
            else:
                self.profile_plot.clear()
                axis = "unavailable" if record.orientation_deg is None else f"{record.orientation_deg:.4f}° from +qx"
                eigen = record.eigenvalue_ratio
                ratio_text = "unavailable" if eigen is None else f"{eigen:.4g}"
                self.details.setText(
                    f"Observed q-space principal axis: {axis}.\n"
                    f"Pixels: {record.n_pixels}; eigenvalue ratio: {ratio_text}; status: {record.status}.\n"
                    f"Weights: {record.weighting}.\n{record.interpretation}"
                )
            self._draw_measurement_overlays()

        def _render_profile(self, record: LineProfile) -> None:
            self.profile_plot.clear()
            distance = "路径距离" if not self.language.lower().startswith("en") else "Path distance"
            self.profile_plot.getPlotItem().getAxis("bottom").enableAutoSIPrefix(False)
            self.profile_plot.setLabel("bottom", f"{distance} ({_display_unit(record.axis_unit)})")
            intensity = "观测强度" if not self.language.lower().startswith("en") else "Observed intensity"
            self.profile_plot.setLabel("left", intensity)
            self.profile_plot.plot(
                record.axis, record.intensity, pen=pg.mkPen("#246b8e", width=2),
                symbol="o", symbolSize=4, symbolBrush="#e28743", symbolPen=None,
                connect="finite",
            )
            fit = record.fwhm
            if fit.half_height is not None:
                self.profile_plot.addItem(pg.InfiniteLine(pos=fit.half_height, angle=0,
                                                         pen=pg.mkPen("#a65e20", style=QtCore.Qt.PenStyle.DashLine)))
            for crossing in (fit.x_left, fit.x_right):
                if crossing is not None:
                    self.profile_plot.addItem(pg.InfiniteLine(pos=crossing, angle=90,
                                                             pen=pg.mkPen("#a65e20", style=QtCore.Qt.PenStyle.DashLine)))
            self.profile_plot.setTitle(f"Observed samples: {int(np.count_nonzero(record.valid))}/{record.valid.size}")

        def _draw_measurement_overlays(self) -> None:
            for item in self._overlay_items:
                try:
                    self.image_plot.removeItem(item)
                except Exception:
                    pass
            self._overlay_items.clear()
            if self._data is None:
                return
            if self._pending_anchor is not None:
                marker = pg.ScatterPlotItem([self._pending_anchor[0]], [self._pending_anchor[1]],
                                            symbol="o", size=11, pen=pg.mkPen("#f4c95d", width=2), brush=None)
                self.image_plot.addItem(marker)
                self._overlay_items.append(marker)
            for record in self._records:
                if record.source != self._source:
                    continue
                if isinstance(record, PointMeasurement):
                    overlay = pg.ScatterPlotItem([record.col], [record.row], symbol="+", size=13,
                                                 pen=pg.mkPen("#ffca6a", width=2))
                elif isinstance(record, LineProfile):
                    overlay = pg.PlotDataItem([record.cols[0], record.cols[-1]],
                                              [record.rows[0], record.rows[-1]],
                                              pen=pg.mkPen("#ffca6a", width=1.5))
                else:
                    (x0, y0), (x1, y1) = record.start, record.end
                    overlay = pg.PlotDataItem([x0, x1], [y0, y1], pen=pg.mkPen("#6fe3c1", width=2))
                self.image_plot.addItem(overlay)
                self._overlay_items.append(overlay)

        def clear_records(self) -> None:
            if not self._records and not self._pending_selections:
                return
            self._records.clear()
            self._selections.clear()
            self._pending_selections.clear()
            self._active_record_index = None
            self._refresh_table()
            self.profile_plot.clear()
            self.details.setText(_tr("no_record", self.language))
            self._draw_measurement_overlays()
            self.documentChanged.emit(self.document())

        def _confirm_clear(self) -> None:
            answer = QtWidgets.QMessageBox.question(
                self,
                _tr("clear", self.language),
                _tr("clear_confirm", self.language),
                QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
                QtWidgets.QMessageBox.StandardButton.No,
            )
            if answer == QtWidgets.QMessageBox.StandardButton.Yes:
                self.clear_records()

        def _choose_export(self) -> None:
            if not self._records:
                self.status.setText(_tr("nothing_export", self.language))
                return
            title = _tr("export", self.language)
            path, selected_filter = QtWidgets.QFileDialog.getSaveFileName(
                self, title, "local_measurements.csv", _tr("export_filter", self.language),
            )
            if not path:
                return
            if path.lower().endswith(".json") or "JSON" in selected_filter.upper():
                target = Path(path)
                if target.suffix.lower() != ".json":
                    target = target.with_suffix(".json")
                export_local_measurements_json(target, self._records, overwrite=True)
            else:
                target = Path(path)
                if target.suffix.lower() != ".csv":
                    target = target.with_suffix(".csv")
                export_local_measurements_csv(target, self._records, overwrite=True)
            self.status.setText(_tr("export_done", self.language).format(path=target))

        def export_to(self, path: str | Path, *, overwrite: bool = False) -> Path:
            """Programmatic export helper used by tests and project integrations."""

            target = Path(path)
            if target.suffix.lower() == ".json":
                return export_local_measurements_json(target, self._records, overwrite=overwrite)
            return export_local_measurements_csv(target, self._records, overwrite=overwrite)

else:
    class LocalMeasurementPage:  # pragma: no cover - exercised on minimal installations
        def __init__(self, *_: Any, **__: Any) -> None:
            require_qt()


__all__ = ["LocalMeasurementPage"]
