"""Read-only viewer for SAXSAnalyzer 2D sessions and result evidence."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..legacy_saxs import (
    LegacySaxsError,
    LegacySaxsInspection,
    LegacySaxsLoadedImage,
    inspect_legacy_saxs,
    load_legacy_saxs_image,
    write_compatibility_receipt,
)
from .qt_compat import QT_AVAILABLE, QtCore, QtGui, QtWidgets, require_qt


_TEXT = {
    "en": {
        "title": "SAXSAnalyzer 2D compatibility viewer",
        "intro": "Inspect archived 2D session metadata and result evidence. Source files remain read-only; legacy measurements are preserved as reported and are not converted into WingSAXS observables.",
        "open_file": "Open evidence file…",
        "open_folder": "Open result folder…",
        "load_image": "Load session image",
        "export": "Export compatibility receipt…",
        "close": "Close",
        "empty": "Choose a SAXSAnalyzer session, evidence file, or result folder to inspect.",
        "pick_file_title": "Open legacy SAXS evidence",
        "pick_folder_title": "Open legacy result folder",
        "save_title": "Export compatibility receipt",
        "file_filter": "Legacy evidence (*.json *.npz *.npy *.csv *.txt *.dat);;All files (*)",
        "receipt_filter": "JSON receipt (*.json)",
        "idle": "No legacy source selected.",
        "ready": "Inspection ready. Original source files were not changed.",
        "loaded": "Loaded the legacy 2D image and checked scalar q map. No old measurement was converted.",
        "receipt_written": "Compatibility receipt written: {path}",
        "error": "Could not complete the operation: {error}",
        "overwrite_prompt": "A receipt already exists at this path. Replace it?",
        "overwrite_title": "Replace compatibility receipt",
        "missing_geometry": "Image loading requires a supported Fusion v2 session with complete scalar geometry and an available 2D image.",
        "preview_name": "Legacy evidence preview",
        "status_name": "Compatibility status",
    },
    "zh_CN": {
        "title": "SAXSAnalyzer 二维兼容性查看器",
        "intro": "查看归档的二维会话元数据和结果证据。源文件保持只读；旧测量值按原报告保留，不转换为 WingSAXS 观测量。",
        "open_file": "打开证据文件…",
        "open_folder": "打开结果文件夹…",
        "load_image": "载入会话中的图像",
        "export": "导出兼容性凭据…",
        "close": "关闭",
        "empty": "选择 SAXSAnalyzer 会话、证据文件或结果文件夹进行查看。",
        "pick_file_title": "打开旧版 SAXS 证据",
        "pick_folder_title": "打开旧版结果文件夹",
        "save_title": "导出兼容性凭据",
        "file_filter": "旧版证据 (*.json *.npz *.npy *.csv *.txt *.dat);;所有文件 (*)",
        "receipt_filter": "JSON 凭据 (*.json)",
        "idle": "尚未选择旧版文件。",
        "ready": "已完成检查。原始源文件未更改。",
        "loaded": "已载入旧版二维图像和经核验的标量 q 图。未转换任何旧测量值。",
        "receipt_written": "兼容性凭据已写入：{path}",
        "error": "操作未能完成：{error}",
        "overwrite_prompt": "此路径已有兼容性凭据。是否替换？",
        "overwrite_title": "替换兼容性凭据",
        "missing_geometry": "载入图像需要受支持的 Fusion v2 会话、完整标量几何参数和可用的二维图像。",
        "preview_name": "旧版证据预览",
        "status_name": "兼容性状态",
    },
}


if QT_AVAILABLE:

    class LegacySaxsDialog(QtWidgets.QDialog):
        """Inspect old 2D evidence, optionally load a v2 image, and export a receipt.

        ``imageLoaded`` emits a :class:`LegacySaxsLoadedImage`. The parent
        workbench decides how to present that image and q map; this dialog does
        not insert legacy measurements into the current project or results.
        """

        imageLoaded = QtCore.Signal(object)
        inspectionChanged = QtCore.Signal(object)

        def __init__(
            self,
            parent: QtWidgets.QWidget | None = None,
            *,
            language: str = "zh_CN",
        ) -> None:
            super().__init__(parent)
            self.setObjectName("legacySaxsDialog")
            self.setWindowModality(QtCore.Qt.WindowModality.NonModal)
            self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, False)
            self.setSizeGripEnabled(True)
            self._language = "en" if str(language).lower().startswith("en") else "zh_CN"
            self._inspection: LegacySaxsInspection | None = None
            self._status_key = "idle"
            self._status_values: dict[str, Any] = {}
            self._build_ui()
            self._retranslate_ui()
            self.resize(860, 620)

        @property
        def inspection(self) -> LegacySaxsInspection | None:
            return self._inspection

        @property
        def can_load_image(self) -> bool:
            session = self._inspection.session if self._inspection else None
            return bool(
                session
                and session.geometry_supported
                and session.image_present
                and session.data_path is not None
            )

        def _tr(self, key: str) -> str:
            return _TEXT[self._language][key]

        def _build_ui(self) -> None:
            layout = QtWidgets.QVBoxLayout(self)
            layout.setContentsMargins(18, 16, 18, 14)
            layout.setSpacing(10)

            self.intro_label = QtWidgets.QLabel(self)
            self.intro_label.setWordWrap(True)
            self.intro_label.setObjectName("legacySaxsIntro")
            layout.addWidget(self.intro_label)

            actions = QtWidgets.QHBoxLayout()
            self.open_file_button = QtWidgets.QPushButton(self)
            self.open_file_button.setObjectName("legacyOpenFileButton")
            self.open_file_button.clicked.connect(self._choose_file)
            actions.addWidget(self.open_file_button)
            self.open_folder_button = QtWidgets.QPushButton(self)
            self.open_folder_button.setObjectName("legacyOpenFolderButton")
            self.open_folder_button.clicked.connect(self._choose_folder)
            actions.addWidget(self.open_folder_button)
            actions.addStretch(1)
            layout.addLayout(actions)

            self.preview = QtWidgets.QPlainTextEdit(self)
            self.preview.setObjectName("legacySaxsPreview")
            self.preview.setReadOnly(True)
            self.preview.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
            self.preview.setAccessibleName(self._tr("preview_name"))
            self.preview.setPlainText(self._tr("empty"))
            layout.addWidget(self.preview, 1)

            self.status_label = QtWidgets.QLabel(self)
            self.status_label.setObjectName("legacySaxsStatus")
            self.status_label.setWordWrap(True)
            self.status_label.setAccessibleName(self._tr("status_name"))
            layout.addWidget(self.status_label)

            footer = QtWidgets.QHBoxLayout()
            footer.addStretch(1)
            self.load_image_button = QtWidgets.QPushButton(self)
            self.load_image_button.setObjectName("legacyLoadImageButton")
            self.load_image_button.clicked.connect(self._load_from_button)
            footer.addWidget(self.load_image_button)
            self.export_button = QtWidgets.QPushButton(self)
            self.export_button.setObjectName("legacyExportReceiptButton")
            self.export_button.clicked.connect(self._choose_receipt_target)
            footer.addWidget(self.export_button)
            self.close_button = QtWidgets.QPushButton(self)
            self.close_button.setObjectName("legacyCloseButton")
            self.close_button.clicked.connect(self.close)
            footer.addWidget(self.close_button)
            layout.addLayout(footer)

        def _retranslate_ui(self) -> None:
            self.setWindowTitle(self._tr("title"))
            self.intro_label.setText(self._tr("intro"))
            self.open_file_button.setText(self._tr("open_file"))
            self.open_folder_button.setText(self._tr("open_folder"))
            self.load_image_button.setText(self._tr("load_image"))
            self.export_button.setText(self._tr("export"))
            self.close_button.setText(self._tr("close"))
            self.preview.setAccessibleName(self._tr("preview_name"))
            self.status_label.setAccessibleName(self._tr("status_name"))
            if self._inspection is None:
                self.preview.setPlainText(self._tr("empty"))
            else:
                self._refresh_preview()
            self.status_label.setText(
                self._tr(self._status_key).format(**self._status_values)
            )
            self.load_image_button.setEnabled(self.can_load_image)
            self.export_button.setEnabled(self._inspection is not None)

            from .help import apply_help

            apply_help(self, self._language)

        def set_language(self, language: str) -> None:
            self._language = "en" if str(language).lower().startswith("en") else "zh_CN"
            self._retranslate_ui()

        def _set_status(self, key: str, **values: Any) -> None:
            self._status_key = key
            self._status_values = dict(values)
            self.status_label.setText(self._tr(key).format(**values))

        def set_source(self, path: str | Path) -> LegacySaxsInspection:
            """Inspect a user-selected file or folder without changing it."""

            self._inspection = None
            self.load_image_button.setEnabled(False)
            self.export_button.setEnabled(False)
            try:
                inspection = inspect_legacy_saxs(path)
            except (LegacySaxsError, OSError, ValueError) as exc:
                self.preview.setPlainText(self._tr("error").format(error=exc))
                self._set_status("error", error=str(exc))
                self.inspectionChanged.emit(None)
                raise
            self._inspection = inspection
            self._refresh_preview()
            self.load_image_button.setEnabled(self.can_load_image)
            self.export_button.setEnabled(True)
            if inspection.session is not None and not self.can_load_image:
                self._set_status("missing_geometry")
            else:
                self._set_status("ready")
            self.inspectionChanged.emit(inspection)
            return inspection

        def _refresh_preview(self) -> None:
            if self._inspection is None:
                self.preview.setPlainText(self._tr("empty"))
                return
            chunks = [self._inspection.preview_text]
            if self._inspection.warnings:
                chunks.extend(("", "Warnings:", *self._inspection.warnings))
            self.preview.setPlainText("\n".join(chunks))
            self.preview.moveCursor(QtGui.QTextCursor.MoveOperation.End)

        def load_session_image(self) -> LegacySaxsLoadedImage:
            if self._inspection is None:
                raise LegacySaxsError("Inspect a Fusion session before loading its image.")
            try:
                loaded = load_legacy_saxs_image(self._inspection)
            except (LegacySaxsError, OSError, ValueError) as exc:
                self._set_status("error", error=str(exc))
                raise
            self._set_status("loaded")
            self.imageLoaded.emit(loaded)
            return loaded

        def _load_from_button(self) -> None:
            try:
                self.load_session_image()
            except (LegacySaxsError, OSError, ValueError):
                # The actionable message is already shown in the status label.
                pass

        def export_receipt(
            self,
            target: str | Path,
            *,
            overwrite: bool = False,
        ) -> Path:
            if self._inspection is None:
                raise LegacySaxsError("Inspect a legacy source before exporting its receipt.")
            try:
                written = write_compatibility_receipt(
                    self._inspection, target, overwrite=overwrite
                )
            except (LegacySaxsError, OSError, ValueError) as exc:
                self._set_status("error", error=str(exc))
                raise
            self._set_status("receipt_written", path=str(written))
            return written

        def _choose_file(self) -> None:
            path, _selected_filter = QtWidgets.QFileDialog.getOpenFileName(
                self,
                self._tr("pick_file_title"),
                "",
                self._tr("file_filter"),
            )
            if path:
                self._inspect_selected(path)

        def _choose_folder(self) -> None:
            path = QtWidgets.QFileDialog.getExistingDirectory(
                self, self._tr("pick_folder_title")
            )
            if path:
                self._inspect_selected(path)

        def _inspect_selected(self, path: str | Path) -> None:
            try:
                self.set_source(path)
            except (LegacySaxsError, OSError, ValueError) as exc:
                self._set_status("error", error=str(exc))

        def _choose_receipt_target(self) -> None:
            if self._inspection is None:
                return
            source = self._inspection.path
            default_name = (
                f"{source.name}.wing_saxs_compatibility.json"
                if source.is_dir()
                else f"{source.stem}.wing_saxs_compatibility.json"
            )
            initial = str(source.parent / default_name)
            path, _selected_filter = QtWidgets.QFileDialog.getSaveFileName(
                self,
                self._tr("save_title"),
                initial,
                self._tr("receipt_filter"),
            )
            if path:
                try:
                    overwrite = False
                    if Path(path).exists():
                        answer = QtWidgets.QMessageBox.question(
                            self,
                            self._tr("overwrite_title"),
                            self._tr("overwrite_prompt"),
                            QtWidgets.QMessageBox.StandardButton.Yes
                            | QtWidgets.QMessageBox.StandardButton.No,
                            QtWidgets.QMessageBox.StandardButton.No,
                        )
                        overwrite = answer == QtWidgets.QMessageBox.StandardButton.Yes
                    if overwrite or not Path(path).exists():
                        self.export_receipt(path, overwrite=overwrite)
                except (LegacySaxsError, OSError, ValueError):
                    pass


else:

    class LegacySaxsDialog:  # pragma: no cover - exercised without optional UI deps
        """Placeholder that reports the missing optional Qt stack on creation."""

        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            require_qt()


__all__ = ["LegacySaxsDialog"]
