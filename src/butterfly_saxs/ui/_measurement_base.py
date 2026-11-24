"""Shared Matplotlib canvas, worker-pool, and export scaffolding for 2-D measurement pages."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ._common import confirm_save_path, display_q_unit, safe_single_shot
from .qt_compat import QT_AVAILABLE, QtCore, QtWidgets, require_qt
from .workers import AnalysisWorker, GenerationGuard


if QT_AVAILABLE:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.figure import Figure

    class _MeasurementCanvasPage(QtWidgets.QWidget):
        """Base widget for background-analyzed 2-D Matplotlib measurement pages."""

        profileChanged = QtCore.Signal(object, object)
        _worker_kind = "measurement"

        def __init__(
            self,
            parent: Any = None,
            *,
            page_prefix: str,
            language: str = "zh_CN",
        ) -> None:
            super().__init__(parent)
            self.setObjectName(f"{page_prefix}Page")
            self.language = str(language)
            self.data: np.ndarray | None = None
            self.qx: np.ndarray | None = None
            self.qy: np.ndarray | None = None
            self.valid_mask: np.ndarray | None = None
            self.q_unit = "unknown"
            self.source: str | None = None
            self.profile: Any = None
            self._generation = GenerationGuard()
            self._worker: AnalysisWorker | None = None
            self._closed = False
            self._thread_pool = QtCore.QThreadPool(self)
            self._thread_pool.setMaxThreadCount(1)
            self._labels: dict[str, Any] = {}

        def _build_page_scaffold(
            self,
            *,
            prefix: str,
            settings_max_height: int,
            figsize: tuple[float, float],
            width_ratios: tuple[float, float],
            height_ratios: tuple[float, float],
        ) -> QtWidgets.QGridLayout:
            root = QtWidgets.QVBoxLayout(self)
            root.setContentsMargins(12, 10, 12, 10)
            root.setSpacing(8)

            self.heading = QtWidgets.QLabel()
            self.heading.setObjectName(f"{prefix}Heading")
            heading_font = self.heading.font()
            heading_font.setPointSize(15)
            heading_font.setBold(True)
            self.heading.setFont(heading_font)
            root.addWidget(self.heading)

            self.description = QtWidgets.QLabel()
            self.description.setWordWrap(True)
            self.description.setObjectName(f"{prefix}Description")
            root.addWidget(self.description)

            self.settings_group = QtWidgets.QGroupBox()
            self.settings_group.setObjectName(f"{prefix}Settings")
            controls = QtWidgets.QGridLayout(self.settings_group)
            controls.setHorizontalSpacing(8)
            controls.setVerticalSpacing(6)

            self.settings_scroll = QtWidgets.QScrollArea()
            self.settings_scroll.setObjectName(f"{prefix}SettingsScroll")
            self.settings_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
            self.settings_scroll.setWidgetResizable(True)
            self.settings_scroll.setMinimumHeight(85)
            self.settings_scroll.setMaximumHeight(int(settings_max_height))
            self.settings_scroll.setWidget(self.settings_group)
            root.addWidget(self.settings_scroll)

            actions = QtWidgets.QHBoxLayout()
            self.analyze_button = QtWidgets.QPushButton()
            self.analyze_button.setObjectName(f"{prefix}Analyze")
            self.analyze_button.setProperty("role", "primary")
            self.analyze_button.setEnabled(False)
            self.export_button = QtWidgets.QPushButton()
            self.export_button.setObjectName(f"{prefix}Export")
            self.export_button.setEnabled(False)
            self.status_label = QtWidgets.QLabel()
            self.status_label.setObjectName(f"{prefix}Status")
            self.status_label.setWordWrap(True)
            self.status_label.setAlignment(
                QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop
            )
            self.status_label.setTextInteractionFlags(
                QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
            )
            self.status_label.setMargin(4)
            actions.addWidget(self.analyze_button)
            actions.addWidget(self.export_button)
            actions.addStretch(1)
            root.addLayout(actions)

            self.progress = QtWidgets.QProgressBar()
            self.progress.setObjectName(f"{prefix}Progress")
            self.progress.setRange(0, 0)
            self.progress.setTextVisible(False)
            self.progress.setFixedHeight(4)
            self.progress.hide()
            root.addWidget(self.progress)

            self.feedback_area = QtWidgets.QScrollArea()
            self.feedback_area.setObjectName(f"{prefix}Feedback")
            self.feedback_area.setWidgetResizable(True)
            self.feedback_area.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
            self.feedback_area.setFixedHeight(42)
            self.feedback_area.setWidget(self.status_label)
            root.addWidget(self.feedback_area)

            self.figure = Figure(figsize=figsize, constrained_layout=True)
            grid = self.figure.add_gridspec(
                2, 2, width_ratios=width_ratios, height_ratios=height_ratios
            )
            self.image_axes = self.figure.add_subplot(grid[:, 0])
            self.profile_axes = self.figure.add_subplot(grid[0, 1])
            self.residual_axes = self.figure.add_subplot(
                grid[1, 1], sharex=self.profile_axes
            )
            self.canvas = FigureCanvasQTAgg(self.figure)
            self.canvas.setObjectName(f"{prefix}Canvas")
            self.canvas.setMinimumHeight(240)
            root.addWidget(self.canvas, 1)

            self.analyze_button.clicked.connect(self.start_analysis)
            self.export_button.clicked.connect(self._choose_export_path)
            return controls

        def _populate_control_grid(
            self,
            controls: QtWidgets.QGridLayout,
            items: Sequence[tuple[str, Any]],
        ) -> None:
            for index, (key, widget) in enumerate(items):
                row, col = divmod(index, 3)
                label = QtWidgets.QLabel()
                label.setWordWrap(True)
                label.setBuddy(widget)
                self._labels[key] = label
                controls.addWidget(label, row * 2, col)
                controls.addWidget(widget, row * 2 + 1, col)
                controls.setColumnStretch(col, 1)
                widget.setSizePolicy(
                    QtWidgets.QSizePolicy.Policy.Expanding,
                    QtWidgets.QSizePolicy.Policy.Fixed,
                )

        @staticmethod
        def _double_spin(
            minimum: float = 0.0,
            maximum: float = 1.0e12,
            value: float = 0.0,
            decimals: int = 4,
        ) -> Any:
            spin = QtWidgets.QDoubleSpinBox()
            spin.setRange(float(minimum), float(maximum))
            spin.setDecimals(int(decimals))
            spin.setValue(float(value))
            spin.setKeyboardTracking(False)
            return spin

        def _tr(self, key: str) -> str:  # pragma: no cover - implemented by subclasses
            raise NotImplementedError

        def _retranslate_extra_controls(self) -> None:
            """Hook for subclass-specific non-buddy controls during language updates."""

        def _present_current_result(self) -> None:  # pragma: no cover - subclass hook
            raise NotImplementedError

        def _clear_results(self) -> None:  # pragma: no cover - subclass hook
            self.profile = None

        def _apply_language(self) -> None:
            self.heading.setText(self._tr("heading"))
            self.description.setText(self._tr("description"))
            self.settings_group.setTitle(self._tr("settings"))
            self.settings_scroll.setAccessibleName(self._tr("settings"))
            self.progress.setAccessibleName(self._tr("running"))
            for key, label in self._labels.items():
                label.setText(self._tr(key))
                if label.buddy() is not None:
                    label.buddy().setAccessibleName(self._tr(key))
            self._retranslate_extra_controls()
            self.analyze_button.setText(self._tr("analyze"))
            self.export_button.setText(self._tr("export"))
            if self.jobs_running():
                key = (
                    "changed_running"
                    if self._worker and not self._generation.is_current(self._worker.generation)
                    else "running"
                )
                self._set_feedback(self._tr(key), "running")
            elif self.data is None:
                self._set_feedback(self._tr("no_data"), "empty")
            elif self.profile is None:
                self._set_feedback(self._tr("ready"), "ready")
            else:
                self._present_current_result()
                return
            self._draw()

        def set_language(self, language: str) -> None:
            self.language = str(language)
            self._apply_language()
            from .help import apply_help

            apply_help(self, self.language)

        def _validate_frame_arrays(
            self,
            data: Any,
            *,
            qx: Any,
            qy: Any,
            valid_mask: Any,
            empty_q_error: str = "no finite q coordinates",
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
            image = np.asarray(data, dtype=float)
            if image.ndim != 2 or image.size == 0:
                self.invalidate(
                    self._tr("analysis_error").format(error="data must be a non-empty 2-D image")
                )
                raise ValueError("data must be a non-empty two-dimensional image")
            try:
                qx_map, qy_map = np.broadcast_arrays(
                    np.asarray(qx, dtype=float), np.asarray(qy, dtype=float)
                )
                qx_map = np.broadcast_to(qx_map, image.shape)
                qy_map = np.broadcast_to(qy_map, image.shape)
                if valid_mask is None:
                    usable = np.ones(image.shape, dtype=bool)
                else:
                    usable = np.broadcast_to(np.asarray(valid_mask, dtype=bool), image.shape)
            except (TypeError, ValueError) as exc:
                self.invalidate(
                    self._tr("analysis_error").format(
                        error="qx, qy, and valid_mask must match image shape"
                    )
                )
                raise ValueError("qx, qy, and valid_mask must broadcast to the image shape") from exc

            q_radius = np.hypot(qx_map, qy_map)
            finite_q = q_radius[np.isfinite(q_radius)]
            if not finite_q.size:
                self.invalidate(
                    self._tr("analysis_error").format(error="no finite q coordinates")
                )
                raise ValueError(empty_q_error)
            lower, upper = float(np.min(finite_q)), float(np.max(finite_q))
            if not upper > lower:
                self.invalidate(
                    self._tr("analysis_error").format(error="q map has no radial range")
                )
                raise ValueError("q map has no radial range")
            return image, qx_map, qy_map, usable, q_radius, lower, upper

        def _configure_q_spin_bounds(
            self,
            lower: float,
            upper: float,
            q0: float,
            q1: float,
        ) -> None:
            span = upper - lower
            step = max(span / 100.0, np.finfo(float).eps)
            unit_suffix = f" {display_q_unit(self.q_unit)}"
            self.q_min_spin.blockSignals(True)
            self.q_max_spin.blockSignals(True)
            try:
                self.q_min_spin.setRange(lower, upper)
                self.q_max_spin.setRange(lower, upper)
                self.q_min_spin.setDecimals(10)
                self.q_max_spin.setDecimals(10)
                self.q_min_spin.setSingleStep(step)
                self.q_max_spin.setSingleStep(step)
                self.q_min_spin.setSuffix(unit_suffix)
                self.q_max_spin.setSuffix(unit_suffix)
                self.q_min_spin.setValue(q0)
                self.q_max_spin.setValue(q1)
            finally:
                self.q_min_spin.blockSignals(False)
                self.q_max_spin.blockSignals(False)

        def _finish_set_data(self) -> None:
            self.analyze_button.setEnabled(not self.jobs_running())
            self.export_button.setEnabled(False)
            running = self.jobs_running()
            self._set_feedback(
                self._tr("changed_running") if running else self._tr("ready"),
                "running" if running else "ready",
            )
            self._draw()

        def _reset_base_state(self, message: str | None = None) -> None:
            self._generation.next()
            self.data = None
            self.qx = None
            self.qy = None
            self.valid_mask = None
            self.q_unit = "unknown"
            self.source = None
            self._clear_results()
            self.analyze_button.setEnabled(False)
            self.export_button.setEnabled(False)
            self._set_feedback(message or self._tr("no_data"), "empty")

        def _set_feedback(self, text: str, state: str) -> None:
            self.status_label.setText(text)
            self.status_label.setProperty("state", state)
            self.status_label.style().unpolish(self.status_label)
            self.status_label.style().polish(self.status_label)

        def _prepare_analysis(self) -> None:
            self._clear_results()
            self.export_button.setEnabled(False)
            self.analyze_button.setEnabled(False)
            self._set_feedback(self._tr("running"), "running")
            self._draw()

        def _check_analysis_preconditions(self) -> bool:
            return True

        def _worker_job(self) -> Callable[..., Any]:  # pragma: no cover - subclass hook
            raise NotImplementedError

        def _worker_request_args(self) -> tuple[dict[str, Any], Any]:  # pragma: no cover
            raise NotImplementedError

        def _run_sync_job(self) -> Any:  # pragma: no cover - subclass hook
            raise NotImplementedError

        def _apply_worker_result(self, result: Any) -> Any:  # pragma: no cover
            raise NotImplementedError

        def analyze(self) -> Any:
            """Synchronous integration API; button-triggered fits run in a worker."""

            if self.data is None or self.jobs_running():
                return None
            if not self._check_analysis_preconditions():
                return None
            self._prepare_analysis()
            try:
                raw_result = self._run_sync_job()
            except (TypeError, ValueError, RuntimeError) as exc:
                self._set_feedback(self._tr("analysis_error").format(error=exc), "error")
                return None
            finally:
                self.analyze_button.setEnabled(self.data is not None)
            return self._apply_worker_result(raw_result)

        def start_analysis(self) -> None:
            """Run one immutable frame/settings snapshot without blocking the UI."""

            if self.data is None or self.jobs_running():
                return
            if not self._check_analysis_preconditions():
                return
            self._closed = False
            parameters, payload = self._worker_request_args()
            self._prepare_analysis()
            worker = AnalysisWorker(
                self._worker_job(),
                generation=self._generation.next(),
                kind=self._worker_kind,
                parameters=parameters,
                payload=payload,
            )
            worker.signals.finished.connect(
                self._analysis_finished, QtCore.Qt.ConnectionType.QueuedConnection
            )
            worker.signals.error.connect(
                self._analysis_failed, QtCore.Qt.ConnectionType.QueuedConnection
            )
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
            self._apply_worker_result(result)

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

        def wait_for_done(self, timeout_ms: int = 1500) -> bool:
            """Drain the private worker pool cleanly during teardown."""

            self._thread_pool.clear()
            return bool(self._thread_pool.waitForDone(int(timeout_ms)))

        def closeEvent(self, event: Any) -> None:  # noqa: N802 - Qt API
            self.shutdown()
            if self.jobs_running():
                event.ignore()
                safe_single_shot(50, self, self.close)
                return
            self.wait_for_done()
            event.accept()

        def showEvent(self, event: Any) -> None:  # noqa: N802 - Qt API
            super().showEvent(event)
            self.canvas.draw_idle()

        def _draw(self) -> None:  # pragma: no cover - subclass hook
            raise NotImplementedError

        def _show_empty(self) -> None:
            self._draw()

        def _choose_export_path(self) -> None:  # pragma: no cover - subclass hook
            raise NotImplementedError

        def _prompt_and_export(
            self,
            *,
            default_name: str,
            extra_targets: Callable[[Path], Sequence[Path]],
            format_done: Callable[[Any], str],
        ) -> None:
            chosen, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, self._tr("export"), default_name, self._tr("file_filter")
            )
            if not chosen:
                return
            confirmed = confirm_save_path(
                self,
                chosen,
                suffix=".csv",
                extra_targets=extra_targets,
                title=self._tr("export"),
                prompt=self._tr("export_exists"),
            )
            if confirmed is None:
                return
            target, overwrite = confirmed
            try:
                paths = self.export_to(target, overwrite=overwrite)
                self.status_label.setText(format_done(paths))
            except (OSError, ValueError) as exc:
                self.status_label.setText(self._tr("export_error").format(error=exc))

else:

    class _MeasurementCanvasPage:  # pragma: no cover - without optional UI dependencies
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            require_qt()


__all__ = ["_MeasurementCanvasPage"]
