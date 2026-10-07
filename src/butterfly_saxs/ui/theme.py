"""Window-scoped native Qt styling for the WingSAXS workbench."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .qt_compat import QT_AVAILABLE, QtGui, QtWidgets


COLORS = {
    "background": "#f3f6fa",
    "surface": "#ffffff",
    "text": "#172b46",
    "muted": "#52647b",
    "border": "#cbd5e1",
    "accent": "#1d5fbe",
    "accent_hover": "#174e9c",
    "selection": "#e6effc",
    "danger": "#ad303b",
}

_STYLE = """
QWidget { color: #172b46; }
QMainWindow, QDialog { background: #f3f6fa; }
QWidget[surface="card"] {
    background: #ffffff; border: 1px solid #cbd5e1; border-radius: 7px;
}
QLabel#viewTitle { color: #172b46; font-weight: 600; background: transparent; }
QLabel#viewSubtitle, QLabel#viewStateLabel {
    color: #52647b; background: transparent; border: none;
}
QLabel#viewStateLabel { padding: 16px; }
QGroupBox {
    background: #ffffff; border: 1px solid #cbd5e1; border-radius: 6px;
    margin-top: 12px; padding: 10px 8px 8px;
}
QGroupBox::title {
    subcontrol-origin: margin; left: 10px; padding: 0 4px;
    color: #334b67; font-weight: 600;
}
QPushButton {
    background: #ffffff; border: 1px solid #b9c6d6; border-radius: 5px;
    padding: 5px 11px; min-height: 18px;
}
QPushButton:hover { background: #edf3fb; border-color: #8ca5c5; }
QPushButton:pressed { background: #dce8f7; }
QPushButton[role="primary"] {
    background: #1d5fbe; color: #ffffff; border-color: #1d5fbe; font-weight: 600;
}
QPushButton[role="primary"]:hover { background: #174e9c; border-color: #174e9c; }
QPushButton[role="primary"]:pressed { background: #123f80; }
QPushButton[role="danger"] { color: #ad303b; border-color: #d6a6ac; }
QPushButton[role="danger"]:hover { background: #fcecef; border-color: #ad303b; }
QPushButton:focus {
    border: 2px solid #1d5fbe; padding: 4px 10px;
}
QToolButton:focus { border: 2px solid #1d5fbe; padding: 4px 6px; }
QPushButton[role="primary"]:focus { border-color: #102f59; }
QPushButton:disabled, QToolButton:disabled {
    background: #edf1f6; color: #7b8798; border-color: #d7dfe9;
}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QTextEdit, QPlainTextEdit {
    background: #ffffff; selection-background-color: #1d5fbe;
    selection-color: #ffffff; border: 1px solid #b9c6d6;
    border-radius: 4px; padding: 4px 5px;
}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus,
QTextEdit:focus, QPlainTextEdit:focus { border-color: #1d5fbe; }
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled,
QComboBox:disabled, QTextEdit:disabled, QPlainTextEdit:disabled {
    background: #edf1f6; color: #7b8798; border-color: #d7dfe9;
}
QComboBox::drop-down { border: none; width: 20px; }
QLabel[state] { color: #334b67; background: #eaf1f9; padding: 6px; border-radius: 4px; }
QLabel[state="error"] { color: #962e35; background: #fcecef; }
QLabel[state="warning"] { color: #795509; background: #fff5d9; }
QLabel[state="complete"] { color: #22634e; background: #eaf6ef; }
QWidget#workspaceHeader { background: #f3f6fa; border-bottom: 1px solid #d5dfe9; }
QWidget#butterflyControlsPanel, QWidget#batchOptionsPanel { background: #f3f6fa; }
QAbstractItemView {
    background: #ffffff; alternate-background-color: #f5f8fc;
    border: 1px solid #cbd5e1; border-radius: 4px;
    selection-background-color: #e6effc; selection-color: #172b46;
    gridline-color: #e1e7ef;
}
QAbstractItemView::item { padding: 3px; }
QHeaderView::section {
    background: #edf2f8; color: #334b67; border: none;
    border-right: 1px solid #d5dfe9; border-bottom: 1px solid #cbd5e1;
    padding: 5px 7px; font-weight: 600;
}
QTabWidget::pane { background: #ffffff; border: 1px solid #cbd5e1; border-radius: 5px; }
QTabBar::tab {
    background: #edf2f8; color: #52647b; border: 1px solid #cbd5e1;
    padding: 7px 12px; margin-right: 2px;
    border-top-left-radius: 5px; border-top-right-radius: 5px;
}
QTabBar::tab:selected {
    background: #ffffff; color: #1d5fbe; border-bottom-color: #ffffff; font-weight: 600;
}
QTabBar::tab:hover:!selected { background: #e6effc; color: #172b46; }
QTabBar::tab:focus { border-color: #1d5fbe; }
QToolBar {
    background: #ffffff; border: none; border-bottom: 1px solid #d5dfe9;
    spacing: 5px; padding: 5px;
}
QToolButton {
    background: transparent; border: 1px solid transparent;
    border-radius: 4px; padding: 5px 7px;
}
QToolButton:hover, QToolButton:checked { background: #e6effc; border-color: #c1d2ec; }
QToolButton:pressed { background: #dce8f7; }
QMenuBar, QMenu { background: #ffffff; }
QMenuBar::item, QMenu::item { padding: 5px 10px; }
QMenuBar::item:selected, QMenu::item:selected { background: #e6effc; color: #172b46; }
QStatusBar { background: #edf2f8; color: #52647b; border-top: 1px solid #d5dfe9; }
QDockWidget::title { background: #edf2f8; color: #334b67; padding: 7px; }
QSplitter::handle { background: #e1e7ef; }
QSplitter::handle:hover { background: #b7cae4; }
QScrollArea { border: none; background: transparent; }
QProgressBar {
    background: #edf2f8; border: 1px solid #cbd5e1; border-radius: 4px;
    text-align: center; color: #172b46;
}
QProgressBar::chunk { background: #a8c6ee; border-radius: 3px; }
QToolTip { background: #172b46; color: #ffffff; border: none; padding: 6px; }
"""


def apply_theme(window: Any) -> None:
    """Style one window and its descendants without changing QApplication.

    Buttons can declare ``role`` as ``primary``, ``secondary`` (the default
    appearance), or ``danger`` before this function is called. If a role is
    changed later, re-polish that button to update Qt's property selectors.
    """

    if not QT_AVAILABLE:
        return
    palette = QtGui.QPalette(window.palette())
    roles = QtGui.QPalette.ColorRole
    for role, color in (
        (roles.Window, COLORS["background"]),
        (roles.Base, COLORS["surface"]),
        (roles.AlternateBase, "#f5f8fc"),
        (roles.WindowText, COLORS["text"]),
        (roles.Text, COLORS["text"]),
        (roles.Button, COLORS["surface"]),
        (roles.ButtonText, COLORS["text"]),
        (roles.Highlight, COLORS["accent"]),
        (roles.HighlightedText, COLORS["surface"]),
        (roles.Link, COLORS["accent"]),
        (roles.PlaceholderText, COLORS["muted"]),
    ):
        palette.setColor(role, QtGui.QColor(color))
    for role in (roles.WindowText, roles.Text, roles.ButtonText):
        palette.setColor(QtGui.QPalette.ColorGroup.Disabled, role, QtGui.QColor("#7b8798"))
    window.setPalette(palette)
    font = QtGui.QFont(window.font())
    if "Segoe UI" in QtGui.QFontDatabase.families():
        font.setFamily("Segoe UI")
    elif font.family() == "":
        font = QtWidgets.QApplication.font()
    font.setPointSizeF(10.0)
    window.setFont(font)
    icon_dir = Path(__file__).with_name("icons").as_posix()
    indicators = f'''
QComboBox::down-arrow {{ image: url("{icon_dir}/chevron-down.svg"); width: 10px; height: 10px; }}
QSpinBox::up-button, QDoubleSpinBox::up-button {{ width: 18px; }}
QSpinBox::down-button, QDoubleSpinBox::down-button {{ width: 18px; }}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
    image: url("{icon_dir}/chevron-up.svg"); width: 9px; height: 9px;
}}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
    image: url("{icon_dir}/chevron-down.svg"); width: 9px; height: 9px;
}}
'''
    window.setStyleSheet(_STYLE + indicators)


def style_plot(plot: Any) -> None:
    """Give a pyqtgraph plot explicit, legible colors without global config."""

    if not QT_AVAILABLE or plot is None:
        return
    plot.setBackground(COLORS["surface"])
    for name in ("bottom", "left"):
        axis = plot.getAxis(name)
        axis.setPen(COLORS["border"])
        axis.setTextPen(COLORS["muted"])
        axis.setStyle(tickFont=QtGui.QFont("Segoe UI", 9), autoExpandTextSpace=True)
        axis.label.setDefaultTextColor(QtGui.QColor(COLORS["muted"]))
    plot.showGrid(x=True, y=True, alpha=0.14)


__all__ = ["COLORS", "apply_theme", "style_plot"]
