"""Install concise, translated help on existing Qt controls without changing data."""

from . import help_primary, help_secondary
from .help_plots import apply_plot_help
from .i18n import translate, validate_language
from .qt_compat import QT_AVAILABLE, QtCore, QtGui, QtWidgets


def _text(pair: tuple[str, str], language: str) -> str:
    return pair[1 if language == "en" else 0]


def _set_help(control, text: str) -> None:
    control.setToolTip(text)
    control.setWhatsThis(text)
    if isinstance(control, QtGui.QAction):
        control.setStatusTip(text)
    elif isinstance(control, QtWidgets.QWidget):
        control.setAccessibleDescription(text)
        # Qt's embedded editor otherwise intercepts hovering over spin-box text.
        if isinstance(control, QtWidgets.QAbstractSpinBox):
            control.lineEdit().setToolTip(text)
            control.lineEdit().setWhatsThis(text)
        elif isinstance(control, QtWidgets.QComboBox) and control.isEditable():
            control.lineEdit().setToolTip(text)
            control.lineEdit().setWhatsThis(text)


def _field(owner, name: str):
    control = getattr(owner, name, None)
    if control is None:
        control = owner.findChild(QtCore.QObject, name)
    return control


def refresh_evaluation_help(workbench, language: str) -> None:
    """Update the action description when the repeat-count setting changes."""
    resamples = workbench._evaluation_resamples()
    if resamples:
        text = translate(language, "tooltip.butterfly_evaluate", resamples=resamples)
    else:
        text = _text(("按已识别的亮弧计算形状和结果；此次不重复模拟图像噪声。",
                      "Fit the shape and results from the traced arcs without repeated image-noise simulations."), language)
    _set_help(workbench.evaluate_button, text)


def apply_help(root, language: str = "zh_CN") -> None:
    """Refresh help after constructing or translating a page or dialog.

    Catalog entries address stable attributes or object names, never translated
    captions. The function changes only help roles and does not connect signals.
    """
    if not QT_AVAILABLE:
        return
    language = validate_language(language)
    owners = [root, *root.findChildren(QtWidgets.QWidget)]
    for owner in owners:
        class_name = type(owner).__name__
        for catalog in (help_primary, help_secondary):
            for name, pair in catalog.CONTROL_HELP.get(class_name, {}).items():
                control = _field(owner, name)
                if control is not None:
                    text = _text(pair, language)
                    if name == "flags_label":
                        text += "\n" + control.text()
                    _set_help(control, text)
            for name, options in catalog.OPTION_HELP.get(class_name, {}).items():
                combo = _field(owner, name)
                if not isinstance(combo, QtWidgets.QComboBox):
                    continue
                for index in range(combo.count()):
                    pair = options.get(combo.itemData(index))
                    if pair is not None:
                        text = _text(pair, language)
                        combo.setItemData(index, text, QtCore.Qt.ItemDataRole.ToolTipRole)
                        combo.setItemData(index, text, QtCore.Qt.ItemDataRole.WhatsThisRole)
            for name, tabs in getattr(catalog, "TAB_HELP", {}).get(class_name, {}).items():
                control = _field(owner, name)
                if isinstance(control, QtWidgets.QTabWidget):
                    for index, pair in tabs.items():
                        if index < control.count():
                            control.setTabToolTip(index, _text(pair, language))
                            control.setTabWhatsThis(index, _text(pair, language))
            for name, columns in getattr(catalog, "COLUMN_HELP", {}).get(class_name, {}).items():
                table = _field(owner, name)
                if isinstance(table, QtWidgets.QTableWidget):
                    for index, pair in columns.items():
                        item = table.horizontalHeaderItem(index)
                        if item is not None:
                            item.setToolTip(_text(pair, language))
                elif isinstance(table, QtWidgets.QTreeWidget):
                    for index, pair in columns.items():
                        table.headerItem().setToolTip(index, _text(pair, language))
        for name, key in help_primary.CATALOG_HELP.get(class_name, {}).items():
            control = _field(owner, name)
            if control is not None:
                _set_help(control, translate(language, key))
        if class_name == "ButterflyWorkbench":
            refresh_evaluation_help(owner, language)
            for index in range(owner.display_scale_combo.count()):
                key = "tooltip.combo.display_" + str(owner.display_scale_combo.itemData(index))
                owner.display_scale_combo.setItemData(index, translate(language, key), QtCore.Qt.ItemDataRole.ToolTipRole)
        if owner.objectName() == "butterflyNumericDialog":
            combo = owner.findChild(QtWidgets.QComboBox, "butterflyNumericMode")
            if combo is not None:
                for index in range(combo.count()):
                    data = combo.itemData(index)
                    key = {"exclude_polygon": "polygon_exclude", "include_polygon": "polygon_include",
                           "exclude_rectangle": "rectangle_exclude", "include_rectangle": "rectangle_include"}.get(data, data)
                    pair = help_primary.OPTION_HELP["ButterflyWorkbench"]["correction_mode_combo"].get(key)
                    if pair is not None:
                        text = _text(pair, language)
                        if key.startswith("polygon_") or key.startswith("rectangle_"):
                            text = _text(("在顶点框输入坐标，点击确定后应用所选区域。", "Enter coordinates in the Points field, then press OK to apply the selected area."), language)
                        elif key == "seed":
                            text = _text(("在下方输入水平和竖直坐标，点击确定后添加一个引导点。", "Enter horizontal and vertical coordinates below, then press OK to add a guide point."), language)
                        combo.setItemData(index, text, QtCore.Qt.ItemDataRole.ToolTipRole)

    # Object-name help includes dynamic lamellar controls and branch switches.
    for control in [*owners, *root.findChildren(QtGui.QAction)]:
        name = control.objectName()
        if isinstance(control, QtGui.QAction) and not name and control.property("textKey"):
            name = "lamellar_" + str(control.property("textKey"))
        for catalog in (help_primary, help_secondary):
            pair = catalog.OBJECT_HELP.get(name)
            if pair is not None:
                _set_help(control, _text(pair, language))
        if isinstance(control, QtWidgets.QMenu):
            control.setToolTipsVisible(True)

    # Preserve existing translated main-window help, and expose it through
    # Qt's What's This mode as well. Form captions share their field's help.
    for control in owners:
        if control.toolTip():
            control.setWhatsThis(control.toolTip())
        if isinstance(control, QtWidgets.QAbstractSpinBox) and control.toolTip():
            _set_help(control, control.toolTip())
        if isinstance(control, QtWidgets.QAbstractItemView) and control.toolTip():
            control.viewport().setToolTip(control.toolTip())
        if callable(getattr(control, "getPlotItem", None)):
            apply_plot_help(control, language, _set_help)
        if isinstance(control, QtWidgets.QTabWidget):
            for index in range(control.count()):
                pair = help_primary.PAGE_HELP.get(type(control.widget(index)).__name__)
                if pair is not None:
                    control.setTabToolTip(index, _text(pair, language))
                    control.setTabWhatsThis(index, _text(pair, language))
    for form in root.findChildren(QtWidgets.QFormLayout):
        for row in range(form.rowCount()):
            field_item = form.itemAt(row, QtWidgets.QFormLayout.ItemRole.FieldRole)
            label_item = form.itemAt(row, QtWidgets.QFormLayout.ItemRole.LabelRole)
            if field_item is None or label_item is None:
                continue
            field = field_item.widget()
            label = label_item.widget()
            if field is not None and label is not None and field.toolTip():
                _set_help(label, field.toolTip())
    for action in root.findChildren(QtGui.QAction):
        if action.toolTip():
            action.setWhatsThis(action.toolTip())
