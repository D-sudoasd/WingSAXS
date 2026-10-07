"""Help for the plot library's embedded display controls."""

AXIS_HELP = {
    "linkCombo": ("选择另一个图，让两图的这条坐标轴保持相同显示范围；留空取消关联。", "Choose another plot to share this axis's displayed range; leave blank to unlink."),
    "autoPercentSpin": ("设置自动取范围时要包含多少比例的数据点；降低可减少极端值对缩放的影响。", "Set the fraction of data included in automatic ranges; lower it to reduce the effect of extreme values."),
    "autoRadio": ("让坐标轴随当前数据自动调整显示范围。", "Adjust this axis's displayed range automatically to the data."),
    "manualRadio": ("固定使用下方输入的显示范围。", "Use the fixed display limits entered below."),
    "minText": ("输入坐标轴显示的最小值，使用这条轴的单位。", "Enter the lowest displayed value in this axis's units."),
    "maxText": ("输入坐标轴显示的最大值，使用这条轴的单位。", "Enter the highest displayed value in this axis's units."),
    "invertCheck": ("反转这条坐标轴的显示方向，数据数值不变。", "Reverse this axis's displayed direction without changing values."),
    "mouseCheck": ("允许用鼠标缩放或移动这条坐标轴。", "Allow mouse zooming and panning along this axis."),
    "visibleOnlyCheck": ("自动调整范围时，只使用另一条轴范围内能看见的数据。", "Use only data visible within the other axis when adjusting this axis's range."),
    "autoPanCheck": ("保持当前缩放大小，只自动移动显示范围以跟随数据。", "Keep the zoom level and automatically move the view to follow the data."),
}

PLOT_HELP = {
    "avgParamList": ("选择按哪些曲线参数分组后显示平均曲线。", "Choose the curve parameters used to group and display averaged curves."),
    "averageGroup": ("勾选后显示同组曲线的平均值，原始数据保留。", "Display the average of curves in each group while keeping the original data."),
    "clipToViewCheck": ("只绘制当前横轴范围内的曲线部分，减少绘图工作量。", "Draw only curve sections in the visible horizontal range to reduce rendering work."),
    "maxTracesCheck": ("限制同时显示的曲线数量，用下方数字设置上限。", "Limit the number of curves displayed at once using the number below."),
    "maxTracesSpin": ("输入最多同时显示多少条曲线。", "Enter the maximum number of curves shown at once."),
    "forgetTracesCheck": ("超过显示数量上限时，从绘图列表移除旧曲线；不会删除原文件。", "Remove older curves from the plot list beyond the display limit; source files stay unchanged."),
    "downsampleCheck": ("减少用于显示的曲线点数，提升大量数据的绘图速度。", "Display fewer curve points for faster plotting of large datasets."),
    "autoDownsampleCheck": ("根据窗口大小自动选择需要显示的点数。", "Choose the displayed point count automatically for the window size."),
    "downsampleSpin": ("每多少个原始点合成一个显示组；越大绘图越快、细节越少。", "Set how many original points form one display group; larger groups draw faster with less detail."),
    "subsampleRadio": ("每组只显示一个原始点，速度较快，但可能漏掉窄峰。", "Show one original point per group; fast, but narrow peaks may be missed."),
    "meanRadio": ("每组显示平均值，使显示曲线更平滑。", "Show the average of each group for a smoother displayed curve."),
    "peakRadio": ("每组保留最高和最低值，尽量保留窄峰的显示。", "Keep each group's highest and lowest values to preserve visible narrow peaks."),
    "logXCheck": ("横轴按数量级显示，便于同时查看相差很大的位置尺度。", "Show the horizontal axis by orders of magnitude to compare widely different position scales."),
    "logYCheck": ("纵轴按数量级显示，便于查看弱信号；零和负数无法在此轴上显示。", "Show the vertical axis by orders of magnitude to inspect weak signals; zero and negative values cannot be displayed."),
    "derivativeCheck": ("显示曲线变化快慢，而不是原来的数值。", "Display how quickly the curve changes instead of its original values."),
    "phasemapCheck": ("把曲线数值与其变化快慢画在同一图中。", "Plot curve values against their rate of change."),
    "fftCheck": ("把曲线转换为重复频率图，用来查看其中重复变化的间隔。", "Transform the curve into a frequency plot to inspect repeated variations."),
    "subtractMeanCheck": ("从显示曲线中减去它的平均值，突出相对起伏。", "Subtract the curve's average from its display to emphasize relative fluctuations."),
    "pointsGroup": ("勾选后在曲线上显示各个数据点。", "Show individual data-point markers on curves."),
    "autoPointsCheck": ("仅在放大到能分清各点时显示标记。", "Show point markers only when zoomed in enough to distinguish them."),
    "xGridCheck": ("显示横轴位置对应的网格线，方便读数。", "Show grid lines at horizontal-axis ticks for reading values."),
    "yGridCheck": ("显示纵轴位置对应的网格线，方便读数。", "Show grid lines at vertical-axis ticks for reading values."),
    "gridAlphaSlider": ("拖动调整网格线深浅。", "Drag to adjust grid-line opacity."),
    "alphaGroup": ("勾选后启用曲线透明度设置，便于查看重叠曲线。", "Enable curve-opacity settings to inspect overlapping curves."),
    "autoAlphaCheck": ("曲线重叠较多时，自动减淡显示。", "Automatically fade curves when many overlap."),
    "alphaSlider": ("拖动调整曲线透明度。", "Drag to adjust curve opacity."),
}

MENU_HELP = {
    "View All": ("恢复能看见全部数据的显示范围。", "Restore a view that includes all data."),
    "X axis": ("调整横轴范围、方向和鼠标操作。", "Adjust horizontal-axis limits, direction and mouse interaction."),
    "Y axis": ("调整纵轴范围、方向和鼠标操作。", "Adjust vertical-axis limits, direction and mouse interaction."),
    "Mouse Mode": ("选择拖动鼠标时是平移图像，还是画框放大。", "Choose whether mouse dragging pans the plot or draws a zoom box."),
    "3 button": ("左键拖动平移，右键拖动缩放。", "Drag with the left button to pan and the right button to zoom."),
    "1 button": ("左键拖出矩形，放大该区域。", "Drag a rectangle with the left button to zoom into it."),
    "Transforms": ("改变曲线的显示方式，例如对数坐标或变化快慢。", "Change the curve display, such as logarithmic axes or rates of change."),
    "Downsample": ("减少显示点数以加快绘图，原始测量数据保留。", "Reduce displayed points for faster drawing while keeping original measurements."),
    "Average": ("设置按曲线组显示平均值。", "Set averaging for curve groups."),
    "Alpha": ("设置曲线透明度。", "Set curve opacity."),
    "Grid": ("显示坐标网格并调整网格线深浅。", "Show axis grids and adjust their opacity."),
    "Points": ("设置曲线是否显示数据点标记。", "Set whether curves show data-point markers."),
}


def apply_plot_help(plot, language: str, set_help) -> None:
    """Apply help to pyqtgraph controls when that optional library is in use."""
    from .qt_compat import QtCore, QtWidgets

    item = plot.getPlotItem()
    index = 1 if language == "en" else 0
    for owner, catalog in [(item.ctrl, PLOT_HELP), *[(ctrl, AXIS_HELP) for ctrl in item.vb.menu.ctrl]]:
        for name, pair in catalog.items():
            control = getattr(owner, name, None)
            if control is not None:
                set_help(control, pair[index])
                if isinstance(control, QtWidgets.QComboBox):
                    for i in range(control.count()):
                        name = control.itemText(i)
                        text = (f"与 {name} 保持相同坐标轴范围。" if index == 0 else f"Share this axis range with {name}.") if name else (
                            "独立调整这条轴的显示范围。" if index == 0 else "Adjust this axis range independently.")
                        control.setItemData(i, text, QtCore.Qt.ItemDataRole.ToolTipRole)
    item.autoBtn.setToolTip(("自动调整坐标轴范围，让所有数据都可见。", "Fit the axis ranges to show all data.")[index])
    for menu in (item.ctrlMenu, item.vb.menu):
        menu.setToolTipsVisible(True)
        for action in [*menu.actions(), *[a for m in menu.findChildren(QtWidgets.QMenu) for a in m.actions()]]:
            pair = MENU_HELP.get(action.text())
            if pair is not None:
                set_help(action, pair[index])
