"""Help for the main workspace and butterfly tracing controls (Chinese, English)."""

CONTROL_HELP = {
    "RefinementMainWindow": {
        "angular_plot": ("查看各个方向的亮度曲线；滚轮缩放，拖动平移。", "Inspect brightness by direction; scroll to zoom and drag to pan."),
        "coverage_plot": ("查看各个方向有多少有效数据；滚轮缩放，拖动平移。", "Inspect valid-data coverage by direction; scroll to zoom and drag to pan."),
        "ridge_plot": ("查看亮弧位置随方向的变化，以及被排除的点；滚轮缩放，拖动平移。", "Inspect arc positions by direction and excluded points; scroll to zoom and drag to pan."),
        "radial_profile_plot": ("查看从中心向外方向的亮度曲线；滚轮缩放，拖动平移。", "Inspect brightness profiles outward from the centre; scroll to zoom and drag to pan."),
        "evolution_plot": ("查看所选数值如何随图像序列变化；滚轮缩放，拖动平移，缺失值会断开曲线。", "Inspect the selected value across the image sequence; scroll to zoom and drag to pan. Missing values break the line."),
        "lobe_table": ("查看各个亮瓣的方向和亮度；这些数值来自原图。", "Inspect the direction and brightness of each bright lobe measured in the image."),
        "ridge_table": ("查看沿亮弧找到的位置；被排除的点不会参与椭圆计算。", "Inspect positions found along the bright arcs; excluded points do not enter the ellipse fit."),
        "ellipse_table": ("查看亮弧的椭圆形状和计算质量；椭圆描述图上形状。", "Inspect the shape and fit quality of the ellipse describing the image arcs."),
        "radial_table": ("查看各个方向上亮峰的位置和宽度；缺少数据的项目留空。", "Inspect peak positions and widths along each direction; missing values stay empty."),
        "batch_table": ("查看待处理图像及处理结果；选中行后可从列表移除，原文件保留。", "Inspect queued images and their results; remove selected rows from the list while keeping the files."),
        "evolution_table": ("查看每张图像对应的数值；空白表示该项没有可用结果。", "Inspect values for each image; a blank means no result is available for that quantity."),
        "legacy_saxs_action": ("打开旧版分析结果的读取窗口，检查内容后可载入其中的图像。", "Open the reader for older analysis files, inspect them, then load their image."),
        "ellipse_fixed_angle_check": ("勾选后保持椭圆方向不变，只调整其他未固定的数值。", "Keep the ellipse direction at the entered angle while fitting the other free values."),
        "ellipse_center_qx_spin": ("输入椭圆中心的水平位置，单位与图像的 q 坐标一致。", "Enter the horizontal ellipse-centre position in the image's q-coordinate unit."),
        "ellipse_center_qy_spin": ("输入椭圆中心的竖直位置，单位与图像的 q 坐标一致。", "Enter the vertical ellipse-centre position in the image's q-coordinate unit."),
        "ellipse_angle_min_spin": ("输入允许的最小椭圆角度，单位为度；计算不会越过此限制。", "Enter the lowest allowed ellipse angle in degrees; the fit stays above this limit."),
        "ellipse_angle_max_spin": ("输入允许的最大椭圆角度，单位为度；计算不会越过此限制。", "Enter the highest allowed ellipse angle in degrees; the fit stays below this limit."),
        "rmse_label": ("原图与计算图像的平均差异大小；同一图像和设置下，越小通常越贴近原图。", "Typical difference between the image and the calculated image; smaller usually means a closer fit with the same image and settings."),
        "ndata_label": ("本次计算实际使用的有效像素数量。", "Number of valid pixels actually used in this calculation."),
        "coverage_label": ("所选范围内有多少位置有有效数据；缺失像素不会被补成测量值。", "How much of the selected range contains valid data; missing pixels are not filled with measurements."),
        "flags_label": ("本次结果需要检查的原因；查看后决定是否调整范围、排除区域或计算设置。", "Reasons to review this result and consider changing the range, excluded areas or fit settings."),
    },
    "PatternView": {
        "plot": ("滚轮缩放，拖动平移；右键打开显示设置，调整只影响查看方式。", "Scroll to zoom, drag to pan, and right-click for display settings; these change only the view."),
    },
    "ButterflyWorkbench": {
        "frame_list": ("单击一行切换到对应图像，查看它的亮弧和结果。", "Click a row to switch images and inspect its arcs and results."),
        "point_list": ("单击一个测量点，查看它所在位置的亮度曲线和计算情况。", "Click a measured point to inspect its intensity profile and fit details."),
        "reference_axis_spin": ("输入样品拉伸方向的角度，单位为度；其他方向以它为参照。", "Enter the specimen's drawing direction in degrees; other directions are measured from it."),
        "trace_method_combo": ("选择怎样从原图寻找亮弧；先用圆环找峰，必要时再试其他方法。", "Choose how to find bright arcs in the image; start with peaks around rings, then try another method if needed."),
        "annular_radial_bins": ("把所选范围分成多少个圆环；更多圆环能细分位置，但每环可用像素更少。", "Number of rings across the selected range; more rings give finer positions but fewer pixels per ring."),
        "annular_angle_bins": ("把每个圆环分成多少个方向；更多方向能细分亮峰，但也更容易受噪声影响。", "Number of directions around each ring; more directions separate peaks more finely but reveal more noise."),
        "sector_width_spin": ("输入每个扇形区域的总角度，单位为度；越宽会平均更多像素。", "Enter the full angular width of each sector in degrees; wider sectors average more pixels."),
        "sector_step_spin": ("输入相邻测量方向间隔多少度；越小测量越密、耗时越长。", "Enter the spacing between measurement directions in degrees; smaller steps give more measurements and take longer."),
        "evaluation_resamples_combo": ("选择反复模拟图像噪声并重新计算的次数，用来查看结果波动；次数越多越慢。", "Choose how often to simulate image noise and repeat the fit to inspect result variation; more repeats take longer."),
        "sensitivity_check": ("勾选后略改测量设置再计算，检查结果是否容易随设置变化。", "Repeat the calculation with small setting changes to check how much the result depends on them."),
        "uv_magnification_spin": ("放大测量点偏离椭圆的竖直显示幅度，方便看清小偏差；测量数值不变。", "Magnify the displayed vertical deviations from the ellipse to inspect small differences; measured values stay unchanged."),
        "display_reset_button": ("恢复默认的亮暗显示设置，原图和计算结果不变。", "Restore the default image contrast settings; image data and results stay unchanged."),
        "overlay_mode_combo": ("选择图上叠加哪些测量点或计算曲线，便于与原图对照。", "Choose which measured points or fitted curves to draw over the image for comparison."),
        "global_max_check": ("标出原图最亮的像素，方便定位；它可能是坏点，不一定是有效亮峰。", "Mark the brightest original pixel for reference; it may be a bad pixel rather than a valid peak."),
        "supported_peaks_check": ("标出周围也有亮度变化的峰，方便与最亮像素比较。", "Mark peaks supported by neighbouring intensity changes for comparison with the brightest pixel."),
        "peak_table": ("单击亮峰所在行，在图上定位并查看该峰的亮度曲线。", "Click a peak row to locate it in the image and inspect its intensity profiles."),
        "reset_peak_zoom_button": ("恢复亮峰曲线的完整显示范围，方便重新查看整个峰。", "Show the full peak-profile range again to inspect the whole peak."),
        "quantity_table": ("查看结果及其状态；候选值仅供检查，空白表示尚无可用测量。", "Inspect values and their states; candidates need review, and a blank means no usable measurement is available."),
        "correction_mode_combo": ("选择添加引导点或排除区域，再点“编辑”并在图上操作。", "Choose a guide point or an area to exclude, then press Edit and work on the image."),
        "seed_branch_combo": ("选择引导点属于哪组亮弧；不确定时保留“自动”。", "Choose which arc group the guide point belongs to; keep Auto if unsure."),
        "seed_side_combo": ("选择引导点在图的上半部还是下半部；不确定时选“未知”。", "Choose whether the guide point belongs to the upper or lower part of the image; choose Unknown if unsure."),
        "correct_button": ("进入所选编辑模式；矩形拖动绘制，多边形逐点单击并双击完成。", "Enter the selected edit mode; drag rectangles, or click polygon corners and double-click to finish."),
        "undo_button": ("撤销最近一次手动编辑，清除旧结果；重新识别或评估后查看新结果。", "Undo the latest manual edit and clear the old result; trace or evaluate again for a new result."),
        "redo_button": ("恢复刚撤销的手动编辑，清除旧结果；重新识别或评估后查看新结果。", "Restore the undone edit and clear the old result; trace or evaluate again for a new result."),
        "numeric_edit_button": ("打开坐标输入窗口，用数字添加引导点或绘制排除区域。", "Open coordinate entry to add a guide point or define an excluded area using numbers."),
        "identify_button": ("按当前方法和范围，从原图寻找亮弧并标出测量点。", "Find bright arcs in the original image with the selected method and range, and mark measured points."),
        "apply_batch_button": ("用当前设置立即处理批处理列表中的图像；请先在批处理页添加图像。", "Immediately process the images in the Batch list with the current settings; add images there first."),
        "cancel_button": ("停止正在进行的识别或评估，保留已显示的结果。", "Stop the current tracing or evaluation request and keep the displayed result."),
        "export_button": ("选择保存位置，导出当前亮弧、参数和检查记录，供后续查看。", "Choose where to save the current arcs, parameters and review record for later inspection."),
        "qspace": ("单击测量点或亮峰查看曲线；编辑时可添加引导点或画区域，按 Esc 退出绘制。", "Click a measured point or peak to inspect its profile; in edit mode add guides or draw areas, and press Esc to leave drawing."),
        "ellipse_diagnostic": ("查看测量点偏离椭圆的程度；偏离较大的位置需要检查。", "Inspect how far measured points lie from the ellipse; review large deviations."),
    },
    "_ProfilePanel": {
        "table": ("查看图中曲线对应的数值；空白表示没有数据。", "Inspect the values behind the plotted curves; blanks mean no data."),
        "plot": ("滚轮缩放曲线，拖动移动显示范围；数值不会被改动。", "Scroll to zoom and drag to move the plot; data values stay unchanged."),
    },
}

CATALOG_HELP = {
    "ButterflyWorkbench": {
        "q_min_edit": "tooltip.q_min", "q_max_edit": "tooltip.q_max",
        "display_scale_combo": "tooltip.display_scale", "display_percentile_spin": "tooltip.display_percentile",
        "show_excluded_check": "tooltip.show_excluded_points", "figure_export_button": "tooltip.butterfly_figure",
    },
}

OPTION_HELP = {
    "RefinementMainWindow": {
        "ellipse_preset_combo": {
            "standard": ("让计算按测量点确定椭圆形状，适合先做初步检查。", "Let the measured points determine the ellipse shape for an initial inspection."),
            "flat_ellipse": ("限制短轴相对长轴的比例，用于明显压扁的亮弧；限制会影响结果。", "Limit the short-to-long axis ratio for visibly flattened arcs; the limit affects the result."),
            "very_flat_ellipse": ("使用更小的短轴与长轴比例范围，适合极扁亮弧；结果需结合原图检查。", "Use a smaller short-to-long axis ratio range for very flat arcs; review the result against the image."),
        },
        "ellipse_residual_combo": {
            "sampson": ("用测量点到椭圆的近似距离调整形状，计算较快。", "Adjust the ellipse using an approximate point-to-curve distance for faster fitting."),
            "geometric": ("用测量点到椭圆上最近位置的实际距离调整形状，计算较慢。", "Adjust the ellipse using the distance to the nearest position on the curve; fitting is slower."),
        },
    },
    "ButterflyWorkbench": {
        "trace_method_combo": {
            "annular_peak": ("沿一圈圈圆环比较亮度，找到亮峰并连成亮弧。", "Compare brightness around successive rings to locate peaks and connect them into arcs."),
            "radial_sector": ("沿不同方向从中心向外比较亮度，每个方向独立寻找亮峰。", "Compare brightness outward from the centre, finding peaks independently in each direction."),
            "curvature": ("根据图上亮度变化的弯曲程度寻找亮弧；较依赖平滑设置。", "Find arcs from the curvature of image-intensity changes; this depends more on smoothing settings."),
        },
        "evaluation_resamples_combo": {
            0: ("只计算一次形状，速度最快，不估计重复计算的波动范围。", "Fit the shape once for the fastest result, without estimating variation across repeats."),
            32: ("模拟图像噪声并重复计算 32 次，用来查看结果波动。", "Simulate image noise and repeat the fit 32 times to inspect result variation."),
            128: ("模拟图像噪声并重复计算 128 次，查看更多重复结果，耗时更长。", "Simulate image noise and repeat the fit 128 times for more repeats at a longer run time."),
        },
        "overlay_mode_combo": {
            "measured_only": ("只显示原图，便于检查实际亮度分布。", "Show the original image alone to inspect the measured brightness distribution."),
            "observed_ridges": ("在原图上标出找到的亮弧测量点。", "Mark measured positions along the bright arcs over the image."),
            "geometry_candidate": ("叠加亮弧对应的候选椭圆，检查曲线是否贴近测量点。", "Draw the candidate ellipse to check how closely it follows the measured arc points."),
            "full2d_model": ("显示按全部所用像素计算的模型椭圆；需先运行强度优化。", "Show the ellipse from the pixel-intensity fit; run intensity optimization first."),
            "compare": ("同时显示亮弧椭圆与像素强度模型椭圆，比较两种计算。", "Show both the measured-arc and pixel-intensity ellipses to compare the fits."),
        },
        "correction_mode_combo": {
            "seed": ("单击图像添加引导点，帮助计算定位亮弧；引导点不会补造测量数据。", "Click to add a guide point for locating arcs; guides do not create measured data."),
            "exclude_point": ("单击已有测量点，将它排除出后续计算。", "Click a measured point to exclude it from subsequent fits."),
            "rectangle_exclude": ("在图上拖出矩形，排除矩形内的测量点。", "Drag a rectangle to exclude measured points inside it."),
            "rectangle_include": ("在图上拖出矩形，恢复矩形内先前排除的测量点。", "Drag a rectangle to restore previously excluded measured points inside it."),
            "polygon_exclude": ("依次单击区域顶点，双击完成；区域内的测量点将被排除。", "Click successive corners and double-click to finish; measured points inside are excluded."),
            "polygon_include": ("依次单击区域顶点，双击完成；恢复区域内先前排除的测量点。", "Click successive corners and double-click to finish; previously excluded measured points inside are restored."),
        },
        "seed_branch_combo": {
            -1: ("不指定亮弧组，由计算自行判断。", "Leave the arc group unspecified for the fit to determine."),
            0: ("把引导点指定到 A 组亮弧。", "Assign the guide point to arc group A."),
            1: ("把引导点指定到 B 组亮弧。", "Assign the guide point to arc group B."),
        },
        "seed_side_combo": {
            "unknown": ("不指定上半部或下半部。", "Leave the upper or lower side unspecified."),
            "upper": ("把引导点指定到图的上半部。", "Assign the guide point to the upper part of the image."),
            "lower": ("把引导点指定到图的下半部。", "Assign the guide point to the lower part of the image."),
        },
    },
}

OBJECT_HELP = {
    "ScrollLeftButton": ("向左滚动页签，查看被隐藏的页面入口。", "Scroll tabs to the left to reveal hidden pages."),
    "ScrollRightButton": ("向右滚动页签，查看被隐藏的页面入口。", "Scroll tabs to the right to reveal hidden pages."),
    "workspaceOpenButton": ("选择一张散射图像载入工作区，然后设置分析范围并识别亮弧。", "Choose a scattering image to load, then set the analysis range and trace its bright arcs."),
    "workspaceContext": ("查看当前图像名称、像素尺寸和坐标单位，确认正在分析的输入。", "Check the current image name, pixel dimensions and coordinate units to confirm the input being analysed."),
    "butterflyNumericMode": ("选择添加引导点，或按输入坐标排除、恢复区域内的测量点。", "Choose to add a guide point, or exclude or restore measured points inside the entered area."),
    "butterflyNumericQx": ("输入引导点的水平坐标，单位与图像坐标一致。", "Enter the guide point's horizontal coordinate in the image's coordinate unit."),
    "butterflyNumericQy": ("输入引导点的竖直坐标，单位与图像坐标一致。", "Enter the guide point's vertical coordinate in the image's coordinate unit."),
    "butterflyNumericPoints": ("矩形填两个对角点，多边形至少三个顶点；格式如 0.1,0.2; 0.3,0.4，逗号分隔水平、竖直坐标，分号分隔各点。", "For a rectangle enter two opposite corners; for a polygon enter at least three corners. Use 0.1,0.2; 0.3,0.4 with commas between horizontal and vertical coordinates and semicolons between points."),
    "butterflyNumericApply": ("按输入的坐标添加引导点或修改排除区域，然后关闭窗口。", "Apply the entered guide point or area edit, then close this dialog."),
    "butterflyNumericCancel": ("关闭输入窗口，放弃这次坐标编辑。", "Close this dialog and discard the coordinate edit."),
    "butterflyRailToggle": ("展开或收起图像与测量点列表，为图像留出更多空间。", "Show or hide the image and point lists to make more room for the image."),
    "branchUnknownVisible": ("显示尚未分到某组亮弧的点；取消勾选只隐藏显示。", "Show points not assigned to an arc group; unchecking only hides them."),
}
for _branch in (0, 1):
    for _side, _zh in (("Upper", "上半部"), ("Lower", "下半部")):
        OBJECT_HELP[f"branch{_branch}{_side}Visible"] = (
            f"显示或隐藏 {'A' if _branch == 0 else 'B'} 组亮弧的{_zh}测量点，计算使用的点不变。",
            f"Show or hide {'upper' if _side == 'Upper' else 'lower'} points in arc group {'A' if _branch == 0 else 'B'}; fit inputs stay unchanged.",
        )

TAB_HELP = {
    "RefinementMainWindow": {
        "measurementTableTabs": {
            0: ("查看各个亮瓣的方向与亮度。", "Inspect each bright lobe's direction and intensity."),
            1: ("查看沿亮弧找到的位置，以及哪些点被排除。", "Inspect measured positions along the bright arcs and which points are excluded."),
            2: ("查看亮弧对应的椭圆形状与计算质量。", "Inspect the ellipse describing the arcs and its fit quality."),
            3: ("查看从中心向外方向的亮峰位置和宽度。", "Inspect peak positions and widths outward from the centre."),
        },
    },
    "ButterflyWorkbench": {
        "diagnostics_tabs": {
            0: ("查看所选测量位置的亮度曲线及其与椭圆的偏差。", "Inspect the selected position's intensity profile and distance from the ellipse."),
            1: ("查看所选亮峰在圆周方向和从中心向外方向的亮度曲线。", "Inspect intensity profiles around the ring and outward from the centre for the selected peak."),
        },
    },
    "_ProfilePanel": {
        "view_tabs": {
            0: ("用曲线查看亮度如何随位置变化。", "Inspect how intensity changes with position in a plot."),
            1: ("查看曲线背后的数值，便于逐项核对。", "Inspect the values behind the plotted curves for comparison."),
        },
    },
}

PAGE_HELP = {
    "LamellarPage": ("把已有结果画成薄片状结构的示意图，可调整假设参数；这不是直接测得的三维结构。", "Illustrate existing results as thin-layer structures with adjustable assumptions; this is not a measured 3D structure."),
    "LocalMeasurementPage": ("在原图上选点、线或条带，查看选中位置的亮度与宽度。", "Select points, lines or stripes in the image to inspect local intensity and width."),
    "AzimuthalPage": ("沿选定圆环比较各方向亮度，查看亮瓣方向和角度宽度。", "Compare brightness around a selected ring to inspect lobe directions and angular widths."),
    "Density2DPage": ("在选定方向查看靠近图像中心的亮度变化，并尝试曲线计算。", "Inspect intensity changes near the image centre in a selected direction and try curve fitting."),
}

COLUMN_HELP = {
    "ButterflyWorkbench": {
        "quantity_table": {
            0: ("要查看的量，例如图上椭圆的大小或角度。", "Quantity being inspected, such as image-ellipse size or angle."),
            1: ("当前可用测量值；空白表示尚无可用测量。", "Available measured value; a blank means no usable measurement."),
            2: ("该值是已获得结果、待检查候选，还是无法计算。", "Whether the value is available, a candidate to review, or unavailable."),
            3: ("供检查的候选值，还不能当成确认的测量。", "Candidate value for review; it is not a confirmed measurement."),
            4: ("反复改变模拟噪声并计算得到的波动范围。", "Range of values obtained by repeating the fit with simulated noise changes."),
            5: ("数值不可用或需要检查的具体原因。", "Specific reason a value is unavailable or needs review."),
        },
    },
}
