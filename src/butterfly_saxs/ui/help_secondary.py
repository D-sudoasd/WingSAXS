"""Bilingual help for analysis pages and dialogs, without a Qt dependency.

Dynamic LamellarControls combo keys in OPTION_HELP are object names. The
installer resolves an instance attribute first, then a descendant objectName.
Menu action entries in OBJECT_HELP use ``lamellar_`` plus their textKey.
"""

CONTROL_HELP: dict[str, dict[str, tuple[str, str]]] = {
    "LocalMeasurementPage": {
        "export_button": (
            "选择保存位置，导出当前手动测量的数值与来源记录。",
            "Choose a destination to export the current manual measurements and their sources.",
        ),
        "clear_button": (
            "确认后清除手动选点和测量记录，再在图像上重新选取。",
            "Confirm to clear manual selections and measurements, then select new locations on the image.",
        ),
        "mode_combo": (
            "选择点值、线剖面或局部条带；随后在图像上单击选取测量位置。",
            "Choose a point, line profile, or local stripe, then click the image to select measurement locations.",
        ),
        "patch_radius": (
            "设置点周围统计区域的半径，单位为像素；0 只取该像素，较大值汇总更多有效像素。",
            "Set the radius around a point in pixels; 0 uses that pixel alone, while larger values summarize more valid pixels.",
        ),
        "roi_half_width": (
            "设置局部条带两侧的宽度，单位为像素；单击两个端点后，条带内有效强度用于估计主要方向。",
            "Set the stripe half-width in pixels; after clicking two endpoints, valid intensities inside the stripe estimate its main direction.",
        ),
        "scale_combo": (
            "调整图像亮度显示，便于找到弱信号；测量仍使用原始强度。",
            "Adjust image brightness scaling to locate weak signals; measurements still use the original intensities.",
        ),
        "image_plot": (
            "左键单击测点；线剖面和局部条带需要依次单击两个端点。滚轮缩放，拖动平移。",
            "Left-click a measurement point; lines and stripes require two endpoint clicks. Scroll to zoom and drag to pan.",
        ),
        "profile_plot": (
            "选中测量表的一行查看相应强度剖面；滚轮缩放，拖动平移。空缺位置不补造信号。",
            "Select a measurement row to view its intensity profile. Scroll to zoom and drag to pan; missing positions remain unfilled.",
        ),
        "table": (
            "单击一行，在图像上突出对应测量并查看详细结果；此表只读。",
            "Click a row to highlight its measurement on the image and view details; this table is read-only.",
        ),
        "details": (
            "查看选中测量的坐标、强度和单位；可用鼠标选中文字复制。",
            "Inspect coordinates, intensities, and units for the selected measurement; select text with the mouse to copy it.",
        ),
    },
    "AzimuthalPage": {
        "q_min_spin": (
            "q 表示图上距散射中心的尺度；输入所选圆环的内侧界限，使用当前图像单位，须小于上限。",
            "q measures distance from the scattering centre on the image. Enter the ring's inner limit in current image units, below the maximum.",
        ),
        "q_max_spin": (
            "q 表示图上距散射中心的尺度；输入圆环外侧界限，与下限一起选定分析区域，使用当前图像单位。",
            "q measures distance from the scattering centre on the image. Enter the outer ring limit in current image units to set the analysis region.",
        ),
        "bins_spin": (
            "设置一整圈分成多少个角度区间；增加可显示更细角度变化，但每个区间的有效像素可能减少。",
            "Set how many angular bins cover a full circle; more bins resolve finer changes but may contain fewer valid pixels each.",
        ),
        "model_combo": (
            "选择描述角向强度峰的曲线形状；调整后点击“分析环带”重新拟合。",
            "Choose the curve shape for angular intensity peaks, then click Analyze annulus to fit again.",
        ),
        "max_peaks_spin": (
            "限制自动寻找并拟合的峰数量；峰太多时可降低此值，漏掉明显峰时可提高。",
            "Limit the number of peaks detected and fitted; lower it for excessive peaks or raise it when clear peaks are missed.",
        ),
        "min_separation_spin": (
            "设置两个候选峰的最小角度间距；增大可避免把同一个宽峰分成多个峰。",
            "Set the minimum angular separation between candidate peaks; increase it to avoid splitting one broad peak into several.",
        ),
        "height_fraction_spin": (
            "设置自动选峰的相对高度门槛；增大可排除较弱起伏，减小可保留弱峰。",
            "Set the relative height threshold for peak detection; increase it to reject weak fluctuations or decrease it to retain weak peaks.",
        ),
        "initial_width_spin": (
            "输入拟合开始时的峰半高全宽，即峰在半高处的角度宽度；它是起始值，不是固定结果。",
            "Enter the initial full width at half maximum, the angular width at half peak height; it starts the fit rather than fixing the result.",
        ),
        "eta_spin": (
            "设置 pseudo-Voigt 的两种峰形混合比例：0 为圆滑窄尾峰，1 为长尾峰；只在该模型下启用。",
            "Set the fixed pseudo-Voigt mixture: 0 gives a smooth short-tailed peak and 1 a long-tailed peak; enabled only for this model.",
        ),
        "analyze_button": (
            "从所选 q 环带的有效像素计算角向强度，并拟合所选峰模型；同时显示拟合与实测的差值。",
            "Measure angular intensity from valid pixels in the selected q annulus and fit the chosen peak model; also display observed-minus-fit differences.",
        ),
        "export_button": (
            "选择 CSV 保存位置，导出角向强度和峰参数；先完成当前设置的分析。",
            "Choose a CSV destination to export angular intensities and peak parameters; first analyze the current settings.",
        ),
        "canvas": (
            "左图显示实际选中的环带，右图比较实测、拟合和残差；修改上方设置后重新分析。",
            "The left plot shows the selected annulus; the right plots compare observations, fit, and residuals. Change settings above and analyze again.",
        ),
    },
    "Density2DPage": {
        "q_min_spin": (
            "q 表示图上距散射中心的尺度；输入扇形区域的内侧界限，使用当前图像单位，须小于上限。",
            "q measures distance from the scattering centre on the image. Enter the sector's inner limit in current image units, below the maximum.",
        ),
        "q_max_spin": (
            "q 表示图上距散射中心的尺度；输入扇形区域的外侧界限，使用当前图像单位，再重新分析。",
            "q measures distance from the scattering centre on the image. Enter the sector's outer limit in current image units, then analyze again.",
        ),
        "azimuth_spin": (
            "设置扇区中心方向，单位为度；预览中显示所选方向对应的区域。",
            "Set the sector's central direction in degrees; the preview shows the corresponding selected region.",
        ),
        "half_width_spin": (
            "设置中心方向两侧各包含的角度；总开角为此值的两倍，增大可汇总更多像素。",
            "Set the angular extent on each side of the centre; the full opening is twice this value, and larger values include more pixels.",
        ),
        "n_q_spin": (
            "将 q 范围分成此数量的区间并计算均值；分箱越多曲线越细，但空区间也可能增多。",
            "Divide the q range into this many bins and calculate their means; more bins give finer curves but may leave more empty bins.",
        ),
        "power_law_check": (
            "勾选幂律模型，估计靠近图像中心的强度随 q 变化的快慢；结果用于比较曲线形状。",
            "Select the power-law model to estimate how rapidly intensity near the image centre changes with q; use the result to compare curve shapes.",
        ),
        "ornstein_zernike_check": (
            "勾选 Ornstein–Zernike 模型，估计密度起伏延续的距离；只有图像经过物理 q 标定才报告实际长度。",
            "Select the Ornstein–Zernike model to estimate the distance over which density fluctuations persist; physical q calibration is required to report an actual length.",
        ),
        "analyze_button": (
            "提取所选扇区的实测均值，并分别拟合勾选的模型；至少选择一个模型。",
            "Measure mean intensity in the selected sector and fit each selected model separately; choose at least one model.",
        ),
        "export_button": (
            "导出当前扇区的 CSV 强度曲线与 JSON 拟合记录，包括覆盖率、单位和候选状态。",
            "Export the current sector's CSV intensity profile and JSON fit record, including coverage, units, and candidate status.",
        ),
        "canvas": (
            "左图核对所选 q 扇区，右图查看实测曲线、模型候选和残差；空分箱保持空缺。",
            "Check the selected q sector on the left and measured profiles, model candidates, and residuals on the right; empty bins remain missing.",
        ),
    },
    "LamellarPage": {
        "import_button": (
            "打开菜单，选择结果 JSON 或结果文件夹；读取已有分析结果生成薄片状结构示意。",
            "Open the menu to choose a result JSON or result folder; existing analysis results generate a lamellar schematic.",
        ),
        "undo_button": ("撤销上一次薄片状结构示意设置修改。", "Undo the last change to lamellar schematic settings."),
        "redo_button": ("恢复刚撤销的薄片状结构示意设置修改。", "Redo the last undone change to lamellar schematic settings."),
        "settings_button": ("显示或收起右侧示意设置，腾出空间查看图像。", "Show or hide schematic settings on the right to give the views more space."),
        "scattering_button": (
            "把薄片状结构示意投影到二维，用 FFT 预测散射亮度图；打开窗口对比投影与模拟结果。",
            "Project the schematic of thin layers into 2D and use an FFT to predict a scattering brightness map; open the window to compare the projection and simulation.",
        ),
        "publication_button": (
            "打开发表画板，调整薄片状结构示意构图、标注和导出尺寸；图形仍保留参数来源与假设。",
            "Open the publication artboard to adjust schematic layout, annotations, and export size; parameter sources and assumptions are retained.",
        ),
        "export_button": (
            "打开菜单，选择导出当前帧图像与来源，或序列 PNG 和 GIF。",
            "Open the menu to export current-frame figures and provenance, or sequence PNG files and a GIF.",
        ),
        "q_view": (
            "对照观测散射图与拟合几何，检查示意使用的参数来源；左键单击可选中可见的测点或峰标记。",
            "Compare the observed scattering image and fitted geometry to check the schematic's parameter sources; left-click to select a visible point or peak marker.",
        ),
        "canvas": (
            "查看薄片状结构在样品面内的示意投影；将鼠标放在图上滚动可围绕光标缩放。",
            "View the schematic in-plane projection; scroll over the plot to zoom around the pointer.",
        ),
        "view3d": (
            "按住左键拖动旋转，滚轮缩放；三维位置、厚度和排列是条件示意，不是唯一结构重建。",
            "Left-drag to rotate and scroll to zoom; 3D positions, thickness, and arrangement are a conditional schematic, not a unique reconstruction.",
        ),
        "camera_combo": ("选择固定观察方向，便于比较不同帧的薄片状结构示意。", "Choose a standard viewing direction to compare lamellar schematics across frames."),
        "previous_button": ("显示序列中的上一帧，并同步各个视图。", "Show the previous sequence frame and synchronize the views."),
        "play_button": ("开始或暂停逐帧播放；到最后一帧会停止。", "Start or pause frame-by-frame playback; it stops at the last frame."),
        "next_button": ("显示序列中的下一帧，并同步各个视图。", "Show the next sequence frame and synchronize the views."),
        "slider": ("拖动滑块选择帧；上方各视图同步到该帧。", "Drag the slider to select a frame; the views above synchronize to it."),
        "speed": ("选择每秒播放帧数；较低速度便于逐帧检查。", "Choose playback frames per second; lower speeds make individual frames easier to inspect."),
        "sources_button": (
            "查看参数数值、单位、来源、状态及示意假设，区分观测支持、候选和手动设定。",
            "Inspect parameter values, units, sources, status, and schematic assumptions to distinguish supported observations, candidates, and manual settings.",
        ),
        "trajectory": (
            "查看序列中的表观周期变化；单击对应帧位置切换，缺失结果保持空缺。",
            "Inspect apparent period across the sequence; click a frame position to select it, while missing results remain gaps.",
        ),
        "cancel_button": ("请求停止当前读取、更新或导出任务。", "Request cancellation of the current loading, update, or export task."),
    },
    "LamellarControls": {
        "single_preview_button": (
            "预览一组成组叠放的薄片状结构（片层堆栈）；没有可用测量时使用相对尺度，可继续调整。",
            "Preview one group of stacked thin layers (a lamellar packet); without usable measurements it uses relative units, and you can adjust the settings.",
        ),
        "meso_preview_button": (
            "预览多组成组叠放的薄片状结构（片层堆栈）；没有可用测量时使用相对尺度。",
            "Preview several groups of stacked thin layers (lamellar packets); without usable measurements it uses relative units.",
        ),
        "advanced_toggle": ("展开或收起深度、随机起伏和出平面角度等示意假设。", "Expand or collapse schematic assumptions for depth, random variation, and out-of-plane angle."),
        "palette_combo": ("选择不同方向亮弧对应的薄片状结构示意配色，仅改变外观。", "Choose colours for the thin-layer groups corresponding to bright arcs in different directions; this changes appearance only."),
        "reset_button": (
            "恢复默认示意设置；有可用结果时自动选择可用的周期来源。",
            "Restore default schematic settings; when results are usable, select an available period source automatically.",
        ),
    },
    "PublicationDialog": {
        "export_button": (
            "选择输出位置，保存当前发表图及参数来源；先等待预览完成。",
            "Choose an output destination to save the current publication figure and parameter sources; wait for the preview to finish first.",
        ),
        "template_combo": ("选择只展示结构示意，或同时展示 SAXS 证据与结构示意。", "Choose a structure schematic alone, or SAXS evidence alongside the schematic."),
        "width_preset": ("选择单栏、双栏或自定义画板宽度；预设会同时调整高度。", "Choose single-column, double-column, or custom artboard width; presets also adjust height."),
        "width_spin": ("输入导出画板宽度，单位为毫米；与高度一起确定构图比例。", "Enter exported artboard width in millimetres; together with height it defines the composition ratio."),
        "height_spin": ("输入导出画板高度，单位为毫米；预览按新比例更新。", "Enter exported artboard height in millimetres; the preview updates to the new ratio."),
        "dpi_combo": ("选择每英寸像素数；更高分辨率让位图更清晰，也增加计算和文件大小。", "Choose pixels per inch; higher resolution gives sharper raster images and increases computation and file size."),
        "language_combo": ("选择导出图内的文字语言，预览随之更新。", "Choose the text language inside the exported figure; the preview updates accordingly."),
        "palette_combo": ("选择发表图的冷暖配色或灰度配色。", "Choose a warm/cool or grayscale palette for the publication figure."),
        "background_combo": ("选择白色或透明背景，便于后续排版。", "Choose a white or transparent background for later layout work."),
        "bevel_spin": ("调整薄片边缘倒角的示意比例，使边缘更易辨认；不改变测得参数。", "Adjust the schematic edge-bevel fraction to make edges clearer; measured parameters remain unchanged."),
        "roughness_spin": ("调整三维表面的哑光程度；数值越大，表面反光越弱。", "Adjust the 3D surface's matte appearance; larger values reduce reflections."),
        "inset_check": ("勾选后加入层间关系特写，便于辨认周期和厚度示意。", "Select to add a close-up of interlayer relationships and clarify schematic period and thickness."),
        "labels_check": ("显示或隐藏参数与法向标注；法向是垂直于薄片表面的方向。", "Show or hide parameter and normal-direction labels; a normal is perpendicular to the lamellar surface."),
        "ao_check": ("添加柔和层间阴影，帮助区分遮挡关系；只改变显示。", "Add soft shadows between layers to clarify occlusion; this changes appearance only."),
        "reset_labels_button": ("将拖动过的标注恢复到自动位置。", "Restore dragged annotations to their automatic positions."),
        "auto_camera_button": ("自动选择平衡视角并取景，使当前示意完整显示。", "Choose a balanced viewing angle and automatic framing to show the current schematic fully."),
        "use_camera_button": ("采用片层工作台当前三维视角，并更新发表预览。", "Use the lamellar workbench's current 3D viewing angle and update the publication preview."),
        "canvas": (
            "按住左键拖动标注调整位置，释放后保存位置；导出保留图形来源与示意假设。",
            "Left-drag annotations to reposition them and release to save their positions; exports retain figure sources and schematic assumptions.",
        ),
        "cancel_button": ("请求取消正在生成的预览或导出任务。", "Request cancellation of the current preview or export task."),
        "close_button": ("关闭发表画板，返回片层工作台。", "Close the publication artboard and return to the lamellar workbench."),
    },
    "FigureExportDialog": {
        "parent_dir_edit": ("输入已存在的输出父目录；程序在其中建立独立的图包目录。", "Enter an existing output parent folder; the program creates a separate figure-package folder inside it."),
        "browse_button": ("浏览文件夹并选择图包的输出父目录。", "Browse folders and choose the output parent folder for the figure package."),
        "width_combo": ("选择图的单栏或双栏宽度，按毫米确定导出尺寸。", "Choose single-column or double-column figure width to set the export size in millimetres."),
        "dpi_combo": ("选择位图分辨率；更高 dpi 增加像素数、导出时间和文件大小。", "Choose raster resolution; higher dpi increases pixel count, export time, and file size."),
        "start_button": ("按当前设置生成图包，包含图像、数据和来源记录；完成后可打开图包首页。", "Generate a figure package with images, data, and provenance using current settings; open its index when complete."),
        "open_button": ("打开已导出图包的首页；若标为旧快照，它对应导出时的数据与设置。", "Open the exported package index; if marked as an older snapshot, it represents the data and settings at export time."),
        "cancel_button": ("正在导出时请求取消；空闲时关闭窗口。", "Request cancellation during export; close the window when idle."),
        "paper_preview": ("查看画板比例示意；这是尺寸预览，不是最终分析图。", "Inspect the artboard aspect-ratio sketch; it previews size rather than the final analysis figure."),
        "snapshot_label": ("核对图包对应的数据快照；改变当前分析不会修改已经导出的图包。", "Check the data snapshot represented by the package; changing the current analysis does not change an exported package."),
    },
    "LamellarScatteringDialog": {
        "rows": ("设置投影网格行数；增加可更细地表示薄片状结构，也会增加计算量。", "Set the projection grid's row count; more rows resolve slabs more finely and increase computation."),
        "columns": ("设置投影网格列数；修改后点击重新计算查看新的模拟结果。", "Set the projection grid's column count; click Recalculate after changing it to view the new simulation."),
        "margin": ("设置结构外侧留白相对于总宽度的比例；改变留白会影响 FFT 预测亮度图区分相邻 q 值的能力。", "Set the margin around the structure as a fraction of its span; changing it affects how finely the FFT-predicted brightness map resolves neighbouring q values."),
        "calculate_button": ("将薄片状结构沿出平面方向投影到二维，再用 FFT 计算预测散射亮度图；这是二维投影模拟。", "Project the thin-layer schematic into 2D along the out-of-plane direction, then use an FFT to calculate a predicted scattering brightness map; this is a 2D projection simulation."),
        "export_button": ("选择位置，保存结构二维投影、FFT 预测亮度图和模拟记录；不会覆盖现有文件。", "Choose a destination to save the 2D structural projection, FFT-predicted brightness map, and simulation records; existing files are not overwritten."),
        "canvas": ("左图显示结构投影后各处材料的多少，右图显示 FFT 预测的相对散射亮度；颜色压缩亮度范围以显示弱信号。", "The left plot shows the amount of material in the structural projection; the right shows relative scattering brightness predicted by an FFT. Colour scaling compresses brightness to reveal weak signals."),
        "status": ("查看网格分辨率和警告；可选中文字复制，网格过粗时增大行列数后重新计算。", "Inspect grid resolution and warnings; select text to copy it, and increase rows and columns and recalculate if the grid is too coarse."),
    },
    "LegacySaxsDialog": {
        "open_file_button": ("选择旧版 SAXS 文件，查看能读取的数据、标定与兼容性信息；原文件保持不变。", "Choose a legacy SAXS file to inspect readable data, calibration, and compatibility information; the original file remains unchanged."),
        "open_folder_button": ("选择旧版结果文件夹，检查内容并显示兼容性摘要。", "Choose a legacy result folder to inspect its contents and display a compatibility summary."),
        "preview": ("滚动查看兼容性检查记录；内容只读，可选择文字复制。", "Scroll through the compatibility inspection record; it is read-only and you can select text to copy it."),
        "load_image_button": ("读取具备受支持标定和图像的旧版会话，送入当前工作台；不会直接导入旧测量结果。", "Load a legacy session with a supported calibration and image into the current workbench; legacy measurements are not imported directly."),
        "export_button": ("选择位置保存 JSON 兼容性记录，保留检查来源与警告。", "Choose a destination to save a JSON compatibility receipt with inspection sources and warnings."),
        "close_button": ("关闭兼容性窗口，返回当前工作台。", "Close the compatibility window and return to the current workbench."),
    },
}

OBJECT_HELP: dict[str, tuple[str, str]] = {
    "lamellar_mode": ("片层是薄片状结构，堆栈是一组叠放的片层；选择显示一组或多组的示意排列。", "Lamellae are thin layers; a packet is a group of stacked layers. Choose a schematic arrangement of one or several groups."),
    "lamellar_period_source": ("选择从亮环峰位置、拟合椭圆或手动数值设定薄片间重复距离（周期）；实际长度需要物理 q 标定。", "Use a bright ring peak, a fitted ellipse, or a manual value for the repeating distance between layers (period); actual lengths require physical q calibration."),
    "lamellar_selected_branch": ("分支指图上不同方向的亮弧；选择全部或其中一组，分别查看对应的薄片状结构示意。", "Branches are bright arcs in different directions on the image. Select all or one group to inspect the corresponding thin-layer schematic."),
    "lamellar_thickness_ratio": ("设置薄片厚度占薄片间重复距离（周期）的比例；厚度比例是示意假设。", "Set thin-layer thickness as a fraction of the repeating distance between layers (period); this ratio is a schematic assumption."),
    "lamellar_width_ratio": ("设置薄片宽度为薄片间重复距离（周期）的多少倍，改变示意横向大小。", "Set thin-layer width as a multiple of the repeating distance between layers (period) to change the schematic width."),
    "lamellar_depth_ratio": ("设置薄片深度为薄片间重复距离（周期）的多少倍；此三维尺寸是示意假设。", "Set thin-layer depth as a multiple of the repeating distance between layers (period); this 3D dimension is a schematic assumption."),
    "lamellar_layer_count": ("设置每组成组叠放的结构包含多少薄片，即每堆栈层数；增大后示意更高。", "Set the number of thin layers in each stacked group (packet); more layers make the schematic taller."),
    "lamellar_stack_count": ("设置薄片成组叠放后共有多少组（堆栈）；仅多堆栈模式启用。", "Set how many groups of stacked thin layers (packets) to show; enabled only in multiple-packet mode."),
    "lamellar_spread_deg": ("设置各组成组叠放的薄片（堆栈）方向差异，单位为度；数值越大方向越分散。", "Set differences in direction between groups of stacked layers (packets), in degrees; larger values give a wider spread."),
    "lamellar_manual_period": ("输入手动示意的周期，使用下方设定的单位；此值不代表测量结果。", "Enter the manual schematic period in the units selected below; this value is not a measurement."),
    "lamellar_manual_angle_deg": ("设置手动片层法向角度，即垂直于薄片表面的方向，单位为度。", "Set the manual lamellar normal angle, the direction perpendicular to the layer surface, in degrees."),
    "lamellar_manual_second_orientation": ("加入第二组假设的薄片方向；勾选后可编辑垂直于薄片表面的方向（法向）角度。", "Add a second assumed group of thin layers; select to edit the angle perpendicular to their surfaces (the normal)."),
    "lamellar_manual_second_angle_deg": ("设置第二组薄片的垂直方向（法向）角度；需先勾选加入第二方向。", "Set the angle perpendicular to the second group of thin layers (its normal); first enable the second assumed direction."),
    "lamellar_manual_unit": ("选择手动假设的长度单位；选择 nm 表示主动设定纳米尺度，不是由像素 q 换算。", "Choose the units for assumed manual lengths; nm specifies an assumed nanometre scale, not a conversion from pixel q."),
    "lamellar_spacing_jitter_pct": ("设置相邻层间距围绕周期的随机偏差上限；最小间距受厚度约束，它不是峰位拟合误差。", "Set the maximum random deviation of adjacent layer spacing around the period; thickness constrains the minimum spacing, and this is not fitted peak-position error."),
    "lamellar_position_jitter_pct": ("设置成组叠放的薄片（堆栈）在预留间隙内随机移动的比例；增大使排列更不规则。", "Set how far groups of stacked layers (packets) move randomly within reserved gaps; larger values make their arrangement less regular."),
    "lamellar_lateral_shift_ratio": ("设置相邻薄片横向错移为周期的多少倍；正负值改变错移方向。", "Set lateral slip between adjacent layers as a multiple of period; positive and negative values select opposite slip directions."),
    "lamellar_out_of_plane_deg": ("设置垂直于薄片表面的方向（法向）偏离样品平面的角度，查看假设的三维倾斜。", "Set how far the direction perpendicular to a thin layer (its normal) tilts away from the sample plane to inspect the assumed 3D tilt."),
    "lamellar_seed": ("设置随机种子；相同参数和种子可重复生成相同的示意排列。", "Set the random seed; identical settings and seed reproduce the same schematic arrangement."),
    "lamellar_focus": ("放大当前视图；再次点击恢复多视图布局。", "Enlarge this view; click again to restore the multiple-view layout."),
    "lamellar_reset_camera": ("复位三维相机的方向和缩放。", "Reset the 3D camera direction and zoom."),
    "lamellar_file": ("选择结果 JSON 读取已有参数和来源，用于生成薄片状结构示意。", "Choose a result JSON to read existing parameters and sources for a lamellar schematic."),
    "lamellar_folder": ("选择结果文件夹，载入可用结果组成帧序列。", "Choose a result folder to load available results as a frame sequence."),
    "lamellar_export_one": ("导出当前帧的示意图片与来源记录。", "Export the current frame's schematic figures and provenance."),
    "lamellar_export_sequence": ("导出序列 PNG 和 GIF，保留逐帧来源记录。", "Export sequence PNG files and a GIF, retaining frame-specific provenance."),
    "lamellarScatteringClose": ("关闭模拟 FFT 窗口，返回片层工作台。", "Close the simulated FFT window and return to the lamellar workbench."),
    "lamellarSourcesTabs": ("切换查看参数与假设，或完整来源记录。", "Switch between parameters and assumptions, and the complete provenance record."),
    "lamellarSourcesTree": ("逐行核对参数数值、单位、来源和状态；悬停数值查看可用区间或原因，此表只读。", "Check each parameter's value, units, source, and status; hover over a value for an available interval or reason. This table is read-only."),
    "lamellarSourcesManifest": ("查看薄片状结构示意采用的假设；可滚动阅读并选中文字复制。", "Inspect assumptions used in the lamellar schematic; scroll to read and select text to copy it."),
    "lamellarSourcesRecord": ("查看完整 JSON 来源记录，包含参数状态和诊断；内容只读，可选中文字复制。", "Inspect the complete JSON provenance record, including parameter status and diagnostics; it is read-only and you can select text to copy it."),
    "lamellarSourcesClose": ("关闭参数来源窗口，返回片层工作台。", "Close the parameter-source window and return to the lamellar workbench."),
}

COLUMN_HELP: dict[str, dict[str, dict[int, tuple[str, str]]]] = {
    "LocalMeasurementPage": {
        "table": {
            0: ("测量方式：点值、线剖面或局部条带方向。", "Measurement type: point, line profile, or local stripe orientation."),
            1: ("该测量对应的原始帧，选点不会自动转移到其他帧。", "The source frame for this measurement; selections are not automatically transferred to another frame."),
            2: ("点测量显示 q；线剖面显示 FWHM（峰在半高处的全宽）；条带显示主要方向。单击行查看单位与详情。", "Points show q; lines show FWHM (full peak width at half height); stripes show their main direction. Click a row for units and details."),
            3: ("点测量显示强度；线剖面显示有效点数/总点数；条带显示像素数与计算状态。", "Points show intensity; lines show valid/total sample counts; stripes show pixel count and calculation status."),
        },
    },
    "QDialog": {
        "lamellarSourcesTree": {
            0: ("薄片状结构示意使用的参数名称。", "The parameter used by the lamellar schematic."),
            1: ("参数数值；破折号表示未确定，括号内可能显示候选值。悬停查看区间或原因。", "Parameter value; a dash means undetermined and parentheses may show a candidate value. Hover for an interval or reason."),
            2: ("该数值的单位；相对单位不表示实际长度。", "Units for this value; relative units do not represent physical lengths."),
            3: ("说明参数来自散射分析、派生关系还是手动设定。", "Indicates whether the parameter comes from scattering analysis, a derived relation, or a manual setting."),
            4: ("区分有观测支持、候选、示意设定和不可用参数。", "Distinguishes observation-supported, candidate, assumed, and unavailable parameters."),
        },
    },
}

TAB_HELP: dict[str, dict[str, dict[int, tuple[str, str]]]] = {
    "QDialog": {
        "lamellarSourcesTabs": {
            0: ("查看参数来源表和当前薄片状结构示意的假设。", "Inspect the parameter-source table and assumptions for the current lamellar schematic."),
            1: ("查看完整 JSON 来源记录，可选中文字复制。", "Inspect the complete JSON provenance record and select text to copy it."),
        },
    },
}

OPTION_HELP: dict[str, dict[str, dict[object, tuple[str, str]]]] = {
    "LocalMeasurementPage": {
        "mode_combo": {
            "point": ("左键单击一个像素，读取该点和周围有效像素的强度。", "Left-click a pixel to read its intensity and statistics for surrounding valid pixels."),
            "line": ("依次左键单击两个端点，显示连线上的实测强度剖面。", "Left-click two endpoints in sequence to display measured intensity along the line."),
            "roi": ("依次左键单击两个端点，用条带内强度估计局部主要方向。", "Left-click two endpoints in sequence to estimate the local main direction from stripe intensities."),
        },
        "scale_combo": {
            "linear": ("亮度随强度线性变化，便于直接比较强弱。", "Brightness varies linearly with intensity for direct strength comparisons."),
            "log": ("压缩较强信号的亮度范围，让较弱信号更容易看见。", "Compress the brightness range of strong signals to make weaker signals easier to see."),
            "asinh": ("用反双曲正弦压缩亮度范围，也能显示零值和负值。", "Use inverse-hyperbolic-sine scaling to compress brightness while displaying zero and negative values."),
        },
    },
    "AzimuthalPage": {
        "model_combo": {
            "pseudo_voigt": ("混合圆滑窄尾与长尾两种峰形，可调整两者比例。", "Mix smooth short-tailed and long-tailed peak shapes, with an adjustable proportion."),
            "gaussian": ("使用圆滑的峰形；远离峰中心后强度快速减弱，峰尾较短。", "Use a smooth peak shape whose intensity falls rapidly away from the centre, giving short tails."),
            "lorentzian": ("使用长尾峰形；远离峰中心后强度仍缓慢减弱。", "Use a long-tailed peak shape whose intensity falls slowly away from the centre."),
            "von_mises": ("使用适合圆周角度的周期峰形，跨越一整圈边界仍保持连续。", "Use a periodic peak shape for circular angles that stays continuous across a full-circle boundary."),
        },
    },
    "LamellarPage": {
        "camera_combo": {
            "isometric": ("从斜上方观察，便于同时看到三个方向。", "View obliquely from above to see three directions at once."),
            "front": ("从正面观察薄片状结构示意。", "View the lamellar schematic from the front."),
            "side": ("从侧面观察薄片状结构示意。", "View the lamellar schematic from the side."),
            "top": ("沿出平面方向俯看薄片状结构示意。", "Look down along the out-of-plane direction at the lamellar schematic."),
        },
        "speed": {
            1: ("每秒播放 1 帧，便于详细检查。", "Play 1 frame per second for detailed inspection."),
            2: ("每秒播放 2 帧，缓慢比较连续变化。", "Play 2 frames per second for slow comparisons of successive changes."),
            5: ("每秒播放 5 帧，观察序列演化。", "Play 5 frames per second to inspect sequence evolution."),
            10: ("每秒播放 10 帧，快速浏览序列。", "Play 10 frames per second to browse the sequence quickly."),
        },
    },
    "LamellarControls": {
        "lamellar_mode": {
            "single": ("显示一组成组叠放的薄片状结构，便于看清层间关系。", "Show one group of stacked thin layers to clarify interlayer relationships."),
            "multi": ("显示多组成组叠放的薄片状结构；可调整组数与方向差异。", "Show several groups of stacked thin layers; adjust their number and differences in direction."),
        },
        "lamellar_period_source": {
            "radial": ("从亮环距中心的峰位置 q*，用 2π/q* 估计重复距离；实际长度需要物理 q 标定。", "Use the bright ring's peak distance q* from the centre to estimate repeating spacing as 2π/q*; actual lengths require physical q calibration."),
            "ellipse": ("从拟合椭圆估计重复距离，生成薄片状结构示意；同一椭圆可能对应不同三维结构。", "Estimate repeating spacing from the fitted ellipse to generate a thin-layer schematic; the same ellipse may correspond to different 3D structures."),
            "manual": ("手动设置薄片间重复距离（周期）和垂直于薄片表面的方向（法向），查看假设结构。", "Set the repeating distance between layers (period) and direction perpendicular to their surfaces (normal) manually to inspect an assumed structure."),
        },
        "lamellar_selected_branch": {
            -1: ("同时显示图上不同方向亮弧对应的全部示意组。", "Show all schematic groups corresponding to bright arcs in different image directions."),
            0: ("仅显示 A 组亮弧对应的薄片状结构示意。", "Show only the thin-layer schematic corresponding to bright-arc group A."),
            1: ("仅显示 B 组亮弧对应的薄片状结构示意。", "Show only the thin-layer schematic corresponding to bright-arc group B."),
        },
        "lamellar_manual_unit": {
            "relative": ("使用相对长度，只比较比例，不表示实际纳米尺寸。", "Use relative lengths to compare proportions without assigning physical nanometre dimensions."),
            "nm": ("将手动输入解释为假设的纳米尺度，不替代实测标定。", "Interpret manual inputs as assumed nanometre dimensions; this does not replace measured calibration."),
        },
        "palette_combo": {
            "blue_orange": ("用蓝色和橙色区分不同方向亮弧对应的示意组。", "Use blue and orange to distinguish schematic groups corresponding to differently directed bright arcs."),
            "grayscale": ("使用灰度显示，便于黑白输出。", "Use grayscale for monochrome output."),
        },
    },
    "PublicationDialog": {
        "template_combo": {
            "structure": ("以薄片状结构示意为主图，突出层间关系。", "Use the lamellar structure schematic as the main figure to highlight interlayer relationships."),
            "evidence": ("同时展示 SAXS 观测与结构示意，方便对照参数来源。", "Show SAXS observations alongside the structure schematic to compare parameter sources."),
        },
        "width_preset": {
            183.0: ("采用 183 mm 双栏宽度并匹配当前构图高度。", "Use a 183 mm double-column width with height matched to the current composition."),
            89.0: ("采用 89 mm 单栏宽度并匹配当前构图高度。", "Use an 89 mm single-column width with height matched to the current composition."),
            0.0: ("自行输入画板宽度和高度，单位为毫米。", "Enter custom artboard width and height in millimetres."),
        },
        "dpi_combo": {
            600: ("以 600 dpi 导出位图。", "Export raster images at 600 dpi."),
            450: ("以 450 dpi 导出，像素数少于 600 dpi。", "Export at 450 dpi with fewer pixels than 600 dpi."),
            1200: ("以 1200 dpi 导出，细节更多，计算和文件更大。", "Export at 1200 dpi for more detail, computation, and larger files."),
        },
        "language_combo": {
            "en": ("导出图内采用英文标注。", "Use English annotations inside the exported figure."),
            "zh_CN": ("导出图内采用中文标注。", "Use Chinese annotations inside the exported figure."),
        },
        "palette_combo": {
            "editorial": ("用冷暖哑光颜色区分不同方向亮弧对应的薄片状结构示意。", "Use matte warm and cool colours to distinguish thin-layer schematics corresponding to differently directed bright arcs."),
            "grayscale": ("使用灰度配色，便于黑白印刷。", "Use a grayscale palette for monochrome printing."),
        },
        "background_combo": {
            "white": ("输出白色背景，适合直接放入白色页面。", "Export a white background for direct placement on white pages."),
            "transparent": ("输出透明背景，便于叠放到排版页面。", "Export a transparent background for placement over layout pages."),
        },
    },
    "FigureExportDialog": {
        "width_combo": {
            89.0: ("采用 89 mm 单栏图宽。", "Use an 89 mm single-column figure width."),
            183.0: ("采用 183 mm 双栏图宽。", "Use a 183 mm double-column figure width."),
        },
        "dpi_combo": {
            300: ("以 300 dpi 输出位图，像素数和文件较少。", "Export raster images at 300 dpi with fewer pixels and smaller files."),
            600: ("以 600 dpi 输出位图，显示更细的曲线与文字。", "Export raster images at 600 dpi for finer curves and text."),
            1200: ("以 1200 dpi 输出位图，计算与文件大小相应增加。", "Export raster images at 1200 dpi with increased computation and file size."),
        },
    },
}
