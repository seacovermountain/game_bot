// src/monster_detector.rs
//! 怪物名字识别模块 - 基于 OpenCV 的白色文字检测
//!
//! 核心思路(基于对实际截图的像素级分析得出):
//! 1. 自动裁掉桌面客户端的系统窗口标题栏(不是游戏内容,不能写死像素)
//! 2. 战斗区域用"比例"而不是绝对像素定义 ROI,排除顶部图标栏/右侧小地图/
//!    底部技能栏聊天框/左侧摇杆按钮这些固定 UI
//! 3. 白色文字阈值不能用标准的 S<30,V>200——实测发现游戏内文字受光照/
//!    火焰特效影响做了抗锯齿混色,大部分笔画根本到不了纯白,必须放宽到
//!    S<50, V>140,否则会大量漏检
//! 4. 闭运算把同一个词的所有笔画粘合成一个连通块,再用尺寸/面积过滤掉
//!    非文字噪点
//!
//! 已知天然被过滤掉、不需要额外处理的元素(颜色本身就不是白色):
//! - 物品掉落名字(金色/紫色,如"飞书残卷")
//! - 系统提示文字(绿色"发现怪物...","自动战斗中...")
//! - 聊天框系统消息(红/蓝底白字——注意聊天框在ROI排除区内,双重保险)
//! - 伤害飘字(红/绿色数字)
//!
//! 尚未处理、需要你根据实际情况决定是否要单独处理的:
//! - 目标/BOSS 信息条(顶部居中,金色文字+进度条,动态出现消失)——
//!   目前策略是让它落在顶部排除区内,如果你的 UI 布局导致它跟战斗区
//!   有重叠,需要单独写规则排除(可以用它固定的"金色文字+红色/彩色
//!   进度条"这个组合特征来识别并单独排除)
//! - 密集刷怪导致名字连通域粘连合并(见 TextBox::looks_merged,可作为
//!   触发"这个框可能是多个名字合并"的信号)
//!
//! 💡 集成说明(相比原始独立脚本做的改动):
//! - 去掉了独立的 `fn main()` + `imread("screenshot.png")` 测试入口,
//!   因为这里是被 `main.rs` 里 `mod monster_detector;` 引入的模块,
//!   不是独立二进制。
//! - 新增 `rgba_bytes_to_bgr_mat` / `detect_monsters_from_rgba`,直接对接
//!   项目里 `util::capture_window()` 吐出来的 RGBA 原始像素(和
//!   `match_icon.rs` 里零拷贝封装 Mat 的方式保持一致),不需要你自己
//!   先存盘再 `imread` 读回来。
//! - `debug_dump_boxes` 保留,方便你在真实截图上肉眼核对 ROI/阈值是否
//!   标定准确。

use opencv::{
    Result,
    core::{self, CV_32S, Mat, Point, Rect, Scalar, Size, Vector},
    imgcodecs, imgproc,
    prelude::*,
};

/// 检测出的候选文字框(原图坐标系)
#[derive(Debug, Clone, Copy)]
pub struct TextBox {
    pub x: i32,
    pub y: i32,
    pub w: i32,
    pub h: i32,
    pub area: i32,
}

impl TextBox {
    pub fn rect(&self) -> Rect {
        Rect::new(self.x, self.y, self.w, self.h)
    }

    /// 粗略判断这个框是不是"疑似多个名字粘连在一起"
    /// 阈值需要你用实际截图标定的"单个名字典型宽度"来调
    pub fn looks_merged(&self, typical_name_width: i32) -> bool {
        self.w > (typical_name_width as f64 * 1.6) as i32
    }

    /// 🎯 文字框中心点,方便后续换算成鼠标点击/锁定目标的屏幕坐标
    pub fn center(&self) -> (i32, i32) {
        (self.x + self.w / 2, self.y + self.h / 2)
    }
}

/// 检测参数配置。所有 ROI 边界用比例(0.0~1.0)表示,而不是绝对像素,
/// 这样同一套配置在不同分辨率的截图上都能用,只需要针对客户端UI布局
/// 标定一次比例。
#[derive(Debug, Clone)]
pub struct DetectorConfig {
    // 战斗区域 ROI,相对于"游戏画面"(已经去掉标题栏之后)的比例
    pub roi_left_frac: f64,
    pub roi_top_frac: f64,
    pub roi_right_frac: f64,
    pub roi_bottom_frac: f64,

    // 白色阈值(HSV)。V 下限务必别用200,抗锯齿边缘像素到不了那么亮
    pub s_max: f64,
    pub v_min: f64,

    // 🌍 跨平台开关:是否需要自动裁掉顶部系统标题栏。
    // - macOS 桌面客户端通常带亮色标题栏,原本的"亮→暗"启发式对它有效。
    // - Windows(尤其深色主题)标题栏可能本身就是暗色，"亮→暗"判定不出来；
    //   同时如果你在 Windows 上用的截图方式本来就只截"客户区"(没有标题栏),
    //   这一步会导致误裁。所以做成开关 + 通用双向跳变检测(见 detect_canvas_top)。
    pub auto_crop_titlebar: bool,

    // 闭运算核大小,用于把同一个词的多个字符/笔画粘合成一个连通块
    pub close_kernel_w: i32,
    pub close_kernel_h: i32,

    // 连通域尺寸过滤(需要按你的分辨率实测标定)
    pub min_h: i32,
    pub max_h: i32,
    pub min_w: i32,
    pub max_w: i32,
    pub min_area: i32,

    // 🎯 宽高比过滤:怪物名字是"宽矮"的横排文字条(通常 w/h >= 1.3~1.5)，
    // 而 UI 图标、头像、按钮边角这类误检大多接近正方形。用这个比值
    // 把接近正方形的噪声块排除掉，不用只靠 ROI 边界死磕。
    pub min_aspect_ratio: f64,

    // 🩹 兜底重试专用的"小闭运算核":有些怪物的名字文字紧贴着贴图上的
    // 高光/金属反光(缝隙只有几像素),用上面 close_kernel_w/h 那组参数
    // 闭运算后,名字会直接跟贴图粘成一大块,尺寸超过 max_w/max_h 被整体
    // 丢弃。与其把全局闭运算核调小(容易把其他本来能正常识别的怪物名字
    // 拆散、影响面更大),不如只对"因为太大而被丢弃"的色块单独重试一次:
    // 回到闭运算之前的原始掩码,只在这个色块范围内用更小的核重新做一次
    // 闭运算+连通域分析,看看能不能拆出一个尺寸合格的文字框。
    pub fallback_close_kernel_w: i32,
    pub fallback_close_kernel_h: i32,
}

impl Default for DetectorConfig {
    fn default() -> Self {
        // 这套比例是按 3264x1864(桌面客户端截图,已去除标题栏后)标定的,
        // 换分辨率/换客户端需要重新用 debug_dump_boxes 跑一遍自己核对
        Self {
            roi_left_frac: 0.08,
            roi_top_frac: 0.20,
            roi_right_frac: 0.68,
            roi_bottom_frac: 0.62,

            auto_crop_titlebar: true,

            s_max: 50.0,
            v_min: 140.0,

            close_kernel_w: 35,
            close_kernel_h: 13,

            min_h: 15,
            max_h: 55,
            min_w: 30,
            max_w: 320,
            min_area: 150,

            min_aspect_ratio: 1.3,

            // 实测(见调试记录):宽 25 / 高 5 能在"名字紧贴怪物贴图高光"的
            // 场景里,把名字从贴图里拆出来,同时还足够把同一行文字的多个
            // 字粘合成一个连通块。
            fallback_close_kernel_w: 25,
            fallback_close_kernel_h: 5,
        }
    }
}

/// 🖼️ 把 `util::capture_window()` 吐出的原始 RGBA 像素零拷贝封装成
/// OpenCV BGR Mat(和 `match_icon::find_icon_opencv` 里的做法保持一致,
/// 只是这里最终要的是 BGR 而不是 RGB,因为后面走的是 BGR2GRAY / BGR2HSV)
pub fn rgba_bytes_to_bgr_mat(raw_rgba: &[u8], width: u32, height: u32) -> Result<Mat> {
    let data_vec4b = unsafe {
        std::slice::from_raw_parts(
            raw_rgba.as_ptr() as *const core::Vec4b,
            (width * height) as usize,
        )
    };
    let src_rgba = Mat::new_rows_cols_with_data(height as i32, width as i32, data_vec4b)?;

    let mut src_bgr = Mat::default();
    imgproc::cvt_color(
        &src_rgba,
        &mut src_bgr,
        imgproc::COLOR_RGBA2BGR,
        0,
        core::AlgorithmHint::ALGO_HINT_DEFAULT,
    )?;

    Ok(src_bgr)
}

/// 自动检测桌面客户端窗口标题栏的下边界。
///
/// 🌍 跨平台版本:原来只认"亮→暗"(macOS 亮色标题栏)这一种模式。
/// 现在改成检测任意方向的显著亮度跳变——不管是 macOS 的"亮标题栏→暗画面"，
/// 还是 Windows 深色主题的"暗标题栏→更暗/更亮画面"，只要相邻两行平均亮度
/// 差值超过阈值,就认定为标题栏和游戏画面的分界线。逐行扫描,找第一次出现
/// 的大幅跳变。
pub fn detect_canvas_top(img: &Mat) -> Result<i32> {
    let gray = {
        let mut g = Mat::default();
        imgproc::cvt_color(
            img,
            &mut g,
            imgproc::COLOR_BGR2GRAY,
            0,
            core::AlgorithmHint::ALGO_HINT_DEFAULT,
        )?;
        g
    };

    let width = img.cols();
    let scan_width = width.min(800); // 只扫左上角一部分,足够判断,速度更快
    let sample_roi = Rect::new(0, 0, scan_width, gray.rows().min(200));
    let region = Mat::roi(&gray, sample_roi)?;

    const JUMP_THRESHOLD: f64 = 60.0;

    let mut prev_mean: Option<f64> = None;
    for y in 0..region.rows() {
        let row = region.row(y)?;
        let mean = core::mean(&row, &core::no_array())?[0];

        if let Some(prev) = prev_mean {
            // 双向跳变都算:|亮度差| 超过阈值即可，不再要求方向必须是"由亮变暗"
            if (prev - mean).abs() > JUMP_THRESHOLD {
                return Ok(y);
            }
        }
        prev_mean = Some(mean);
    }
    Ok(0) // 没检测到标题栏(比如已经是纯游戏画面截图,或者截图方式本来就不含标题栏),不裁
}

/// 主检测函数:输入 BGR Mat,输出怪物名字候选框(原图坐标系)
pub fn detect_monster_names(img: &Mat, cfg: &DetectorConfig) -> Result<Vec<TextBox>> {
    let canvas_top = if cfg.auto_crop_titlebar {
        detect_canvas_top(img)?
    } else {
        0
    };
    let canvas_h = img.rows() - canvas_top;
    let canvas_w = img.cols();

    let roi = Rect::new(
        (canvas_w as f64 * cfg.roi_left_frac) as i32,
        canvas_top + (canvas_h as f64 * cfg.roi_top_frac) as i32,
        (canvas_w as f64 * (cfg.roi_right_frac - cfg.roi_left_frac)) as i32,
        (canvas_h as f64 * (cfg.roi_bottom_frac - cfg.roi_top_frac)) as i32,
    );

    let roi_img = Mat::roi(img, roi)?;

    let mut hsv = Mat::default();
    imgproc::cvt_color(
        &roi_img,
        &mut hsv,
        imgproc::COLOR_BGR2HSV,
        0,
        core::AlgorithmHint::ALGO_HINT_DEFAULT,
    )?;

    let lower = Scalar::new(0.0, 0.0, cfg.v_min, 0.0);
    let upper = Scalar::new(180.0, cfg.s_max, 255.0, 0.0);
    let mut mask = Mat::default();
    core::in_range(&hsv, &lower, &upper, &mut mask)?;

    let kernel = imgproc::get_structuring_element(
        imgproc::MORPH_RECT,
        Size::new(cfg.close_kernel_w, cfg.close_kernel_h),
        Point::new(-1, -1),
    )?;
    let mut closed = Mat::default();
    imgproc::morphology_ex(
        &mask,
        &mut closed,
        imgproc::MORPH_CLOSE,
        &kernel,
        Point::new(-1, -1),
        1,
        core::BORDER_CONSTANT,
        imgproc::morphology_default_border_value()?,
    )?;

    let mut labels = Mat::default();
    let mut stats = Mat::default();
    let mut centroids = Mat::default();
    let num = imgproc::connected_components_with_stats(
        &closed,
        &mut labels,
        &mut stats,
        &mut centroids,
        8,
        CV_32S,
    )?;

    let mut boxes = Vec::new();
    for i in 1..num {
        let x = *stats.at_2d::<i32>(i, imgproc::CC_STAT_LEFT)?;
        let y = *stats.at_2d::<i32>(i, imgproc::CC_STAT_TOP)?;
        let w = *stats.at_2d::<i32>(i, imgproc::CC_STAT_WIDTH)?;
        let h = *stats.at_2d::<i32>(i, imgproc::CC_STAT_HEIGHT)?;
        let area = *stats.at_2d::<i32>(i, imgproc::CC_STAT_AREA)?;

        if h > cfg.min_h && h < cfg.max_h && w > cfg.min_w && w < cfg.max_w && area > cfg.min_area {
            let aspect_ratio = w as f64 / h as f64;
            if aspect_ratio >= cfg.min_aspect_ratio {
                boxes.push(TextBox {
                    x: x + roi.x,
                    y: y + roi.y,
                    w,
                    h,
                    area,
                });
                continue;
            }
        }

        // 🩹 兜底重试:这个色块之所以没通过筛选,是不是"太大"了
        // (超过 max_w 或 max_h)?如果是,很可能是名字文字跟旁边贴图的
        // 高光粘在一起了——回到闭运算之前的原始掩码,只在这个色块的
        // 范围内,用更小的核重新做一次闭运算 + 连通域分析,看看能不能
        // 拆出一个尺寸合格的文字框。(太小/宽高比不对被刷掉的色块,
        // 本来就是噪点,不需要重试。)
        if w >= cfg.max_w || h >= cfg.max_h {
            let recovered =
                recover_oversized_blob(&mask, Rect::new(x, y, w, h), cfg, roi.x, roi.y)?;
            boxes.extend(recovered);
        }
    }

    Ok(boxes)
}

/// 🩹 对"因为太大被判定不合格"的色块做兜底重试:回到闭运算之前的原始
/// 掩码,截取这个色块的范围(留一点边距),用比全局默认小得多的闭运算核
/// 重新做一次闭运算 + 连通域分析,只保留同样通过标准尺寸/宽高比筛选的
/// 结果——避免真的只是"一大片无关噪点"的情况也被硬凑出候选框。
///
/// `roi_x`/`roi_y` 是外层 ROI 在整帧图里的偏移,`blob_rect` 是色块在
/// ROI 坐标系里的位置,返回的 `TextBox` 已经换算回整帧坐标系。
fn recover_oversized_blob(
    raw_mask: &Mat,
    blob_rect: Rect,
    cfg: &DetectorConfig,
    roi_x: i32,
    roi_y: i32,
) -> Result<Vec<TextBox>> {
    const MARGIN: i32 = 4;

    let x = (blob_rect.x - MARGIN).max(0);
    let y = (blob_rect.y - MARGIN).max(0);
    let w = (blob_rect.width + MARGIN * 2).min(raw_mask.cols() - x);
    let h = (blob_rect.height + MARGIN * 2).min(raw_mask.rows() - y);

    if w <= 0 || h <= 0 {
        return Ok(Vec::new());
    }

    let sub_mask = Mat::roi(raw_mask, Rect::new(x, y, w, h))?;

    let small_kernel = imgproc::get_structuring_element(
        imgproc::MORPH_RECT,
        Size::new(cfg.fallback_close_kernel_w, cfg.fallback_close_kernel_h),
        Point::new(-1, -1),
    )?;
    let mut sub_closed = Mat::default();
    imgproc::morphology_ex(
        &sub_mask,
        &mut sub_closed,
        imgproc::MORPH_CLOSE,
        &small_kernel,
        Point::new(-1, -1),
        1,
        core::BORDER_CONSTANT,
        imgproc::morphology_default_border_value()?,
    )?;

    let mut sub_labels = Mat::default();
    let mut sub_stats = Mat::default();
    let mut sub_centroids = Mat::default();
    let sub_num = imgproc::connected_components_with_stats(
        &sub_closed,
        &mut sub_labels,
        &mut sub_stats,
        &mut sub_centroids,
        8,
        CV_32S,
    )?;

    let mut recovered = Vec::new();
    for i in 1..sub_num {
        let sx = *sub_stats.at_2d::<i32>(i, imgproc::CC_STAT_LEFT)?;
        let sy = *sub_stats.at_2d::<i32>(i, imgproc::CC_STAT_TOP)?;
        let sw = *sub_stats.at_2d::<i32>(i, imgproc::CC_STAT_WIDTH)?;
        let sh = *sub_stats.at_2d::<i32>(i, imgproc::CC_STAT_HEIGHT)?;
        let sarea = *sub_stats.at_2d::<i32>(i, imgproc::CC_STAT_AREA)?;

        if sh > cfg.min_h
            && sh < cfg.max_h
            && sw > cfg.min_w
            && sw < cfg.max_w
            && sarea > cfg.min_area
        {
            let aspect_ratio = sw as f64 / sh as f64;
            if aspect_ratio >= cfg.min_aspect_ratio {
                recovered.push(TextBox {
                    x: roi_x + x + sx,
                    y: roi_y + y + sy,
                    w: sw,
                    h: sh,
                    area: sarea,
                });
            }
        }
    }

    Ok(recovered)
}

/// 🎯 一步到位的入口函数:直接吃 `util::capture_window()` 产出的
/// (raw_rgba, width, height),吐出候选怪物名字框。
/// main.rs / game_status.rs 里想接入怪物识别,调这一个函数就够了。
pub fn detect_monsters_from_rgba(
    raw_rgba: &[u8],
    width: u32,
    height: u32,
    cfg: &DetectorConfig,
) -> Result<Vec<TextBox>> {
    let bgr = rgba_bytes_to_bgr_mat(raw_rgba, width, height)?;
    detect_monster_names(&bgr, cfg)
}

/// 调试用:把检测出的框画到原图上并保存,方便你在自己的截图上核对
/// ROI 比例/阈值是否标定准确。强烈建议先跑这个,肉眼确认没问题了
/// 再接入自动化逻辑。
pub fn debug_dump_boxes(img: &Mat, boxes: &[TextBox], out_path: &str) -> Result<()> {
    let mut vis = img.clone();
    for b in boxes {
        imgproc::rectangle(
            &mut vis,
            b.rect(),
            Scalar::new(0.0, 0.0, 255.0, 0.0),
            3,
            imgproc::LINE_8,
            0,
        )?;
    }
    let params = Vector::new();
    imgcodecs::imwrite(out_path, &vis, &params)?;
    Ok(())
}

/// 🗂️ 调试/建模板专用:把每一个候选框单独裁剪成小图存盘。
/// 用途:从真实截图里跑一次检测,把这里存出来的一堆小图肉眼过一遍,
/// 挑出真正是"怪物名字"的那几张,改名成怪物名字(跟 config.toml 里
/// 完全一致),丢进 templates/monster_names/ 目录,就是给
/// monster_matcher 用的模板库了。
///
/// 文件名格式: box_{序号}_{x}_{y}.png,方便你对照 debug_dump_boxes
/// 存出来的标注大图找到对应位置。
pub fn debug_crop_boxes(img: &Mat, boxes: &[TextBox], out_dir: &str) -> Result<()> {
    std::fs::create_dir_all(out_dir).ok();

    for (i, b) in boxes.iter().enumerate() {
        let rect = core::Rect::new(
            b.x.max(0),
            b.y.max(0),
            b.w.min(img.cols() - b.x.max(0)),
            b.h.min(img.rows() - b.y.max(0)),
        );
        if rect.width <= 0 || rect.height <= 0 {
            continue;
        }

        let cropped = Mat::roi(img, rect)?;
        let out_path = format!("{}/box_{:02}_{}_{}.png", out_dir, i, b.x, b.y);
        let params = Vector::new();
        imgcodecs::imwrite(&out_path, &cropped, &params)?;
    }

    println!(
        "   🗂️ [调试] 已把 {} 个候选框单独裁剪存到目录: {}",
        boxes.len(),
        out_dir
    );

    Ok(())
}
