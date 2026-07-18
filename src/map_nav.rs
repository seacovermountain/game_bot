// src/map_nav.rs
//! 大地图导航模块 - 复用游戏引擎自带的"点地图自动寻路"能力,
//! 而不是自己实现像素级 A* 寻路。
//!
//! 💡 设计动机(源于实测确认的游戏行为):
//! - 大地图面板打开后,黑色部分是不可行走区域,灰色(带石纹理)部分是
//!   可行走走廊/房间。
//! - 在这个面板里点击任意一个坐标,游戏引擎会自己算出一条寻路路线
//!   (画面上会出现青色虚线),角色随后自动沿这条路线走过去 —— 这意味着
//!   我们完全不需要自己写黑白连通域寻路算法,只需要:
//!     1. 打开大地图面板
//!     2. 在画布的"灰色可行走"区域里随机选一个像素点
//!     3. 点击这个点,触发引擎自动寻路
//!     4. 关闭面板,回到游戏主界面(需要手动点关闭按钮,不关闭会挡住
//!        后台的战斗/移动判断这两个逻辑)
//! - 随机选点不需要知道"世界坐标"是多少,因为点击的是这个面板本身的
//!   像素位置,坐标换算完全是引擎内部的事,我们只要保证点在"灰色"区域
//!   (避免点到黑色不可行走区域导致无法寻路/无反应)即可。
//!
//! ⚠️ 目前 ROI/点击位置几个关键数值是基于一次实际截图(3000x1716)反推
//! 出来的比例,理论上跨分辨率通用,但强烈建议先跑一轮实测确认:
//! - 打开地图的点击位置:大概估算(罗盘图标本身很大,点中间大部分位置
//!   都能触发,容错度较高,但仍建议先用小范围测试确认)。
//! - 画布 ROI 边界:已经用实际截图精确量出来了,应该比较准。
//! - 关闭按钮:用图标模板匹配定位,不依赖固定坐标,更稳。

use crate::match_icon;
use crate::mouse_action;
use crate::util;
use enigo::Enigo;
use opencv::{
    Result,
    core::{self, Mat, Rect},
    imgproc,
    prelude::*,
};
use rand::Rng;
use std::{thread, time::Duration};
use xcap::Window;

#[derive(Debug, Clone)]
pub struct NavConfig {
    // 🎯 打开大地图的点击位置(相对窗口宽高的比例)。
    // ⚠️ 目测估算值,罗盘图标本体较大、容错度高,但仍建议实测确认/微调。
    pub open_click_x_frac: f64,
    pub open_click_y_frac: f64,

    // 🗺️ 大地图面板打开后,黑白可行走画布的 ROI(相对窗口比例)。
    // ✅ 已用实际截图(3000x1716)精确量出边界:
    // x:[435,2087] y:[575,1405] -> 换算成比例。
    pub canvas_left_frac: f64,
    pub canvas_top_frac: f64,
    pub canvas_right_frac: f64,
    pub canvas_bottom_frac: f64,

    // 🎨 可行走区域(灰色走廊)的灰度亮度下限。低于这个值判定为黑色
    // 不可行走区域。画布背景纯黑接近 0,走廊纹理灰度普遍在 10~90+，
    // 用一个比较低的下限即可把两者分开。
    pub walkable_brightness_min: f64,

    // 🧹 形态学开运算核大小,用于过滤画布里装饰性的小光点/噪点
    // (孤立的几像素小亮点,不代表真正可行走的走廊)。
    pub denoise_kernel_size: i32,

    // 🎯 额外向内收缩的腐蚀核大小。靠近可行走区域边界的像素点点击
    // 经常没反应(大概率是判定区域比视觉上的灰色区域小一圈),
    // 所以选点前再做一次腐蚀,把候选范围往区域内部收缩一些。
    pub erode_margin_kernel_size: i32,

    // 🔘 关闭按钮图标模板路径 + 最低置信度。用图标匹配定位,
    // 不依赖固定坐标,面板弹出位置万一有偏移也不受影响。
    pub close_button_template: String,
    pub close_button_min_confidence: f32,
}

impl Default for NavConfig {
    fn default() -> Self {
        Self {
            // TODO: 目测估算,建议实测确认这个点击位置能不能稳定打开地图
            open_click_x_frac: 0.90,
            open_click_y_frac: 0.16,

            // ✅ 已用 DEBUG 截图精确量出(3000x1716 分辨率下 x:[435,2087] y:[575,1405])
            canvas_left_frac: 0.145,
            canvas_top_frac: 0.335,
            canvas_right_frac: 0.696,
            canvas_bottom_frac: 0.819,

            walkable_brightness_min: 10.0,
            denoise_kernel_size: 7,
            // ⚠️ 需要实测调整:数值越大,选出的点离边界越远、越保险,
            // 但如果走廊本身很窄,调太大可能导致整条走廊都被腐蚀没了。
            erode_margin_kernel_size: 15,

            close_button_template: util::get_secure_template_path("close_map.png"),
            close_button_min_confidence: 0.5,
        }
    }
}

/// 把 canvas ROI 配置换算成原图(物理像素)坐标系下的绝对矩形
fn compute_canvas_rect(frame_w: i32, frame_h: i32, cfg: &NavConfig) -> Rect {
    let x = (frame_w as f64 * cfg.canvas_left_frac) as i32;
    let y = (frame_h as f64 * cfg.canvas_top_frac) as i32;
    let w = (frame_w as f64 * (cfg.canvas_right_frac - cfg.canvas_left_frac)) as i32;
    let h = (frame_h as f64 * (cfg.canvas_bottom_frac - cfg.canvas_top_frac)) as i32;
    Rect::new(x, y, w.max(1), h.max(1))
}

/// 🌍 跟 asset_loader::compute_scale_factor 是同一套逻辑:算物理像素
/// 与窗口逻辑尺寸之间的缩放系数,方便把"截图里量出来的像素坐标"换算成
/// "鼠标点击需要的绝对屏幕坐标"。
fn compute_scale_factor(window: &Window, physical_w: u32, physical_h: u32) -> (f32, f32) {
    let logical_w = window.width().max(1) as f32;
    let logical_h = window.height().max(1) as f32;

    let scale_x = physical_w as f32 / logical_w;
    let scale_y = physical_h as f32 / logical_h;

    if scale_x.is_finite() && scale_x > 0.1 && scale_y.is_finite() && scale_y > 0.1 {
        (scale_x, scale_y)
    } else {
        let fallback = match_icon::get_screen_scale_factor();
        (fallback, fallback)
    }
}

/// 🎲 在画布(BGR Mat,已经是裁出来的黑白可行走区域)里随机挑一个
/// "灰色可行走、且跟主体区域相通、离边界有一定安全距离"的像素点,
/// 返回相对画布自身的坐标(不是原图坐标)。
///
/// 完整流程:
/// 1. 转灰度 -> 阈值二值化(亮度够高才算可行走)
/// 2. 开运算去掉装饰性小光点噪声
/// 3. 连通域分析,只保留面积最大的那一块区域(避免选到孤立、走不通的
///    小角落 —— 这块最大区域基本就是整张地图真正连通的可行走网络)
/// 4. 在这块区域基础上再做一次腐蚀,让候选点整体往区域内部收缩,
///    避免选到贴着边界、点击容易没反应的像素
/// 5. 在最终剩下的像素里随机选一个
pub fn pick_random_walkable_point(canvas_bgr: &Mat, cfg: &NavConfig) -> Result<Option<(i32, i32)>> {
    let mut gray = Mat::default();
    imgproc::cvt_color(
        canvas_bgr,
        &mut gray,
        imgproc::COLOR_BGR2GRAY,
        0,
        core::AlgorithmHint::ALGO_HINT_DEFAULT,
    )?;

    let mut mask = Mat::default();
    imgproc::threshold(
        &gray,
        &mut mask,
        cfg.walkable_brightness_min,
        255.0,
        imgproc::THRESH_BINARY,
    )?;

    // 第一次形态学开运算:去掉装饰性小光点噪声
    let denoise_kernel = imgproc::get_structuring_element(
        imgproc::MORPH_RECT,
        core::Size::new(cfg.denoise_kernel_size, cfg.denoise_kernel_size),
        core::Point::new(-1, -1),
    )?;
    let mut opened = Mat::default();
    imgproc::morphology_ex(
        &mask,
        &mut opened,
        imgproc::MORPH_OPEN,
        &denoise_kernel,
        core::Point::new(-1, -1),
        1,
        core::BORDER_CONSTANT,
        imgproc::morphology_default_border_value()?,
    )?;

    // 🎯 连通域分析,只保留面积最大的那一块(整张地图真正连通的
    // 可行走网络),排除掉噪声去除后仍然残留的孤立小区域。
    let mut labels = Mat::default();
    let mut stats = Mat::default();
    let mut centroids = Mat::default();
    let num = imgproc::connected_components_with_stats(
        &opened,
        &mut labels,
        &mut stats,
        &mut centroids,
        8,
        core::CV_32S,
    )?;

    if num <= 1 {
        // 只有背景,没有任何可行走区域
        return Ok(None);
    }

    let mut largest_label = 1;
    let mut largest_area = 0i32;
    for i in 1..num {
        let area = *stats.at_2d::<i32>(i, imgproc::CC_STAT_AREA)?;
        if area > largest_area {
            largest_area = area;
            largest_label = i;
        }
    }

    // 只保留最大连通域对应的像素,其余全部置为背景(0)
    let mut main_region = Mat::default();
    core::compare(
        &labels,
        &core::Scalar::new(largest_label as f64, 0.0, 0.0, 0.0),
        &mut main_region,
        core::CMP_EQ,
    )?;

    // 🎯 再做一次腐蚀,让候选点整体往区域内部收缩,避开贴边点击没反应的问题
    let erode_kernel = imgproc::get_structuring_element(
        imgproc::MORPH_RECT,
        core::Size::new(cfg.erode_margin_kernel_size, cfg.erode_margin_kernel_size),
        core::Point::new(-1, -1),
    )?;
    let mut safe_region = Mat::default();
    imgproc::erode(
        &main_region,
        &mut safe_region,
        &erode_kernel,
        core::Point::new(-1, -1),
        1,
        core::BORDER_CONSTANT,
        imgproc::morphology_default_border_value()?,
    )?;

    // 收集候选像素坐标,随机选一个。
    // 用 at_2d 逐像素访问(项目里 monster_detector.rs/position_reader.rs
    // 已经验证过这套 API 可用),避免用不确定是否存在的 at_row API。
    let rows = safe_region.rows();
    let cols = safe_region.cols();
    let mut candidates: Vec<(i32, i32)> = Vec::new();

    for y in 0..rows {
        for x in 0..cols {
            let v = *safe_region.at_2d::<u8>(y, x)?;
            if v > 0 {
                candidates.push((x, y));
            }
        }
    }

    if candidates.is_empty() {
        // 腐蚀力度可能设太大,把整块区域都收缩没了,退化到用腐蚀前的
        // 主连通域(不做内缩)兜底,总比完全选不出点强。
        println!("⚠️  [地图导航] 腐蚀收缩后无候选点,回退到不做内缩的主连通域");
        for y in 0..rows {
            for x in 0..cols {
                let v = *main_region.at_2d::<u8>(y, x)?;
                if v > 0 {
                    candidates.push((x, y));
                }
            }
        }
    }

    if candidates.is_empty() {
        return Ok(None);
    }

    let mut rng = rand::thread_rng();
    let idx = rng.gen_range(0..candidates.len());
    Ok(Some(candidates[idx]))
}

/// 🔘 在整帧截图里定位关闭按钮("X"图标),返回物理像素坐标 + 置信度
fn find_close_button(
    raw_rgba: &[u8],
    width: u32,
    height: u32,
    cfg: &NavConfig,
) -> Option<((u32, u32), f32)> {
    match_icon::find_icon_opencv(
        raw_rgba,
        width,
        height,
        &cfg.close_button_template,
        cfg.close_button_min_confidence,
        "close_map",
    )
}

/// 🖱️ 点击打开大地图面板(固定比例位置估算,罗盘图标较大、容错度高)
pub fn open_big_map(window: &Window, enigo: &mut Enigo, cfg: &NavConfig) {
    let click_x = window.x() + (window.width() as f64 * cfg.open_click_x_frac) as i32;
    let click_y = window.y() + (window.height() as f64 * cfg.open_click_y_frac) as i32;

    println!(
        "🗺️  [地图导航] 点击打开大地图面板: ({}, {})",
        click_x, click_y
    );
    mouse_action::click_at(enigo, click_x, click_y, "【地图导航】打开大地图");

    // 给面板弹出动画留点时间
    thread::sleep(Duration::from_millis(200));
}

/// 🖱️ 关闭大地图面板。用图标匹配定位关闭按钮,而不是写死坐标,
/// 面板弹出位置/尺寸万一有偏移也不受影响。
/// 返回是否成功找到并点击了关闭按钮。
pub fn close_big_map(
    window: &Window,
    enigo: &mut Enigo,
    raw_rgba: &[u8],
    width: u32,
    height: u32,
    cfg: &NavConfig,
) -> bool {
    match find_close_button(raw_rgba, width, height, cfg) {
        Some(((px, py), score)) => {
            let (scale_x, scale_y) = compute_scale_factor(window, width, height);
            let click_x = window.x() + (px as f32 / scale_x) as i32;
            let click_y = window.y() + (py as f32 / scale_y) as i32;
            println!(
                "🗺️  [地图导航] 找到关闭按钮(置信度 {:.2}%),点击关闭: ({}, {})",
                score * 100.0,
                click_x,
                click_y
            );
            mouse_action::click_at(enigo, click_x, click_y, "【地图导航】关闭大地图");
            thread::sleep(Duration::from_millis(200));
            true
        }
        None => {
            println!("⚠️  [地图导航] 未能定位到关闭按钮,大地图面板可能未成功打开");
            false
        }
    }
}

/// 🎯 主流程:打开大地图 -> 在灰色可行走区域随机选点点击(触发引擎自动
/// 寻路) -> 关闭面板回到游戏主界面。
///
/// 返回值语义:
/// - `Ok(true)`  : 全流程成功,已经点击了一个目标点触发自动寻路
/// - `Ok(false)` : 大地图面板未能成功打开/关闭按钮找不到(大概率这张
///                 地图不支持这个流程,或者点击位置/模板需要重新校准),
///                 调用方应该退回原来的手动方向试探寻路逻辑
/// - `Err(_)`    : OpenCV 相关调用出错
pub fn navigate_to_random_point(
    window: &Window,
    enigo: &mut Enigo,
    cfg: &NavConfig,
) -> Result<bool> {
    open_big_map(window, enigo, cfg);

    let (raw_rgba, width, height) = match util::capture_window(window) {
        Some(v) => v,
        None => {
            println!("⚠️  [地图导航] 打开地图后截图失败");
            return Ok(false);
        }
    };

    // 先确认面板确实打开了(能找到关闭按钮),找不到说明这条路走不通,
    // 让调用方退回旧的手动方向试探逻辑。
    let close_btn = find_close_button(&raw_rgba, width, height, cfg);
    if close_btn.is_none() {
        println!("⚠️  [地图导航] 打开地图后未检测到关闭按钮,判定这张地图暂不支持该流程");
        return Ok(false);
    }

    let bgr = crate::monster_detector::rgba_bytes_to_bgr_mat(&raw_rgba, width, height)?;
    let canvas_rect = compute_canvas_rect(bgr.cols(), bgr.rows(), cfg);
    let canvas_img = Mat::roi(&bgr, canvas_rect)?.try_clone()?;

    let point = pick_random_walkable_point(&canvas_img, cfg)?;

    let clicked = match point {
        Some((local_x, local_y)) => {
            let physical_x = (canvas_rect.x + local_x) as u32;
            let physical_y = (canvas_rect.y + local_y) as u32;

            let (scale_x, scale_y) = compute_scale_factor(window, width, height);
            let click_x = window.x() + (physical_x as f32 / scale_x) as i32;
            let click_y = window.y() + (physical_y as f32 / scale_y) as i32;

            println!(
                "🗺️  [地图导航] 在可行走区域随机选点并点击,触发自动寻路: ({}, {})",
                click_x, click_y
            );
            mouse_action::click_at(enigo, click_x, click_y, "【地图导航】点击地图目标点");

            // 给引擎一点时间计算路径(青色虚线出现)
            thread::sleep(Duration::from_millis(100));
            true
        }
        None => {
            println!("⚠️  [地图导航] 画布内未能找到任何可行走(灰色)像素点");
            false
        }
    };

    // 不管有没有成功点到目标点,都要把面板关掉,否则会挡住后台的
    // 战斗/移动判断。
    let (raw_rgba2, width2, height2) =
        util::capture_window(window).unwrap_or((raw_rgba, width, height));
    close_big_map(window, enigo, &raw_rgba2, width2, height2, cfg);

    Ok(clicked)
}
