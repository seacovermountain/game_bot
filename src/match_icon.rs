// src/match_icon.rs

use crate::util;
use opencv::{
    core::{self, Mat, Point, min_max_loc},
    imgcodecs::{IMREAD_COLOR, imread},
    imgproc::{self, TemplateMatchModes, match_template},
    prelude::*,
};
use std::path::PathBuf;

/// 🎯 模板图截取时所用的参考物理分辨率(宽度)。
///
/// ⚠️ 这是"动态缩放模板匹配"能生效的关键基准值 —— 游戏窗口是可以被
/// 用户随意拖拽缩放的(不是固定几档),如果只靠枚举几个预设缩放比例
/// 去碰运气,永远会有覆盖不到的中间尺寸。正确做法是:记录模板当初
/// 是在多大物理分辨率下截的图,运行时用"当前窗口物理宽度 / 这个基准
/// 宽度"算出精确缩放系数,匹配前先把模板本身缩放到位,再去比对。
///
/// 如果以后重新在某个窗口分辨率下截图建了新模板,请把这个值改成
/// 那次截图时的实际物理宽度(资产模块启动时会打印
/// "🌍 [跨平台缩放] 物理分辨率: WxH" 这一行,抄那个 W 即可)。
pub const TEMPLATE_REFERENCE_PHYSICAL_WIDTH: f64 = 3000.0;

/// 🎯 1:1 物理像素原色硬核对齐匹配（内部直接调用你的工具类方法存图调试）
pub fn find_icon_opencv(
    raw_rgba: &[u8],
    width: u32,
    height: u32,
    template_path: &str,
    min_confidence: f32,
    asset_id: &str, // 传入资产 ID 方便给调试大图命名
) -> Option<((u32, u32), f32)> {
    let rgb_image = util::rgba_to_rgb(raw_rgba, width, height); // 💥 调用你的格式转换方法

    let mut debug_save_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    debug_save_path.push(format!("DEBUG_HAYSTACK_{}.png", asset_id));

    // 调用你的存图方法，把脚本看到的真实物理世界截屏存下来
    let _ = util::save_debug_image(&rgb_image, &debug_save_path);
    // =========================================================================

    // 1. 将原始像素直接零拷贝封装成 OpenCV 矩阵
    let data_vec4b = unsafe {
        std::slice::from_raw_parts(
            raw_rgba.as_ptr() as *const core::Vec4b,
            (width * height) as usize,
        )
    };
    let src_rgba = Mat::new_rows_cols_with_data(height as i32, width as i32, data_vec4b).ok()?;

    // 2. 转换成 OpenCV 3通道 RGB 矩阵，用于模板匹配
    let mut src_rgb = Mat::default();
    opencv::imgproc::cvt_color(
        &src_rgba,
        &mut src_rgb,
        opencv::imgproc::COLOR_RGBA2RGB,
        0,
        core::AlgorithmHint::ALGO_HINT_DEFAULT,
    )
    .ok()?;

    // 3. 读取你的资产切图
    let needle_raw = imread(template_path, IMREAD_COLOR).ok()?;
    if needle_raw.empty() {
        println!(
            "   ⚠️ [资产错误] OpenCV 无法读取或找到小模板: {}",
            template_path
        );
        return None;
    }

    // 🎯 按当前窗口物理分辨率 vs 模板截图时的基准分辨率,算出精确缩放
    // 系数,把模板缩放到跟当前画面匹配的实际尺寸 —— 这样不管窗口被拖成
    // 什么尺寸,都能精确适配,而不是靠枚举几个预设比例去猜。
    let scale_factor = width as f64 / TEMPLATE_REFERENCE_PHYSICAL_WIDTH;
    let needle = if (scale_factor - 1.0).abs() > 0.01 {
        let new_w = ((needle_raw.cols() as f64) * scale_factor).round().max(1.0) as i32;
        let new_h = ((needle_raw.rows() as f64) * scale_factor).round().max(1.0) as i32;
        let mut resized = Mat::default();
        let interpolation = if scale_factor < 1.0 {
            imgproc::INTER_AREA // 缩小用 INTER_AREA,抗锯齿效果更好
        } else {
            imgproc::INTER_LINEAR // 放大用线性插值
        };
        if imgproc::resize(
            &needle_raw,
            &mut resized,
            core::Size::new(new_w, new_h),
            0.0,
            0.0,
            interpolation,
        )
        .is_err()
        {
            println!("   ⚠️ [资产错误] 模板缩放失败,退回使用原始尺寸模板");
            needle_raw
        } else {
            resized
        }
    } else {
        // 缩放系数接近 1,不需要额外缩放,省一次 resize 开销
        needle_raw
    };

    if needle.empty() {
        println!("   ⚠️ [资产错误] 模板缩放后为空: {}", template_path);
        return None;
    }

    // 4. 执行工业级 1:1 相关系数彩色图像匹配
    let mut match_result = Mat::default();
    match_template(
        &src_rgb,
        &needle,
        &mut match_result,
        TemplateMatchModes::TM_CCOEFF_NORMED.into(),
        &opencv::core::no_array(),
    )
    .ok()?;

    // 5. 抓出相似度最高的位置
    let mut min_val: f64 = 0.0;
    let mut max_val: f64 = 0.0;
    let mut min_loc = Point::default();
    let mut max_loc = Point::default();

    min_max_loc(
        &match_result,
        Some(&mut min_val),
        Some(&mut max_val),
        Some(&mut min_loc),
        Some(&mut max_loc),
        &opencv::core::no_array(),
    )
    .ok()?;

    let similarity = max_val as f32;
    // println!(
    //     "   📊 [OpenCV 匹配结果] 资产: [{:<8}] | 相似度: {:.2}% (目标: {:.2}%)",
    //     asset_id,
    //     similarity * 100.0,
    //     min_confidence * 100.0
    // );

    if similarity >= min_confidence {
        let needle_width = needle.cols() as u32;
        let needle_height = needle.rows() as u32;

        // 计算物理中心点
        let center_x = max_loc.x as u32 + (needle_width / 2);
        let center_y = max_loc.y as u32 + (needle_height / 2);

        Some(((center_x, center_y), similarity))
    } else {
        None
    }
}

/// 🍏 获取当前操作系统的屏幕缩放因子
pub fn get_screen_scale_factor() -> f32 {
    #[cfg(target_os = "macos")]
    {
        2.0
    }
    #[cfg(target_os = "windows")]
    {
        1.0
    }
    #[cfg(not(any(target_os = "windows", target_os = "macos")))]
    {
        1.0
    }
}
