// src/map_matcher.rs
//! 地图名称识别模块 - 基于模板匹配(跟怪物名字识别 monster_matcher.rs 是同一套思路)
//!
//! 💡 为什么整图模板匹配天然兼容"地图名字长短不一":
//! - 地图名字是一个有限的白名单集合(心之魔域、XX秘境...)，每个地图名字
//!   单独截一张模板图,模板本身的宽度就是这个地图名字实际渲染出来的宽度,
//!   不需要做任何"补齐/对齐"处理。
//! - ROI(从实时截图里裁出来的"地图名字可能出现的区域")只需要比"最长的
//!   那个地图名字"再宽松一些即可,不需要跟每个模板的宽度精确对齐——
//!   `match_template` 本身就是在一块更大的区域里找模板最佳匹配位置,
//!   模板比 ROI 窄是完全正常的用法(参考 match_icon.rs 对按钮的识别方式)。
//! - 因为是整图(而不是单字符)匹配,"心之魔域"4个字 vs 更长的地图名字，
//!   相似度计算时天然就是两种不同尺寸的模板,不会互相干扰。
//!
//! 📋 使用流程:
//! 1. 先用 `debug_dump_map_roi` 把地图名字可能出现的区域整块裁出来存盘,
//!    肉眼确认 ROI 范围是不是刚好框住地图名字文字(不多不少,留一点余量
//!    应付不同长度的地图名字即可)。
//! 2. 手动把当前地图下的这个区域单独裁剪、只保留地图名字文字本身
//!    (不带多余背景/图标),改名成对应的地图名字,例如 `心之魔域.png`,
//!    放进 `templates/map_names/` 目录。
//! 3. 每种地图都实际走一遍,截一张,积累成一个完整的地图名字模板库。
//! 4. 调用 `load_map_templates()` 加载,再调用 `identify_map()` 识别
//!    当前地图。

use opencv::{
    Result,
    core::{self, Mat, Point, Rect, min_max_loc},
    imgcodecs::{self, IMREAD_COLOR, imread},
    imgproc::{self, TemplateMatchModes, match_template},
    prelude::*,
};
use std::fs;
use std::path::Path;

/// 一个地图名字的参考模板小图
#[derive(Debug, Clone)]
pub struct MapTemplate {
    pub name: String,
    pub template: Mat,
}

/// 地图名字识别的 ROI 配置。跟 position_reader::PositionReaderConfig 是
/// 同一套设计思路:比例相对整帧画面,而不是绝对像素,方便跨分辨率。
#[derive(Debug, Clone)]
pub struct MapReaderConfig {
    // ⚠️ 这套比例是基于截图目测估算的初始值,还没有用 DEBUG 图精确核对过,
    // 请先用 debug_dump_map_roi 存一张图肉眼确认,再按实际情况微调。
    // 目前估算:地图名字紧贴在右上角罗盘下方的牌匾里。
    pub roi_left_frac: f64,
    pub roi_top_frac: f64,
    pub roi_right_frac: f64,
    pub roi_bottom_frac: f64,
}

impl Default for MapReaderConfig {
    fn default() -> Self {
        Self {
            // ✅ 已用 DEBUG_MAP_ROI.png 实测校准(3000x1716 截图下,
            // 牌匾紧密边界框约 x:[2280,2970] y:[34,146])，
            // 左右各留了一点余量,应付地图名字字数变化(比如比"心之魔域"更长的名字)。
            roi_left_frac: 0.76,
            roi_top_frac: 0.02,
            roi_right_frac: 0.99,
            roi_bottom_frac: 0.085,
        }
    }
}

/// 把 ROI 配置换算成原图坐标系下的绝对矩形
fn compute_roi_rect(frame_w: i32, frame_h: i32, cfg: &MapReaderConfig) -> Rect {
    let x = (frame_w as f64 * cfg.roi_left_frac) as i32;
    let y = (frame_h as f64 * cfg.roi_top_frac) as i32;
    let w = (frame_w as f64 * (cfg.roi_right_frac - cfg.roi_left_frac)) as i32;
    let h = (frame_h as f64 * (cfg.roi_bottom_frac - cfg.roi_top_frac)) as i32;
    Rect::new(x, y, w.max(1), h.max(1))
}

/// 📂 从目录批量加载地图名字模板。
/// 目录里每张图片的"文件名(不含扩展名)"就是地图名字,
/// 例如 templates/map_names/心之魔域.png -> name = "心之魔域"
pub fn load_map_templates<P: AsRef<Path>>(dir: P) -> Result<Vec<MapTemplate>> {
    let mut templates = Vec::new();

    let entries = match fs::read_dir(&dir) {
        Ok(e) => e,
        Err(_) => {
            println!(
                "   ⚠️ [地图名字模板库] 目录不存在或无法读取: {}(还没建模板库的话这是正常的)",
                dir.as_ref().display()
            );
            return Ok(templates);
        }
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("png") {
            continue;
        }

        let name = match path.file_stem().and_then(|s| s.to_str()) {
            Some(n) => n.to_string(),
            None => continue,
        };

        let template = imread(&path.to_string_lossy(), IMREAD_COLOR)?;
        if template.empty() {
            println!(
                "   ⚠️ [地图名字模板库] 无法读取模板图片: {}",
                path.display()
            );
            continue;
        }

        println!("   📎 [地图名字模板库] 已加载模板: {}", name);
        templates.push(MapTemplate { name, template });
    }

    println!(
        "   ✅ [地图名字模板库] 共加载 {} 个地图名字模板",
        templates.len()
    );

    Ok(templates)
}

/// 🎯 主函数:从当前整帧(BGR Mat)里识别出当前地图名字。
/// 对每个模板都在 ROI 区域内做一次 match_template,取相似度最高、
/// 且超过 min_confidence 的那个作为识别结果。
/// 模板比 ROI 宽/高的情况会被跳过(说明这套 ROI 明显框小了,需要重新标定)。
pub fn identify_map(
    frame_bgr: &Mat,
    cfg: &MapReaderConfig,
    templates: &[MapTemplate],
    min_confidence: f32,
) -> Result<Option<(String, f32)>> {
    if templates.is_empty() {
        return Ok(None);
    }

    let roi_rect = compute_roi_rect(frame_bgr.cols(), frame_bgr.rows(), cfg);
    let roi_img = Mat::roi(frame_bgr, roi_rect)?.try_clone()?;

    // 🎯 按当前整帧的物理宽度 vs 模板截图时的基准分辨率,算出精确缩放
    // 系数 —— 跟 match_icon.rs 里按钮匹配的自适应缩放是同一套思路,
    // 避免窗口分辨率一变,地图名字模板就整体匹配不上。
    let scale_factor =
        frame_bgr.cols() as f64 / crate::match_icon::TEMPLATE_REFERENCE_PHYSICAL_WIDTH;

    let mut best_name: Option<String> = None;
    let mut best_score: f32 = 0.0;

    for tpl in templates {
        let scaled_template = if (scale_factor - 1.0).abs() > 0.01 {
            let new_w = ((tpl.template.cols() as f64) * scale_factor)
                .round()
                .max(1.0) as i32;
            let new_h = ((tpl.template.rows() as f64) * scale_factor)
                .round()
                .max(1.0) as i32;
            let mut resized = Mat::default();
            let interpolation = if scale_factor < 1.0 {
                imgproc::INTER_AREA
            } else {
                imgproc::INTER_LINEAR
            };
            match imgproc::resize(
                &tpl.template,
                &mut resized,
                core::Size::new(new_w, new_h),
                0.0,
                0.0,
                interpolation,
            ) {
                Ok(_) => resized,
                Err(_) => tpl.template.clone(),
            }
        } else {
            tpl.template.clone()
        };

        // 模板必须比 ROI 小(或相等),否则 match_template 会直接报错,跳过即可。
        // 如果所有模板都被跳过,大概率是 ROI 标定得太小,需要检查 MapReaderConfig。
        if scaled_template.cols() > roi_img.cols() || scaled_template.rows() > roi_img.rows() {
            continue;
        }

        let mut result = Mat::default();
        match_template(
            &roi_img,
            &scaled_template,
            &mut result,
            TemplateMatchModes::TM_CCOEFF_NORMED.into(),
            &core::no_array(),
        )?;

        let mut min_val: f64 = 0.0;
        let mut max_val: f64 = 0.0;
        let mut min_loc = Point::default();
        let mut max_loc = Point::default();

        min_max_loc(
            &result,
            Some(&mut min_val),
            Some(&mut max_val),
            Some(&mut min_loc),
            Some(&mut max_loc),
            &core::no_array(),
        )?;

        let score = max_val as f32;
        if score > best_score {
            best_score = score;
            best_name = Some(tpl.name.clone());
        }
    }

    if best_score >= min_confidence {
        Ok(best_name.map(|n| (n, best_score)))
    } else {
        Ok(None)
    }
}

/// 🗂️ 调试用:把地图名字 ROI 区域整块裁出来存盘,肉眼核对 ROI 范围对不对。
/// 用法参考 position_reader::debug_dump_position_roi。
pub fn debug_dump_map_roi(frame_bgr: &Mat, cfg: &MapReaderConfig, out_path: &str) -> Result<()> {
    let roi_rect = compute_roi_rect(frame_bgr.cols(), frame_bgr.rows(), cfg);
    let roi_img = Mat::roi(frame_bgr, roi_rect)?.try_clone()?;
    let params = opencv::core::Vector::new();
    imgcodecs::imwrite(out_path, &roi_img, &params)?;
    Ok(())
}
