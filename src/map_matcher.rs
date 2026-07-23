// src/map_matcher.rs
//! 地图名字 ROI 工具模块。
//!
//! ⚠️ 历史说明:这个模块最初是按"模板匹配"设计的(整图截一份地图名字
//! 模板,跟怪物名字识别 monster_matcher.rs 是同一套思路)。后来地图名字
//! 识别改成了 OCR(见 `text_ocr::match_map_name`,直接从整帧 OCR 结果
//! 里挑出落在牌匾区域的文字,不再需要维护 `templates/map_names/` 模板库),
//! 原来的 `identify_map()` / `load_map_templates()` / `MapTemplate` 已经
//! 删除。
//!
//! 现在这个模块只保留两样东西,供 `bot_loop.rs` 调试用:
//! - `MapReaderConfig`:牌匾 ROI 比例配置(OCR 版还是要用同一块 ROI
//!   去判断"文字块是不是落在牌匾里")。
//! - `debug_dump_map_roi()`:把牌匾 ROI 区域整块裁出来存盘,方便肉眼
//!   核对 ROI 范围有没有框对。

use opencv::{
    Result,
    core::{Mat, Rect},
    imgcodecs,
    prelude::*,
};

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

/// 🗂️ 调试用:把地图名字 ROI 区域整块裁出来存盘,肉眼核对 ROI 范围对不对。
/// 用法参考 position_reader::debug_dump_position_roi。
pub fn debug_dump_map_roi(frame_bgr: &Mat, cfg: &MapReaderConfig, out_path: &str) -> Result<()> {
    let roi_rect = compute_roi_rect(frame_bgr.cols(), frame_bgr.rows(), cfg);
    let roi_img = Mat::roi(frame_bgr, roi_rect)?.try_clone()?;
    let params = opencv::core::Vector::new();
    imgcodecs::imwrite(out_path, &roi_img, &params)?;
    Ok(())
}
