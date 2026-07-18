// src/item_ocr.rs
//! 物品识别模块 - 基于 OCR(而不是颜色阈值 + 模板匹配)。
//!
//! 💡 为什么物品这边不能照搬怪物识别的"模板匹配"思路:
//! 物品种类太多,没法像怪物名字(~20个)那样一个个截图建模板库,
//! config.toml 里的 loot.target_items 可能是几十上百个物品名字,
//! 逐个截图不现实。
//!
//! 改用 OCR:直接把画面里的文字"读"成字符串,再跟白名单做模糊字符串
//! 匹配(编辑距离),不需要为每个物品单独准备参考图。
//! 这一步顺便也不再需要关心物品名字到底是什么颜色(绿/红/黄/白都行),
//! 因为 OCR 的文字检测阶段本身就是"找文字区域",不依赖颜色。
//!
//! OCR 引擎用的是 `ocr-rs`(zibo-chen/rust-paddle-ocr,MNN 后端,
//! 本地免费离线运行,不需要联网/API Key,中文识别效果较好)。

use image::{DynamicImage, RgbImage};
use ocr_rs::OcrEngine;
use opencv::{core::Mat, prelude::*};
use std::collections::HashSet;
use std::error::Error;

#[derive(Debug, Clone)]
pub struct ItemOcrConfig {
    // ROI 左/上/右复用怪物检测的战斗区域比例,下边界单独放宽,
    // 但要卡在聊天框开始之前(实测聊天框大概从 0.71 开始)。
    pub roi_left_frac: f64,
    pub roi_top_frac: f64,
    pub roi_right_frac: f64,
    pub roi_bottom_frac: f64,

    // 编辑距离容错:识别结果跟白名单允许差几个字符也算命中。
    pub max_edit_distance: usize,
}

impl Default for ItemOcrConfig {
    fn default() -> Self {
        Self {
            roi_left_frac: 0.08,
            roi_top_frac: 0.20,
            roi_right_frac: 0.68,
            roi_bottom_frac: 0.70,
            max_edit_distance: 1,
        }
    }
}

/// 把 ROI 配置换算成原图坐标系下的绝对矩形
fn compute_roi_rect(frame_w: i32, frame_h: i32, cfg: &ItemOcrConfig) -> opencv::core::Rect {
    let x = (frame_w as f64 * cfg.roi_left_frac) as i32;
    let y = (frame_h as f64 * cfg.roi_top_frac) as i32;
    let w = (frame_w as f64 * (cfg.roi_right_frac - cfg.roi_left_frac)) as i32;
    let h = (frame_h as f64 * (cfg.roi_bottom_frac - cfg.roi_top_frac)) as i32;
    opencv::core::Rect::new(x, y, w.max(1), h.max(1))
}

/// 把 OpenCV 的 BGR Mat 转换成 `image` crate 用的 DynamicImage,
/// 供 ocr-rs 的 recognize() 使用。假设 Mat 是连续内存(裁剪后
/// try_clone() 得到的 Mat 通常满足这一点)。
fn mat_bgr_to_dynamic_image(mat: &Mat) -> Result<DynamicImage, Box<dyn Error>> {
    let width = mat.cols() as u32;
    let height = mat.rows() as u32;
    let data = mat.data_bytes()?;

    let expected_len = (width as usize) * (height as usize) * 3;
    if data.len() < expected_len {
        return Err("Mat 数据长度小于预期,可能不是连续内存".into());
    }

    let mut rgb_buf = vec![0u8; expected_len];
    for i in 0..(width as usize * height as usize) {
        // BGR -> RGB
        rgb_buf[i * 3] = data[i * 3 + 2];
        rgb_buf[i * 3 + 1] = data[i * 3 + 1];
        rgb_buf[i * 3 + 2] = data[i * 3];
    }

    let rgb_image =
        RgbImage::from_raw(width, height, rgb_buf).ok_or("无法从像素数据构建 RgbImage")?;
    Ok(DynamicImage::ImageRgb8(rgb_image))
}

/// 判断一个字符是不是中文(CJK 统一表意文字,覆盖绝大多数常用汉字范围)。
fn is_chinese_char(c: char) -> bool {
    let code = c as u32;
    (0x4E00..=0x9FFF).contains(&code) // CJK 统一表意文字(常用区)
        || (0x3400..=0x4DBF).contains(&code) // CJK 扩展 A
        || (0xF900..=0xFAFF).contains(&code) // CJK 兼容表意文字
}

/// 🎯 过滤掉识别结果里的非中文字符(数字、括号、标点等)。
/// 物品名字本身不带任何符号,OCR 偶尔会把游戏里挂着的额外提示
/// (比如"(大量)"这种数量标签)一起框进同一段文字,过滤掉这些
/// 非中文字符,只留纯汉字再去跟白名单比对。
fn filter_chinese_only(text: &str) -> String {
    text.chars().filter(|c| is_chinese_char(*c)).collect()
}

/// 🎯 在识别到的(可能是多个物品名字粘连在一起的)一整段文字里,
/// 用滑动窗口找出白名单里每个物品名字有没有作为"近似子串"出现过,
/// 而不是要求整段文字从头到尾都跟某个白名单物品名字对得上。
///
/// 例如整段是"强效太阳神水治疗药水"(OCR 把两个挨得近的物品名字识别
/// 成了同一个文字框),白名单里"强效太阳神水"(6字)和"治疗药水"(4字)
/// 都能在这段文字里找到编辑距离足够小的子串,两个都会被找出来。
///
/// 返回这一段文字里命中的所有白名单物品名字(去重)。
fn find_whitelist_matches(
    text: &str,
    whitelist: &HashSet<String>,
    max_edit_distance: usize,
) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let mut found = Vec::new();

    for candidate in whitelist {
        let candidate_len = candidate.chars().count();
        if candidate_len == 0 {
            continue;
        }

        let best_dist = if chars.len() < candidate_len {
            // 整段文字本身比白名单物品名字还短,直接整体比一次即可,
            // 没法再切出跟白名单等长的窗口。
            strsim::levenshtein(text, candidate)
        } else {
            let mut min_dist = usize::MAX;
            for start in 0..=(chars.len() - candidate_len) {
                let window: String = chars[start..start + candidate_len].iter().collect();
                let dist = strsim::levenshtein(&window, candidate);
                if dist < min_dist {
                    min_dist = dist;
                }
            }
            min_dist
        };

        if best_dist <= max_edit_distance {
            found.push(candidate.clone());
        }
    }

    found
}

pub struct ItemOcrRecognizer {
    engine: OcrEngine,
}

impl ItemOcrRecognizer {
    /// 加载一次 OCR 引擎(det + rec 模型 + 字典),跟怪物/数字模板库一样,
    /// 只在程序启动时初始化一次,不要每轮循环都重新加载。
    pub fn new(
        det_model_path: &str,
        rec_model_path: &str,
        keys_path: &str,
    ) -> Result<Self, Box<dyn Error>> {
        let engine = OcrEngine::new(det_model_path, rec_model_path, keys_path, None)?;
        Ok(Self { engine })
    }

    /// 从整帧 BGR Mat 裁出物品 ROI,OCR 识别出所有文字块,
    /// 每个文字块跟白名单做编辑距离模糊匹配,返回命中的
    /// (匹配到的白名单物品名, OCR 原始识别文字, OCR 置信度)。
    pub fn detect_items(
        &self,
        frame_bgr: &Mat,
        cfg: &ItemOcrConfig,
        whitelist: &HashSet<String>,
    ) -> Result<Vec<(String, String, f32)>, Box<dyn Error>> {
        let roi_rect = compute_roi_rect(frame_bgr.cols(), frame_bgr.rows(), cfg);
        let roi_mat = Mat::roi(frame_bgr, roi_rect)?.try_clone()?;

        let image = mat_bgr_to_dynamic_image(&roi_mat)?;
        let results = self.engine.recognize(&image)?;

        // 🐛 调试用:把 OCR 这一帧识别到的所有原始文字都打出来,不管有没有
        // 匹配上白名单 —— 排查问题第一步应该是先确认 OCR 到底读出了什么,
        // 而不是直接看"有没有匹配上白名单"这个下游结果。
        if results.is_empty() {
            println!("   🔤 [物品OCR] 本帧未识别到任何文字块");
        } else {
            for block in &results {
                println!(
                    "   🔤 [物品OCR] 识别到文字: \"{}\" | 置信度: {:.2}%",
                    block.text,
                    block.confidence * 100.0
                );
            }
        }

        let mut matched = Vec::new();
        for block in results {
            // 先过滤掉非中文字符(数字/括号/标点等),物品名字本身不带符号,
            // 这些多半是游戏额外挂着的数量提示或者跟血量数字粘连的干扰。
            let cleaned = filter_chinese_only(&block.text);
            if cleaned.is_empty() {
                continue;
            }

            // 再用滑动窗口在这段(可能是多个物品名字粘连在一起的)文字里
            // 找出所有命中的白名单物品名字,而不要求整段文字完全对应
            // 某一个白名单物品。
            for item_name in find_whitelist_matches(&cleaned, whitelist, cfg.max_edit_distance) {
                matched.push((item_name, block.text.clone(), block.confidence));
            }
        }

        Ok(matched)
    }
}
