// src/monster_matcher.rs
//! 怪物名字模板匹配工具 —— 现在只给 `src/bin/test_monster_recognition.rs`
//! 这个独立测试工具用(建模板库时验证识别率)。
//!
//! ⚠️ 历史说明:怪物识别最初是走这套"模板匹配"逻辑(不用通用 OCR,
//! 原因见下方历史注释)。后来 bot_loop.rs 里的实时怪物识别改成了 OCR
//! (见 `text_ocr::match_monsters`,不用维护 `templates/monster_names/`
//! 模板库也能识别任意怪物名字),原来对接实时挂机循环的
//! `identify_monsters()` / `match_single_box()` 已经删除。
//!
//! 现在这个模块只保留 `load_monster_templates()` + `identify_crop()`,
//! 单纯用来对"已经裁好的单张候选框图片"做模板匹配,评估模板库本身的
//! 识别准确率(见 test_monster_recognition.rs)。
//!
//! 💡 原来为什么不用通用 OCR(历史背景,现在实时识别已经改用 OCR 了):
//! - 你要认的怪物名字是固定的白名单(config.toml 里那十几二十个)，
//!   用的还是游戏客户端固定字体/字号,没有"识别任意文字"的需求。
//! - 模板匹配直接复用项目里 match_icon.rs 已经在用的 OpenCV
//!   match_template 能力，天然跨平台,不需要额外依赖。
//!
//! 📋 建模板库流程:
//! 1. 找一局游戏画面里出现了目标怪物时,跑一次
//!    `monster_detector::detect_monsters_from_rgba` + `debug_crop_boxes`，
//!    把当前这一帧所有候选框都单独裁剪存盘。
//! 2. 肉眼过一遍裁出来的小图,挑出真正是"怪物名字"的那几张,
//!    改名成和 `config.toml` 里完全一致的怪物名字,例如
//!    `地火兽骑将.png`,丢进 `templates/monster_names/` 目录。
//! 3. 同一个怪物建议存 1~2 张模板(比如带不带特效遮挡的两种状态)，
//!    命中率会更稳。
//! 4. 调用 `load_monster_templates()` 加载模板库,再用
//!    `cargo run --bin test_monster_recognition` 跑识别率测试。

use opencv::{
    Result,
    core::{self, Mat, Point, min_max_loc},
    imgcodecs::{IMREAD_COLOR, imread},
    imgproc::{TemplateMatchModes, match_template},
    prelude::*,
};
use std::fs;
use std::path::Path;

/// 一个怪物名字的参考模板小图
#[derive(Debug, Clone)]
pub struct MonsterTemplate {
    pub name: String,
    pub template: Mat,
}

/// 📂 从目录批量加载怪物名字模板。
/// 目录里每张图片的"文件名(不含扩展名)"就是怪物名字,
/// 例如 templates/monster_names/地火兽骑将.png -> name = "地火兽骑将"
pub fn load_monster_templates<P: AsRef<Path>>(dir: P) -> Result<Vec<MonsterTemplate>> {
    let mut templates = Vec::new();

    let entries = match fs::read_dir(&dir) {
        Ok(e) => e,
        Err(_) => {
            println!(
                "   ⚠️ [怪物模板库] 目录不存在或无法读取: {}(还没建模板库的话这是正常的)",
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
            println!("   ⚠️ [怪物模板库] 无法读取模板图片: {}", path.display());
            continue;
        }

        println!("   📎 [怪物模板库] 已加载模板: {}", name);
        templates.push(MonsterTemplate { name, template });
    }

    println!(
        "   ✅ [怪物模板库] 共加载 {} 个怪物名字模板",
        templates.len()
    );

    Ok(templates)
}

/// 🎯 识别率测试专用:直接对"已经裁好的单张候选框图片"做模板匹配,
/// 不做"按整帧宽度换算 scale_factor 再缩放模板"这一步。
///
/// 用途:`DEBUG_MONSTER_CROPS/` 里存的就是单独裁剪出来的候选框小图,
/// 跟建模板库时用的是同一次截图、同一个物理分辨率,不需要(也没法)
/// 再按"整帧宽度 / 3000 基准宽度"去缩放模板——crop 图片本身的宽度
/// 跟"整帧宽度"完全不是一回事,硬套那套公式反而会把模板缩得离谱小。
///
/// 返回 (最佳匹配的怪物名字, 匹配分数),不管有没有超过置信度阈值,
/// 阈值判断交给调用方(测试工具需要同时看到"匹配上了但分数不够"和
/// "压根没匹配上任何模板"这两种不同的失败情况)。
pub fn identify_crop(crop_bgr: &Mat, templates: &[MonsterTemplate]) -> Result<Option<(String, f32)>> {
    let mut best_name: Option<String> = None;
    let mut best_score: f32 = 0.0;

    for tpl in templates {
        // 模板必须不大于待匹配的 crop,否则 match_template 会报错,跳过即可
        // (说明这张 crop 裁得比模板还小,多半是候选框本身裁剪不完整)。
        if tpl.template.cols() > crop_bgr.cols() || tpl.template.rows() > crop_bgr.rows() {
            continue;
        }

        let mut result = Mat::default();
        match_template(
            crop_bgr,
            &tpl.template,
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

    Ok(best_name.map(|n| (n, best_score)))
}
