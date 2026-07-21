// src/monster_matcher.rs
//! 怪物名字精确识别模块 - 基于模板匹配(而不是通用 OCR)
//!
//! 💡 为什么不用 Tesseract/PaddleOCR 这类通用文字识别引擎:
//! - 你要认的怪物名字是固定的白名单(config.toml 里那十几二十个)，
//!   用的还是游戏客户端固定字体/字号,没有"识别任意文字"的需求。
//! - 通用 OCR 引擎要在 Mac 和 Windows 上分别装好、配好中文语言包，
//!   环境搭建成本高,还容易出现平台间行为不一致。
//! - 模板匹配直接复用项目里 match_icon.rs 已经在用的 OpenCV
//!   match_template 能力，天然跨平台,不需要额外依赖。
//!
//! 🎁 副作用(意外的好处):模板匹配天然只认识"你给过模板的名字"。
//! monster_detector.rs 检测出来的候选框里,凡是没有对应模板的
//! (血条数字、UI 图标文字、其他玩家名字...)，匹配全部失败,
//! 会被这一步自动丢弃 —— 相当于顺便完成了一轮"语义级别"的降噪,
//! 比单纯调 ROI/阈值/宽高比这些形状特征要干净得多。
//!
//! 📋 使用流程:
//! 1. 找一局游戏画面里出现了目标怪物时,跑一次
//!    `monster_detector::detect_monsters_from_rgba` + `debug_crop_boxes`，
//!    把当前这一帧所有候选框都单独裁剪存盘。
//! 2. 肉眼过一遍裁出来的小图,挑出真正是"怪物名字"的那几张,
//!    改名成和 `config.toml` 里完全一致的怪物名字,例如
//!    `地火兽骑将.png`,丢进 `templates/monster_names/` 目录。
//! 3. 同一个怪物建议存 1~2 张模板(比如带不带特效遮挡的两种状态)，
//!    命中率会更稳。
//! 4. 调用 `load_monster_templates()` 加载模板库,再调用
//!    `identify_monsters()` 对检测到的候选框做识别。

use crate::monster_detector::TextBox;
use opencv::{
    Result,
    core::{self, Mat, Point, min_max_loc},
    imgcodecs::{IMREAD_COLOR, imread},
    imgproc::{self, TemplateMatchModes, match_template},
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

/// 🎯 对单个候选文字框做模板匹配,识别出框里出现的所有怪物名字
/// (可能不止一个——多只怪物挨在一起时,名字连通域会粘成一个框,
/// 这里按"每个模板独立判断是否达到阈值"而不是"整个框只选一个赢家",
/// 把粘在一起的多个名字都找出来)。
///
/// 会在候选框基础上外扩一点边距再裁剪,避免检测框比模板略小/略偏
/// 导致匹配失败。
///
/// ⚠️ 位置去重(NMS):`config.toml` 里有些名字互相是前缀/近似关系,
/// 比如"兽骑统领" / "兽骑统领头目" / "兽骑兵头目",这些模板可能会在
/// 同一段文字的同一个位置都拿到不低的分数。如果不做处理,一个怪物
/// 会被同时报成两个名字。所以先按分数从高到低排序,位置明显重叠的
/// 只保留分数最高的那一个。
fn match_single_box(
    haystack_bgr: &Mat,
    text_box: &TextBox,
    templates: &[MonsterTemplate],
    min_confidence: f32,
) -> Result<Vec<(String, TextBox, f32)>> {
    const PADDING: i32 = 6;

    let img_w = haystack_bgr.cols();
    let img_h = haystack_bgr.rows();

    let x = (text_box.x - PADDING).max(0);
    let y = (text_box.y - PADDING).max(0);
    let w = (text_box.w + PADDING * 2).min(img_w - x);
    let h = (text_box.h + PADDING * 2).min(img_h - y);

    if w <= 0 || h <= 0 {
        return Ok(Vec::new());
    }

    let crop_rect = core::Rect::new(x, y, w, h);
    let cropped = Mat::roi(haystack_bgr, crop_rect)?;

    // 🎯 按当前整帧的物理宽度 vs 模板截图时的基准分辨率,算出精确缩放
    // 系数 —— 跟 match_icon.rs 里按钮匹配的自适应缩放是同一套思路,
    // 避免窗口分辨率一变,怪物名字模板匹配整体失效(会被误判成"识别到的
    // 怪物不在白名单内",其实是压根没匹配上,不容易发现)。
    let scale_factor = img_w as f64 / crate::match_icon::TEMPLATE_REFERENCE_PHYSICAL_WIDTH;

    // 第一步:每个模板独立判断,达到阈值的都先收集起来(不做"赢家通吃")
    let mut candidates: Vec<(String, TextBox, f32)> = Vec::new();

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

        // 模板必须比裁剪区域小(或相等),否则 match_template 会直接报错,跳过即可
        if scaled_template.cols() > cropped.cols() || scaled_template.rows() > cropped.rows() {
            continue;
        }

        let mut result = Mat::default();
        match_template(
            &cropped,
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
        if score >= min_confidence {
            // 换算回整帧坐标系:不是整个候选框,而是按模板尺寸换算出来的
            // 这一小块具体位置,后续瞄准攻击也更精准。
            candidates.push((
                tpl.name.clone(),
                TextBox {
                    x: x + max_loc.x,
                    y: y + max_loc.y,
                    w: scaled_template.cols(),
                    h: scaled_template.rows(),
                    area: scaled_template.cols() * scaled_template.rows(),
                },
                score,
            ));
        }
    }

    // 第二步:按分数从高到低排序,位置去重(NMS)——重叠明显的候选里
    // 只留分数最高的那一个,避免前缀/近似名字模板在同一段文字上重复命中。
    candidates.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

    let mut kept: Vec<(String, TextBox, f32)> = Vec::new();
    for (name, rect, score) in candidates {
        let overlaps_kept = kept
            .iter()
            .any(|(_, kept_rect, _)| rects_overlap_significantly(kept_rect, &rect));
        if !overlaps_kept {
            kept.push((name, rect, score));
        }
    }

    Ok(kept)
}

/// 两个框是否"明显是同一个位置"——用重叠面积 / 较小框面积的比例判断,
/// 而不是简单看中心点距离,这样不同尺寸的模板(比如4字名字 vs 6字名字)
/// 也能正确判断是否指向同一段文字。
fn rects_overlap_significantly(a: &TextBox, b: &TextBox) -> bool {
    let ax2 = a.x + a.w;
    let ay2 = a.y + a.h;
    let bx2 = b.x + b.w;
    let by2 = b.y + b.h;

    let ix1 = a.x.max(b.x);
    let iy1 = a.y.max(b.y);
    let ix2 = ax2.min(bx2);
    let iy2 = ay2.min(by2);

    let iw = (ix2 - ix1).max(0);
    let ih = (iy2 - iy1).max(0);
    let inter_area = (iw as i64) * (ih as i64);
    if inter_area <= 0 {
        return false;
    }

    let area_a = (a.w as i64) * (a.h as i64);
    let area_b = (b.w as i64) * (b.h as i64);
    let min_area = area_a.min(area_b).max(1);

    (inter_area as f64) / (min_area as f64) > 0.3
}

/// 🎯 识别率测试专用:直接对"已经裁好的单张候选框图片"做模板匹配,
/// 不走 `match_single_box` 里那套按整帧宽度换算 `scale_factor` 的逻辑。
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

/// 🏆 对一批候选文字框批量识别,只返回"确实匹配上某个白名单模板"的结果。
/// 匹配不上任何模板的框(血条数字、UI 文字、其他玩家名字等)会被自动
/// 丢弃 —— 相当于顺便完成了一轮"这到底是不是怪物名字"的语义级降噪。
pub fn identify_monsters(
    haystack_bgr: &Mat,
    boxes: &[TextBox],
    templates: &[MonsterTemplate],
    min_confidence: f32,
) -> Result<Vec<(String, TextBox, f32)>> {
    let mut results = Vec::new();

    for b in boxes {
        let matches = match_single_box(haystack_bgr, b, templates, min_confidence)?;
        results.extend(matches);
    }

    Ok(results)
}
