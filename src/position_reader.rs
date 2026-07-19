// src/position_reader.rs
//! 角色坐标读取模块 - 通过模板匹配识别小地图下方的"危险 X,Y"坐标数字
//!
//! 💡 为什么要读游戏内坐标,而不是继续用画面像素差异去猜:
//! - 之前用"画面有没有变化"来判断角色动没动,容易被角色待机动画、
//!   环境特效(火把、粒子)干扰,而且没法区分"画面变了多少对应实际
//!   走了多远"。
//! - 游戏 UI 本身就在小地图下方实时显示了坐标数字,直接读这个数字,
//!   准确、直接,还能顺便知道"走了多远"而不只是"动没动"。
//!
//! 识别方式:跟怪物名字识别(monster_matcher.rs)是同一套思路 —— 不用
//! 通用 OCR,而是给 0~9 十个数字 + 逗号建立模板库,对固定位置的坐标
//! 显示区域做连通域分割(每个数字字符是一个连通块),按从左到右的顺序
//! 逐个字符做模板匹配,拼出完整的坐标字符串再解析成两个整数。
//!
//! 📋 使用流程(跟建怪物名字模板库是一样的套路):
//! 1. 先用 `debug_dump_position_roi` 把坐标区域整块裁出来存盘,肉眼确认
//!    ROI 范围是不是刚好框住那串"危险 214,179"文字(不多不少)。
//! 2. 用 `debug_crop_digit_boxes` 把这块区域里检测到的每个数字字符
//!    单独裁剪存盘。
//! 3. 挑出 0~9 十个数字各一张、逗号一张,分别命名成 `0.png`...`9.png`、
//!    `comma.png`,放进 `templates/digits/` 目录。
//! 4. 调用 `load_digit_templates()` 加载,再调用 `read_position()`
//!    读取当前坐标。

use opencv::{
    Result,
    core::{self, Mat, Point, Rect, Scalar, Size, Vector, min_max_loc},
    imgcodecs::{self, IMREAD_COLOR, imread},
    imgproc::{self, TemplateMatchModes, match_template},
    prelude::*,
};
use std::fs;
use std::path::Path;

/// 一个数字/符号的参考模板小图。label 是 "0".."9" 或 ","。
#[derive(Debug, Clone)]
pub struct DigitTemplate {
    pub label: String,
    pub template: Mat,
}

/// 坐标读取的检测参数。ROI 是相对"游戏画面"(不含标题栏)的比例,
/// 跟 monster_detector::DetectorConfig 是同一套设计思路。
#[derive(Debug, Clone)]
pub struct PositionReaderConfig {
    pub roi_left_frac: f64,
    pub roi_top_frac: f64,
    pub roi_right_frac: f64,
    pub roi_bottom_frac: f64,

    // 🎯 坐标数字实际是白色/浅灰色文字(不是"危险"两个字的红色！)，
    // 跟 monster_detector.rs 里怪物名字识别用的是同一套白色文字阈值思路:
    // 在 HSV 空间卡 S(饱和度)足够低 + V(明度)足够高。
    // V 下限别设太高,数字受描边/半透明底板影响,亮度不一定纯白。
    pub s_max: f64,
    pub v_min: f64,

    pub close_kernel_w: i32,
    pub close_kernel_h: i32,

    // 数字字符的连通域尺寸过滤,比怪物名字的文字框小得多
    pub min_h: i32,
    pub max_h: i32,
    pub min_w: i32,
    pub max_w: i32,
    pub min_area: i32,
}

impl Default for PositionReaderConfig {
    fn default() -> Self {
        // ✅ 这套 ROI 比例是用真实截图做像素级红色文字定位实测标定出来的
        // (3000x1716 截图下,"危险 173,84" 文字紧密边界框是
        // x:[2564,2918] y:[448,503],换算比例后各边留了一点余量,
        // 给坐标数字位数变化(比如从个位数变成4位数)留空间)。
        Self {
            roi_left_frac: 0.83,
            roi_top_frac: 0.25,
            roi_right_frac: 0.99,
            roi_bottom_frac: 0.30,

            s_max: 60.0,
            v_min: 150.0,

            // 🎯 闭运算核宽度必须小于任意两个相邻字符之间的天然像素间隙,
            // 否则会把不同数字粘连成一个连通块(比如"164,"整个粘一起)。
            // 数字字体间距很紧凑,横向核基本不需要,设成 1 相当于不做横向桥接;
            // 纵向核用来修补单个数字笔画内部的抗锯齿断裂。
            close_kernel_w: 1,
            close_kernel_h: 3,

            // 🎯 数字字符的连通域尺寸过滤,比怪物名字的文字框小得多。
            // ⚠️ 下限不能卡太高:逗号笔画天然比数字小很多(可能只有几像素
            // 宽高),之前 min_h:8/min_w:3/min_area:15 会把逗号整个刷掉,
            // 导致 chars.split(',') 永远拼不出合法的两段坐标。
            // 这套 ROI 已经做过白色阈值 + 区域裁剪,噪点来源有限,
            // 调低下限风险可控。
            min_h: 3,
            max_h: 30,
            min_w: 1,
            max_w: 40,
            min_area: 3,
        }
    }
}

/// 检测出的候选字符框(相对传入 ROI 的坐标系,不是整张原图坐标)
#[derive(Debug, Clone, Copy)]
struct CharBox {
    x: i32,
    y: i32,
    w: i32,
    h: i32,
}

impl CharBox {
    fn rect(&self) -> Rect {
        Rect::new(self.x, self.y, self.w, self.h)
    }
}

/// 📂 从目录批量加载数字模板。文件名(不含扩展名)就是字符标签,
/// 比如 `templates/digits/0.png` -> label = "0"；逗号请存成
/// `comma.png`(文件系统对逗号当文件名不一定友好),加载时自动转成 ","。
pub fn load_digit_templates<P: AsRef<Path>>(dir: P) -> Result<Vec<DigitTemplate>> {
    let mut templates = Vec::new();

    let entries = match fs::read_dir(&dir) {
        Ok(e) => e,
        Err(_) => {
            println!(
                "   ⚠️ [坐标数字模板库] 目录不存在或无法读取: {}(还没建模板库的话这是正常的)",
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

        let stem = match path.file_stem().and_then(|s| s.to_str()) {
            Some(n) => n.to_string(),
            None => continue,
        };

        let label = if stem == "comma" {
            ",".to_string()
        } else {
            stem
        };

        let template = imread(&path.to_string_lossy(), IMREAD_COLOR)?;
        if template.empty() {
            println!(
                "   ⚠️ [坐标数字模板库] 无法读取模板图片: {}",
                path.display()
            );
            continue;
        }

        println!("   📎 [坐标数字模板库] 已加载模板: '{}'", label);
        templates.push(DigitTemplate { label, template });
    }

    println!(
        "   ✅ [坐标数字模板库] 共加载 {} 个数字/符号模板",
        templates.len()
    );

    Ok(templates)
}

/// 把 ROI 配置换算成原图坐标系下的绝对矩形
fn compute_roi_rect(frame_w: i32, frame_h: i32, cfg: &PositionReaderConfig) -> Rect {
    let x = (frame_w as f64 * cfg.roi_left_frac) as i32;
    let y = (frame_h as f64 * cfg.roi_top_frac) as i32;
    let w = (frame_w as f64 * (cfg.roi_right_frac - cfg.roi_left_frac)) as i32;
    let h = (frame_h as f64 * (cfg.roi_bottom_frac - cfg.roi_top_frac)) as i32;
    Rect::new(x, y, w.max(1), h.max(1))
}

/// 在裁出来的坐标区域小图里,检测出每个数字/逗号字符的候选框
/// (返回坐标是相对这块小图自身的坐标系)。
fn detect_char_boxes(roi_bgr: &Mat, cfg: &PositionReaderConfig) -> Result<Vec<CharBox>> {
    // 🎯 白色文字阈值:转 HSV,卡 S(饱和度)够低 + V(明度)够高。
    // 跟 monster_detector.rs 里怪物名字检测是同一套思路,
    // 因为坐标数字本身就是白色/浅灰色,不是"危险"两个字的红色。
    let mut hsv = Mat::default();
    imgproc::cvt_color(
        roi_bgr,
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
        core::CV_32S,
    )?;

    let mut boxes = Vec::new();
    for i in 1..num {
        let x = *stats.at_2d::<i32>(i, imgproc::CC_STAT_LEFT)?;
        let y = *stats.at_2d::<i32>(i, imgproc::CC_STAT_TOP)?;
        let w = *stats.at_2d::<i32>(i, imgproc::CC_STAT_WIDTH)?;
        let h = *stats.at_2d::<i32>(i, imgproc::CC_STAT_HEIGHT)?;
        let area = *stats.at_2d::<i32>(i, imgproc::CC_STAT_AREA)?;

        if h > cfg.min_h && h < cfg.max_h && w > cfg.min_w && w < cfg.max_w && area > cfg.min_area {
            boxes.push(CharBox { x, y, w, h });
        }
    }

    // 从左到右排序,数字才能拼对顺序
    boxes.sort_by_key(|b| b.x);

    Ok(boxes)
}

/// 对单个字符候选框做模板匹配,识别是哪个数字/逗号
fn match_char(
    roi_bgr: &Mat,
    char_box: &CharBox,
    templates: &[DigitTemplate],
    min_confidence: f32,
    scale_factor: f64,
) -> Result<Option<(String, f32)>> {
    const PADDING: i32 = 2;

    let img_w = roi_bgr.cols();
    let img_h = roi_bgr.rows();

    let x = (char_box.x - PADDING).max(0);
    let y = (char_box.y - PADDING).max(0);
    let w = (char_box.w + PADDING * 2).min(img_w - x);
    let h = (char_box.h + PADDING * 2).min(img_h - y);

    if w <= 0 || h <= 0 {
        return Ok(None);
    }

    let cropped = Mat::roi(roi_bgr, Rect::new(x, y, w, h))?.try_clone()?;

    let mut best_label: Option<String> = None;
    let mut best_score: f32 = 0.0;

    for tpl in templates {
        // 🎯 按窗口实际物理分辨率 vs 模板截图时的基准分辨率,动态缩放模板,
        // 跟 match_icon.rs 里按钮匹配用的是同一套思路 —— 否则窗口一旦被
        // 拖到跟截图时不一样的尺寸,数字模板匹配会整体失效。
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
        if score > best_score {
            best_score = score;
            best_label = Some(tpl.label.clone());
        }
    }

    // 🐛 调试用:把每个候选框实际匹配到的最高分数打出来,方便定位到底是
    // 哪个字符分数不够、差多少 —— 之前"整体识别失败"的日志只会说
    // "未能识别到角色坐标",看不出具体卡在哪一步。
    // println!(
    //     "   🔢 [坐标字符匹配] 候选框(w={},h={}) 最佳匹配: {:?} | 分数: {:.2}% (阈值: {:.2}%)",
    //     char_box.w,
    //     char_box.h,
    //     best_label,
    //     best_score * 100.0,
    //     min_confidence * 100.0
    // );

    if best_score >= min_confidence {
        Ok(best_label.map(|l| (l, best_score)))
    } else {
        Ok(None)
    }
}

/// 🎯 主函数:从当前整帧(BGR Mat)里读出角色实时坐标 (x, y)。
///
/// 每个候选框独立跟模板匹配,匹配不上(分数不够)的直接跳过丢弃,
/// 不会导致整轮识别失败 —— 标签区域偶尔漏出的噪点、或者任何跟
/// 数字/逗号模板对不上的候选框,都会被这一步自然过滤掉。
/// 最后把匹配成功的字符拼起来,尝试解析成 "数字,数字" 格式,
/// 解析不出来(比如逗号数量不对)才判定这一轮读取失败。
pub fn read_position(
    frame_bgr: &Mat,
    cfg: &PositionReaderConfig,
    templates: &[DigitTemplate],
    min_confidence: f32,
) -> Option<(i32, i32)> {
    if templates.is_empty() {
        return None;
    }

    let roi_rect = compute_roi_rect(frame_bgr.cols(), frame_bgr.rows(), cfg);
    let roi_img = match Mat::roi(frame_bgr, roi_rect).and_then(|r| r.try_clone()) {
        Ok(r) => r,
        Err(_) => return None,
    };

    let boxes = match detect_char_boxes(&roi_img, cfg) {
        Ok(b) => b,
        Err(_) => return None,
    };

    if boxes.is_empty() {
        return None;
    }

    // 🎯 按当前整帧的物理宽度 vs 模板截图时的基准分辨率,算出精确缩放
    // 系数,传给 match_char 动态缩放数字模板 —— 跟 match_icon.rs 里
    // 按钮匹配的自适应缩放是同一套思路,避免窗口分辨率一变整套坐标
    // 识别就失效。
    let scale_factor =
        frame_bgr.cols() as f64 / crate::match_icon::TEMPLATE_REFERENCE_PHYSICAL_WIDTH;

    // 🎯 不再用"候选框间距"去猜哪些是标签区域漏出来的噪点(实测证明
    // 这个假设不总成立,数字内部偶尔间距也会比标签间隙大,会误伤真实
    // 数字)。改成让匹配结果本身说话:每个候选框都去跟模板匹配,
    // 匹配不上(分数不够)的直接跳过丢弃,只用匹配成功的字符拼坐标。
    // 真正的噪点去匹配 0~9/逗号模板,分数几乎不可能达标,会被自然
    // 过滤掉;真实数字从实测看置信度普遍在98%以上,不会被误伤。
    let mut chars = String::new();
    for b in &boxes {
        if let Ok(Some((label, _score))) =
            match_char(&roi_img, b, templates, min_confidence, scale_factor)
        {
            chars.push_str(&label);
        }
        // 匹配失败(分数不够/出错)的候选框直接跳过,不中断整轮识别。
    }

    let parts: Vec<&str> = chars.split(',').collect();
    if parts.len() != 2 {
        return None;
    }

    let x: i32 = parts[0].trim().parse().ok()?;
    let y: i32 = parts[1].trim().parse().ok()?;

    Some((x, y))
}

/// 🗂️ 调试用:把坐标区域整块裁出来存盘,肉眼核对 ROI 范围对不对。
pub fn debug_dump_position_roi(
    frame_bgr: &Mat,
    cfg: &PositionReaderConfig,
    out_path: &str,
) -> Result<()> {
    let roi_rect = compute_roi_rect(frame_bgr.cols(), frame_bgr.rows(), cfg);
    let roi_img = Mat::roi(frame_bgr, roi_rect)?.try_clone()?;
    let params = Vector::new();
    imgcodecs::imwrite(out_path, &roi_img, &params)?;
    Ok(())
}

/// 🗂️ 调试/建模板专用:把坐标区域里检测到的每个字符候选框单独裁剪存盘,
/// 方便挑出来建数字模板库。
pub fn debug_crop_digit_boxes(
    frame_bgr: &Mat,
    cfg: &PositionReaderConfig,
    out_dir: &str,
) -> Result<()> {
    let roi_rect = compute_roi_rect(frame_bgr.cols(), frame_bgr.rows(), cfg);
    let roi_img = Mat::roi(frame_bgr, roi_rect)?.try_clone()?;
    let boxes = detect_char_boxes(&roi_img, cfg)?;

    std::fs::create_dir_all(out_dir).ok();

    for (i, b) in boxes.iter().enumerate() {
        let cropped = Mat::roi(&roi_img, b.rect())?.try_clone()?;
        let out_path = format!("{}/char_{:02}_{}_{}.png", out_dir, i, b.x, b.y);
        let params = Vector::new();
        imgcodecs::imwrite(&out_path, &cropped, &params)?;
    }

    println!(
        "   🗂️ [调试] 已把坐标区域里 {} 个候选字符单独裁剪存到目录: {}",
        boxes.len(),
        out_dir
    );

    Ok(())
}
