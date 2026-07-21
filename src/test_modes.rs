// src/test_modes.rs
//! 🧪 单模块测试循环。
//!
//! `App::init()` 完成的"前提"(找窗口 + 定位按钮资产并缓存)照常只跑一次,
//! 但这里每个函数只跑"识别"这一部分,不碰鼠标——不攻击、不拾取、不移动,
//! 纯粹是"截图 -> 跑某一个识别模块 -> 打印结果 -> 下一轮",方便单独盯着
//! 某个模块的识别效果调参数/建模板,不用每次都跑完整挂机逻辑。
//!
//! main.rs 通过 `TEST_MODULE` 环境变量选择跑哪一个,不设置就跑完整
//! `bot_loop::run()`。用法见 main.rs 顶部注释。

use crate::app::App;
use crate::monster_detector;
use crate::position_reader;
use crate::text_ocr;
use crate::util;
use opencv::prelude::MatTraitConst;
use std::thread::sleep;
use std::time::Duration;

/// 每轮截图之间的间隔,跟 `bot_loop::run` 保持一致,方便对比表现。
const LOOP_INTERVAL: Duration = Duration::from_millis(1200);

/// 截图失败/转 Mat 失败时统一的"跳过本轮"处理:打印原因、睡一轮间隔。
fn skip_round(reason: &str) {
    println!("⚠️  {}", reason);
    sleep(LOOP_INTERVAL);
}

/// 🧪 第一步 —— 只截取候选文字框,存盘,不做任何模板匹配。
/// 对应"先保存 OpenCV 截取的白色部分"这一步:每一轮把检测到的候选框
/// 全部裁剪存到 `DEBUG_MONSTER_CROPS/`,连同一张标注大图
/// `DEBUG_MONSTER_BOXES.png`(方便对照每个裁剪图在画面里的具体位置)。
/// 这里不需要模板库、不需要判断白名单,单纯就是"存图"。
pub fn run_monster_capture(app: &mut App) {
    println!("🧪 [测试模式] 怪物识别 · 第1步:截取候选框存盘 —— 不做模板匹配\n");
    println!("   裁剪图会持续存到: {}", util::get_debug_crop_dir());
    println!(
        "   标注大图: {}/DEBUG_MONSTER_BOXES.png\n",
        env!("CARGO_MANIFEST_DIR")
    );

    loop {
        let Some((raw_rgba, width, height)) = util::capture_window(&app.window) else {
            skip_round("[截图] 失败，跳过本次循环");
            continue;
        };

        let mat = match monster_detector::rgba_bytes_to_bgr_mat(&raw_rgba, width, height) {
            Ok(m) => m,
            Err(e) => {
                skip_round(&format!("[怪物检测] RGBA 转 Mat 失败: {:?}", e));
                continue;
            }
        };

        let boxes = match monster_detector::detect_monster_names(&mat, &app.cfg) {
            Ok(b) => b,
            Err(e) => {
                skip_round(&format!("[怪物检测] 失败: {:?}", e));
                continue;
            }
        };
        println!("🔍 本轮检测到 {} 个候选文字框，存盘中...", boxes.len());

        let box_img_path = format!("{}/DEBUG_MONSTER_BOXES.png", env!("CARGO_MANIFEST_DIR"));
        if let Err(e) = monster_detector::debug_dump_boxes(&mat, &boxes, &box_img_path) {
            println!("   ⚠️ 标注大图存盘失败: {:?}", e);
        }
        let crop_dir = util::get_debug_crop_dir();
        if let Err(e) = monster_detector::debug_crop_boxes(&mat, &boxes, &crop_dir) {
            println!("   ⚠️ 候选框裁剪存盘失败: {:?}", e);
        }

        println!();
        sleep(LOOP_INTERVAL);
    }
}

/// 🧪 第三步 —— 只做模板匹配验证,不存盘。
/// 对应"人工确认 + 标识保存到 monster_names 之后,跑程序验证"这一步:
/// 加载 `templates/monster_names/` 里的模板,对每一轮检测到的候选框做
/// 匹配,打印识别到的怪物名字/位置/置信度,以及是否在挂机白名单内。
pub fn run_monster_verify(app: &mut App) {
    println!("🧪 [测试模式] 怪物识别 · OCR 识别验证 —— 不存盘\n");

    if app.text_ocr_recognizer.is_none() {
        println!("❌ OCR引擎未初始化(models/ 目录下的模型文件缺失或加载失败),无法测试。");
        return;
    }

    loop {
        let Some((raw_rgba, width, height)) = util::capture_window(&app.window) else {
            skip_round("[截图] 失败，跳过本次循环");
            continue;
        };

        let mat = match monster_detector::rgba_bytes_to_bgr_mat(&raw_rgba, width, height) {
            Ok(m) => m,
            Err(e) => {
                skip_round(&format!("[怪物检测] RGBA 转 Mat 失败: {:?}", e));
                continue;
            }
        };

        // 上面已经检查过 recognizer 存在,这里 unwrap 是安全的。
        let recognizer = app.text_ocr_recognizer.as_ref().unwrap();
        let blocks = match recognizer.recognize_frame(&mat, &app.text_ocr_cfg) {
            Ok(b) => b,
            Err(e) => {
                println!("⚠️  [怪物识别] OCR 识别失败: {:?}", e);
                sleep(LOOP_INTERVAL);
                continue;
            }
        };

        let identified = text_ocr::match_monsters(
            &blocks,
            &app.live_status.allowed_monster_set,
            &app.text_ocr_cfg,
        );
        if identified.is_empty() {
            println!("   （本轮 OCR 没有识别出任何白名单怪物名字）");
        } else {
            for (name, b, score) in &identified {
                println!(
                    "   👾 {} ({:.2}%) | 位置: ({}, {})",
                    name,
                    score * 100.0,
                    b.x,
                    b.y
                );
            }
        }

        println!();
        sleep(LOOP_INTERVAL);
    }
}

/// 🧪 只测物品识别:OCR 读文字 + 白名单模糊匹配,打印识别到的物品名字/
/// OCR 原文/置信度。
pub fn run_item_test(app: &mut App) {
    println!("🧪 [测试模式] 物品识别 —— 只跑 text_ocr,不攻击/不拾取/不移动\n");

    if app.text_ocr_recognizer.is_none() {
        println!("❌ OCR 引擎未初始化(models/ 目录下的模型文件缺失或加载失败),无法测试物品识别。");
        return;
    }

    loop {
        let Some((raw_rgba, width, height)) = util::capture_window(&app.window) else {
            skip_round("[截图] 失败，跳过本次循环");
            continue;
        };

        let mat = match monster_detector::rgba_bytes_to_bgr_mat(&raw_rgba, width, height) {
            Ok(m) => m,
            Err(e) => {
                skip_round(&format!("[物品OCR] RGBA 转 Mat 失败: {:?}", e));
                continue;
            }
        };

        // 上面已经检查过 recognizer 存在,这里 unwrap 是安全的。
        let recognizer = app.text_ocr_recognizer.as_ref().unwrap();
        let blocks = match recognizer.recognize_frame(&mat, &app.text_ocr_cfg) {
            Ok(b) => b,
            Err(e) => {
                println!("⚠️  [物品OCR] 识别失败: {:?}", e);
                sleep(LOOP_INTERVAL);
                continue;
            }
        };

        let matched = text_ocr::match_items(
            &blocks,
            &app.live_status.allowed_item_set,
            &app.text_ocr_cfg,
        );
        if matched.is_empty() {
            println!("   （本轮没有识别出任何白名单物品）");
        } else {
            for (item_name, ocr_text, conf) in &matched {
                println!(
                    "   💎 {} (OCR原文: \"{}\" | 置信度: {:.2}%)",
                    item_name,
                    ocr_text,
                    conf * 100.0
                );
            }
        }

        println!();
        sleep(LOOP_INTERVAL);
    }
}

/// 🧪 只测坐标识别:读取角色当前 (x, y)。
pub fn run_position_test(app: &mut App) {
    println!("🧪 [测试模式] 坐标识别 —— 只跑 position_reader,不攻击/不拾取/不移动\n");

    if app.digit_templates.is_empty() {
        println!("⚠️  数字模板库(templates/digits/)是空的,识别结果大概率会一直是空。");
    }

    loop {
        let Some((raw_rgba, width, height)) = util::capture_window(&app.window) else {
            skip_round("[截图] 失败，跳过本次循环");
            continue;
        };

        let mat = match monster_detector::rgba_bytes_to_bgr_mat(&raw_rgba, width, height) {
            Ok(m) => m,
            Err(e) => {
                skip_round(&format!("[坐标识别] RGBA 转 Mat 失败: {:?}", e));
                continue;
            }
        };

        match position_reader::read_position(&mat, &app.pos_cfg, &app.digit_templates, 0.7) {
            Some((x, y)) => println!("📍 当前坐标: ({}, {})", x, y),
            None => println!("   （本轮未能读出坐标）"),
        }

        sleep(LOOP_INTERVAL);
    }
}

/// 🧪 只测地图名字识别:识别当前所在地图。
pub fn run_map_test(app: &mut App) {
    println!("🧪 [测试模式] 地图识别 · OCR 识别验证\n");

    if app.text_ocr_recognizer.is_none() {
        println!("❌ OCR引擎未初始化(models/ 目录下的模型文件缺失或加载失败),无法测试。");
        return;
    }

    loop {
        let Some((raw_rgba, width, height)) = util::capture_window(&app.window) else {
            skip_round("[截图] 失败，跳过本次循环");
            continue;
        };

        let mat = match monster_detector::rgba_bytes_to_bgr_mat(&raw_rgba, width, height) {
            Ok(m) => m,
            Err(e) => {
                skip_round(&format!("[地图识别] RGBA 转 Mat 失败: {:?}", e));
                continue;
            }
        };

        let recognizer = app.text_ocr_recognizer.as_ref().unwrap();
        let blocks = match recognizer.recognize_frame(&mat, &app.text_ocr_cfg) {
            Ok(b) => b,
            Err(e) => {
                println!("⚠️  [地图识别] OCR 识别失败: {:?}", e);
                sleep(LOOP_INTERVAL);
                continue;
            }
        };

        match text_ocr::match_map_name(&blocks, mat.cols(), mat.rows(), &app.text_ocr_cfg) {
            Some((name, score)) => println!("🗺️  当前地图: {} ({:.2}%)", name, score * 100.0),
            None => println!("   （本轮未能从牌匾区域识别出地图名字）"),
        }

        sleep(LOOP_INTERVAL);
    }
}
