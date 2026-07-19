// src/bot_loop.rs
//! 主循环:每一轮截图 -> 识别(怪物候选框/坐标/地图名字/掉落物品)->
//! 决策(拾取 > 攻击 > 移动)。
//!
//! 原来这一大段全部挤在 main() 的 `loop { ... }` 里,一层套一层的
//! match/if-let 让人很难一眼看出"这一轮到底按什么顺序做了哪些事"。
//! 现在按职责拆成一系列小函数,`run()` 里只保留最外层的编排顺序,
//! 想改某一步的逻辑(比如物品识别)直接找对应的函数即可,不用在几百
//! 行里定位缩进层级。

use std::thread::sleep;
use std::time::Duration;

use opencv::core::Mat;

use crate::app::{App, STUCK_CHECK_INTERVAL, STUCK_MOVE_EPSILON};
use crate::game_status::EntityInfo;
use crate::map_matcher;
use crate::map_nav;
use crate::monster_detector::{self, TextBox};
use crate::monster_matcher;
use crate::mouse_action;
use crate::position_reader;
use crate::util;

/// 循环检测怪物:发现白名单怪物就攻击，没发现就移动寻路。
/// 对应原来 main() 里最后那段 `loop { ... }`。
pub fn run(app: &mut App) {
    loop {
        match util::capture_window(&app.window) {
            Some((raw_rgba, width, height)) => {
                run_one_frame(app, &raw_rgba, width, height);
            }
            None => println!("⚠️  [截图] 失败，跳过本次循环"),
        }

        sleep(Duration::from_millis(1200));
    }
}

/// 处理单帧截图:识别 + 决策。跟原来的 `Ok((raw_rgba, width, height)) =>` 分支一一对应。
fn run_one_frame(app: &mut App, raw_rgba: &[u8], width: u32, height: u32) {
    let boxes = match monster_detector::detect_monsters_from_rgba(raw_rgba, width, height, &app.cfg)
    {
        Ok(boxes) => boxes,
        Err(e) => {
            println!("⚠️  [怪物检测] 失败: {:?}", e);
            return;
        }
    };
    if app.live_status.logging.capture {
        println!("🔍 [循环检测] 本次检测到 {} 个候选文字框", boxes.len());
    }

    if app.debug_monster {
        dump_monster_debug(raw_rgba, width, height, &boxes);
    }

    let bgr_mat = monster_detector::rgba_bytes_to_bgr_mat(raw_rgba, width, height);

    dump_position_debug_once(app, &bgr_mat);
    let current_position = read_current_position(app, &bgr_mat);

    dump_and_identify_map(app, &bgr_mat);

    // 💰 拾取优先级最高:发现白名单物品就直接点拾取按钮,本轮剩余的
    // 打怪/移动判断就都跳过。
    let picked_up = try_pick_up_items(app, &bgr_mat);

    if !picked_up {
        let should_attack = detect_monsters_and_should_attack(app, &bgr_mat, &boxes);

        if should_attack {
            attack(app);
        } else {
            maybe_move(app, current_position);
        }
    }

    // 📥 不管这一轮走的是拾取/攻击/移动哪条分支,都统一打印一次当前的
    // 完整状态快照,方便你一眼看清地图/坐标/怪物/物品这几项实时数据。
    if app.live_status.logging.status {
        app.live_status.print_debug_status();
    }

    if picked_up {
        // 拾取那 2 秒等待已经在 try_pick_up_items 里睡过了,这里提前
        // return,跳过外层循环末尾那次额外的 1200ms 等待,尽快进入下一轮。
        return;
    }
}

/// 🐛 DEBUG_MONSTER=1 时,把本轮检测到的候选框(标注大图 + 逐个单独裁剪)
/// 持续导出,方便你挑出目标怪物名字建模板。
fn dump_monster_debug(raw_rgba: &[u8], width: u32, height: u32, boxes: &[TextBox]) {
    if let Ok(mat) = monster_detector::rgba_bytes_to_bgr_mat(raw_rgba, width, height) {
        let box_img_path = format!("{}/DEBUG_MONSTER_BOXES.png", env!("CARGO_MANIFEST_DIR"));
        if let Err(e) = monster_detector::debug_dump_boxes(&mat, boxes, &box_img_path) {
            println!("⚠️  [怪物调试] 标注大图存盘失败: {:?}", e);
        }

        let crop_dir = util::get_debug_crop_dir();
        if let Err(e) = monster_detector::debug_crop_boxes(&mat, boxes, &crop_dir) {
            println!("⚠️  [怪物调试] 候选框裁剪存盘失败: {:?}", e);
        }
    }
}

/// 🐛 数字模板库还没建的话,存一份调试图帮你校准(只存一次,不刷屏)。
fn dump_position_debug_once(app: &mut App, bgr_mat: &opencv::Result<Mat>) {
    if app.dumped_position_debug {
        return;
    }
    if let Ok(mat) = bgr_mat {
        let roi_path = util::get_debug_position_roi_path();
        if let Err(e) = position_reader::debug_dump_position_roi(mat, &app.pos_cfg, &roi_path) {
            println!("⚠️  [坐标调试] ROI 裁剪存盘失败: {:?}", e);
        } else if app.live_status.logging.position {
            println!(
                "✅ [坐标调试] 已保存坐标区域截图到 {}，请核对是否刚好框住\"危险 X,Y\"文字",
                roi_path
            );
        }

        let crop_dir = util::get_debug_digit_crop_dir();
        if let Err(e) = position_reader::debug_crop_digit_boxes(mat, &app.pos_cfg, &crop_dir) {
            println!("⚠️  [坐标调试] 数字字符裁剪存盘失败: {:?}", e);
        }
    }
    app.dumped_position_debug = true;
}

/// 📍 读取角色实时坐标(数字模板库为空时恒为 None),并同步写入 live_status。
fn read_current_position(app: &mut App, bgr_mat: &opencv::Result<Mat>) -> Option<(i32, i32)> {
    let current_position = bgr_mat.as_ref().ok().and_then(|mat| {
        position_reader::read_position(mat, &app.pos_cfg, &app.digit_templates, 0.7)
    });

    if let Some((px, py)) = current_position {
        app.live_status.update_position(px, py);
        if app.live_status.logging.position {
            println!("📍 [坐标] 当前位置: ({}, {})", px, py);
        }
    }

    current_position
}

/// 🗺️ 识别当前地图名字(跟坐标/怪物识别同一帧、同一频率，避免"旧地图
/// 名字 + 新怪物列表"这种状态不一致的情况)。
///
/// 🐛 调试图导出不依赖模板库是否存在:哪怕 templates/map_names/ 还是空
/// 目录(还没开始建模板)，也应该正常存出 DEBUG_MAP_ROI.png 方便你核对
/// ROI 范围。
fn dump_and_identify_map(app: &mut App, bgr_mat: &opencv::Result<Mat>) {
    let Ok(mat) = bgr_mat else { return };

    if !app.dumped_map_debug {
        let map_roi_path = util::get_debug_map_roi_path();
        if let Err(e) = map_matcher::debug_dump_map_roi(mat, &app.map_cfg, &map_roi_path) {
            println!("⚠️  [地图调试] ROI 裁剪存盘失败: {:?}", e);
        } else if app.live_status.logging.map {
            println!(
                "✅ [地图调试] 已保存地图名字区域截图到 {}，请核对是否刚好框住地图名字",
                map_roi_path
            );
        }
        app.dumped_map_debug = true;
    }

    if app.map_templates.is_empty() {
        return;
    }

    match map_matcher::identify_map(mat, &app.map_cfg, &app.map_templates, 0.8) {
        Ok(Some((name, score))) => {
            if app.live_status.logging.map {
                println!(
                    "🗺️  [地图识别] 当前地图: {} | 置信度: {:.2}%",
                    name,
                    score * 100.0
                );
            }
            app.live_status.update_map_name(name);
        }
        Ok(None) => {
            if app.live_status.logging.map {
                println!("⚠️  [地图识别] 未能匹配到任何已知地图名字模板");
            }
        }
        Err(e) => println!("⚠️  [地图识别] 识别失败: {:?}", e),
    }
}

/// 💰 检测地面掉落物品(OCR 识别文字 + 白名单模糊匹配)。
/// 发现白名单物品就点击拾取按钮并等待角色跑过去,返回 `true` 表示
/// "本轮循环应该到此为止,跳过后面的打怪/移动判断"。
fn try_pick_up_items(app: &mut App, bgr_mat: &opencv::Result<Mat>) -> bool {
    let (Ok(mat), Some(recognizer)) = (bgr_mat, &app.item_ocr_recognizer) else {
        return false;
    };

    let matched =
        match recognizer.detect_items(mat, &app.item_ocr_cfg, &app.live_status.allowed_item_set) {
            Ok(matched) => matched,
            Err(e) => {
                println!("⚠️  [物品OCR] 识别失败: {:?}", e);
                return false;
            }
        };

    if matched.is_empty() {
        app.live_status.update_items(Vec::new());
        return false;
    }

    if app.live_status.logging.item {
        println!("💰 [拾取] 发现 {} 个白名单物品", matched.len());
        for (item_name, ocr_text, conf) in &matched {
            println!(
                "   └── 💎 {} (OCR原文: {} | 置信度: {:.2}%)",
                item_name,
                ocr_text,
                conf * 100.0
            );
        }
    }

    // ⚠️ OCR 目前只返回识别到的文字内容,没有解析具体屏幕坐标(拾取本身
    // 也不需要精确坐标,点固定的"拾取"按钮就行),这里 screen_x/y 先占位成 0。
    let entities: Vec<EntityInfo> = matched
        .iter()
        .map(|(name, _, conf)| EntityInfo {
            name: name.clone(),
            screen_x: 0,
            screen_y: 0,
            confidence: *conf,
        })
        .collect();
    app.live_status.update_items(entities);

    let Some(btn_pick_up) = app.coordinates_cache.get("pick_up") else {
        return false;
    };
    let (Some(x), Some(y)) = (btn_pick_up.screen_x, btn_pick_up.screen_y) else {
        return false;
    };

    mouse_action::click_at(&mut app.enigo, x, y, "【自动拾取】发现白名单物品");

    // 🎯 拾取优先级 > 自动攻击 > 自动寻路移动。点击拾取按钮后固定等待
    // 2秒,让角色有时间跑过去把物品捡起来,这段时间内不做怪物攻击/移动
    // 判断(不会打断已经在进行的自动战斗,只是这一轮循环不再额外发出
    // 攻击/移动指令),直接跳到下一轮循环重新截图判断。
    if app.live_status.logging.item {
        println!("⏳ [拾取] 等待角色跑过去拾取(约2秒)，本轮跳过打怪/寻路判断...");
    }
    sleep(Duration::from_secs(2));
    true
}

/// 🎯 识别怪物候选框里有没有白名单怪物,决定这一轮要不要发起攻击。
fn detect_monsters_and_should_attack(
    app: &mut App,
    bgr_mat: &opencv::Result<Mat>,
    boxes: &[TextBox],
) -> bool {
    if app.templates.is_empty() {
        if !boxes.is_empty() {
            if app.live_status.logging.monster {
                println!("⚠️  [自动攻击] 模板库为空，检测到候选文字框，先尝试点击攻击按钮。");
            }
            return true;
        }
        if app.live_status.logging.monster {
            println!("⏳ [自动攻击] 当前未检测到怪物候选文字框，继续轮询...");
        }
        return false;
    }

    let mat = match bgr_mat {
        Ok(mat) => mat,
        Err(e) => {
            println!("⚠️  [调试] RGBA 转 Mat 失败: {:?}", e);
            return false;
        }
    };

    let identified = match monster_matcher::identify_monsters(mat, boxes, &app.templates, 0.75) {
        Ok(identified) => identified,
        Err(e) => {
            println!("⚠️  [怪物识别] 识别失败: {:?}", e);
            return false;
        }
    };

    let allowed_monsters: Vec<_> = identified
        .into_iter()
        .filter(|(name, _, _)| app.live_status.allowed_monster_set.contains(name))
        .collect();

    if allowed_monsters.is_empty() {
        if app.live_status.logging.monster {
            println!("⏳ [自动攻击] 本次识别到的怪物不在白名单内，继续轮询...");
        }
        app.live_status.update_monsters(Vec::new());
        return false;
    }

    if app.live_status.logging.monster {
        println!(
            "🎯 [自动攻击] 发现 {} 个白名单怪物，准备攻击...",
            allowed_monsters.len()
        );
        for (name, b, score) in &allowed_monsters {
            println!(
                "   └── 👾 {}({:.2}%) | 位置: ({}, {})",
                name,
                score * 100.0,
                b.x,
                b.y
            );
        }
    }

    // 📦 写入 game_status 缓存,而不是识别完就扔
    let entities: Vec<EntityInfo> = allowed_monsters
        .iter()
        .map(|(name, b, score)| EntityInfo {
            name: name.clone(),
            screen_x: b.x,
            screen_y: b.y,
            confidence: *score,
        })
        .collect();
    app.live_status.update_monsters(entities);

    true
}

/// ⚔️ 点击攻击按钮,并清空卡住检测器状态,避免战斗结束后用旧的基准
/// 坐标误判"没动"。
fn attack(app: &mut App) {
    // 🎯 发现目标:清空卡住检测器状态,避免战斗结束后用旧的基准坐标误判"没动"。
    app.live_status.reset_movement_check();

    let Some(btn_attack) = app.coordinates_cache.get("attack") else {
        println!("⚠️  [自动攻击] 未能在资产缓存中找到攻击按钮。");
        return;
    };
    let (Some(x), Some(y)) = (btn_attack.screen_x, btn_attack.screen_y) else {
        println!("⚠️  [自动攻击] 攻击按钮坐标未缓存，无法点击。");
        return;
    };

    if app.live_status.logging.monster {
        println!("⚔️  [自动攻击] 点击攻击按钮: {}", btn_attack.name);
    }
    mouse_action::click_at(&mut app.enigo, x, y, "【自动攻击】大剑普通攻击");
}

/// 🚶 没发现目标时,根据角色真实坐标有没有变化决定要不要打开大地图
/// 重新选一个目标点:
/// - 坐标在变 -> 角色正在自动走路,什么都不做,继续等
/// - 坐标没变 -> 已经走到目标点停下了,或者压根没在动(刚进入这个状态/
///   卡住了),该打开大地图选一个新目标点,点击后靠引擎自动寻路把角色带过去
fn maybe_move(app: &mut App, current_position: Option<(i32, i32)>) {
    if !app.live_status.has_position_stalled(
        current_position,
        STUCK_MOVE_EPSILON,
        STUCK_CHECK_INTERVAL,
    ) {
        if app.live_status.logging.movement {
            println!("🚶 [移动] 角色坐标正在变化或仍在观察窗口内,继续等待...");
        }
        return;
    }

    if !app.map_nav_supported {
        if app.live_status.logging.movement {
            println!("⚠️  [移动] 地图导航不可用,当前没有可用的移动方式,原地等待...");
        }
        return;
    }

    if app.live_status.logging.movement {
        println!("🗺️  [移动] 角色坐标未变化,打开大地图选取新目标点...");
    }
    match map_nav::navigate_to_random_point(&app.window, &mut app.enigo, &app.nav_cfg) {
        Ok(true) => {
            if app.live_status.logging.movement {
                println!("✅ [移动] 已触发引擎自动寻路,交给引擎接管这段路");
            }
        }
        Ok(false) => {
            if app.live_status.logging.movement {
                println!("⚠️  [移动] 当前地图不支持该流程,本次会话后续不再尝试自动寻路移动");
            }
            app.map_nav_supported = false;
        }
        Err(e) => println!("⚠️  [移动] 执行出错: {:?}", e),
    }
}
