// src/main.rs
use enigo::{Enigo, Settings};
use std::thread::sleep;
use std::time::Duration;

mod asset_loader;
mod find_window;
mod game_status;
mod item_ocr;
mod map_matcher;
mod map_nav;
mod match_icon;
mod monster_detector;
mod monster_matcher;
mod mouse_action;
mod position_reader;
mod quit_game_bot;
mod util;

fn main() {
    println!("🚀 游戏自动化辅助主程序已启动...");
    println!("⌨️  随时长按 ESC 键满 3 秒即可强制退出...\n");

    // 1. 启动并行后台键盘强杀模块
    quit_game_bot::QuitWatchdog::start_async_loop();

    // 2. 寻找游戏窗口
    let game_title = "24luling";
    let window = find_window::require_game_window(game_title);

    // 3. 📸 静态资产定位
    // ⚠️ 必须在做任何鼠标点击之前完成:这一步只依赖纯净的截图,
    // 不需要也不能碰鼠标。之前把"窗口焦点点击"挪到这一步前面,
    // 导致点击本身改变了游戏画面(角色转向/移动/UI变化等副作用),
    // 紧接着截的图就跟模板对不上了,相似度大幅下降。
    let coordinates_cache = match asset_loader::load_and_cache_assets(&window) {
        Some(cache) => cache,
        None => {
            println!("❌ [严重错误] 基础图标资产未能全部定位，无法进行后续挂机测试，程序退出！");
            std::process::exit(1);
        }
    };

    // 4. 🖱️ 资产匹配完成后,再初始化鼠标并激活游戏窗口的输入焦点。
    // enigo 走的是系统级鼠标事件,不是"发给某个窗口"的定向事件,
    // 如果游戏窗口当前不是系统焦点/前台窗口,后续所有点击可能根本
    // 不会被这个窗口接收到。放在资产匹配之后做,就不会污染匹配用的画面。
    let mut enigo = Enigo::new(&Settings::default()).expect("无法初始化 Enigo");
    {
        let focus_x = window.x() + window.width() as i32 / 2;
        let focus_y = window.y() + window.height() as i32 / 2;
        println!(
            "🖱️  [窗口激活] 点击窗口中心以获取输入焦点: ({}, {})",
            focus_x, focus_y
        );
        mouse_action::click_at(&mut enigo, focus_x, focus_y, "【初始化】激活游戏窗口焦点");
        sleep(Duration::from_millis(300));
    }

    // 5. 初始化识别环境
    let mut live_status = game_status::GameStatusCache::new();
    let cfg = monster_detector::DetectorConfig::default();
    let pos_cfg = position_reader::PositionReaderConfig::default();

    let template_dir = util::get_monster_name_template_dir();
    let templates = match monster_matcher::load_monster_templates(&template_dir) {
        Ok(list) => list,
        Err(e) => {
            println!("⚠️  [怪物模板库] 加载失败: {:?}", e);
            Vec::new()
        }
    };
    if templates.is_empty() {
        println!(
            "ℹ️  [怪物识别] 模板库为空，当前仅通过候选文字框检测是否有怪物。建议先构建 templates/monster_names/ 模板库。"
        );
    }

    let digit_template_dir = util::get_digit_template_dir();
    let digit_templates = match position_reader::load_digit_templates(&digit_template_dir) {
        Ok(list) => list,
        Err(e) => {
            println!("⚠️  [坐标数字模板库] 加载失败: {:?}", e);
            Vec::new()
        }
    };
    if digit_templates.is_empty() {
        println!(
            "ℹ️  [坐标识别] 数字模板库为空，暂时无法读取角色实时坐标，移动模块将退化成\"只按方位试探,不做卡住判断\"。建议先构建 templates/digits/ 模板库。"
        );
    }

    let map_cfg = map_matcher::MapReaderConfig::default();
    let map_template_dir = util::get_map_name_template_dir();
    let map_templates = match map_matcher::load_map_templates(&map_template_dir) {
        Ok(list) => list,
        Err(e) => {
            println!("⚠️  [地图名字模板库] 加载失败: {:?}", e);
            Vec::new()
        }
    };
    if map_templates.is_empty() {
        println!(
            "ℹ️  [地图识别] 地图名字模板库为空，暂时无法识别当前地图。建议先构建 templates/map_names/ 模板库。"
        );
    }

    let item_ocr_cfg = item_ocr::ItemOcrConfig::default();
    // 🎯 OCR 引擎只在启动时加载一次(det/rec 模型 + 字典),不要每轮循环
    // 都重新加载,模型文件需要你提前下载放到 CARGO_MANIFEST_DIR/models/
    let item_ocr_dir = format!("{}/models", env!("CARGO_MANIFEST_DIR"));
    let item_ocr_recognizer = match item_ocr::ItemOcrRecognizer::new(
        &format!("{}/PP-OCRv6_medium_det.mnn", item_ocr_dir),
        &format!("{}/PP-OCRv6_medium_rec.mnn", item_ocr_dir),
        &format!("{}/ppocr_keys_v6_medium.txt", item_ocr_dir),
    ) {
        Ok(r) => Some(r),
        Err(e) => {
            println!(
                "⚠️  [物品OCR] 引擎初始化失败: {:?}，本次会话将无法识别物品。请确认 models/ 目录下模型文件是否齐全。",
                e
            );
            None
        }
    };
    // 🐛 地图名字 ROI 位置目前是目测估算,还没有 DEBUG 图核实过,第一轮先存一张调试图
    let mut dumped_map_debug = false;
    // 数字模板库还没建好的话,第一次跑先把坐标区域裁出来,方便你校准 ROI + 建模板
    let mut dumped_position_debug = !digit_templates.is_empty();

    // 🗺️ 大地图导航(复用引擎自带寻路能力)相关状态。
    let nav_cfg = map_nav::NavConfig::default();
    // 有的地图可能不支持这套"打开大地图点选点"流程(比如面板布局不同、
    // 关闭按钮识别不到),一旦确认不支持就整段会话都不再重试,
    // 直接放弃移动(挂机原地打怪/等下一轮识别),避免每轮循环都浪费时间空跑一次。
    let mut map_nav_supported = true;
    // ⚠️ check_interval(4秒)和 move_epsilon(5.0)需要实测调整。
    // 具体的"坐标是否停滞"判断状态缓存在 live_status(GameStatusCache)里,
    // 跟其他实时信息(血量/坐标/怪物/地图名字)放在一起统一管理,
    // 而不是在这里单独再维护一套平行状态。
    const STUCK_CHECK_INTERVAL: Duration = Duration::from_secs(4);
    const STUCK_MOVE_EPSILON: f64 = 5.0;

    // 🐛 支持通过环境变量持续导出怪物候选框调试图,方便建怪物名字模板库。
    // 跟坐标/地图不同,建怪物模板往往需要站在目标怪物附近多跑几轮循环、
    // 让画面里出现你要的怪物,所以这里做成"开关打开就每轮都导出",而不是
    // 只导出一次,你可以跑一会儿再从 DEBUG_MONSTER_CROPS/ 目录里挑图。
    // 用法: DEBUG_MONSTER=1 cargo run
    let debug_monster = std::env::var("DEBUG_MONSTER").is_ok();

    // 5. 循环检测怪物:发现白名单怪物就攻击，没发现就移动寻路
    loop {
        match util::capture_window(&window) {
            Some((raw_rgba, width, height)) => {
                match monster_detector::detect_monsters_from_rgba(&raw_rgba, width, height, &cfg) {
                    Ok(boxes) => {
                        println!("🔍 [循环检测] 本次检测到 {} 个候选文字框", boxes.len());

                        // 🐛 DEBUG_MONSTER=1 时,把本轮检测到的候选框(标注大图 +
                        // 逐个单独裁剪)持续导出,方便你挑出目标怪物名字建模板。
                        if debug_monster {
                            if let Ok(mat) =
                                monster_detector::rgba_bytes_to_bgr_mat(&raw_rgba, width, height)
                            {
                                let box_img_path = format!(
                                    "{}/DEBUG_MONSTER_BOXES.png",
                                    env!("CARGO_MANIFEST_DIR")
                                );
                                if let Err(e) =
                                    monster_detector::debug_dump_boxes(&mat, &boxes, &box_img_path)
                                {
                                    println!("⚠️  [怪物调试] 标注大图存盘失败: {:?}", e);
                                }

                                let crop_dir = util::get_debug_crop_dir();
                                if let Err(e) =
                                    monster_detector::debug_crop_boxes(&mat, &boxes, &crop_dir)
                                {
                                    println!("⚠️  [怪物调试] 候选框裁剪存盘失败: {:?}", e);
                                }
                            }
                        }

                        let mut should_attack = false;
                        let bgr_mat =
                            monster_detector::rgba_bytes_to_bgr_mat(&raw_rgba, width, height);

                        // 🐛 数字模板库还没建的话,存一份调试图帮你校准(只存一次,不刷屏)
                        if !dumped_position_debug {
                            if let Ok(mat) = &bgr_mat {
                                let roi_path = util::get_debug_position_roi_path();
                                if let Err(e) = position_reader::debug_dump_position_roi(
                                    mat, &pos_cfg, &roi_path,
                                ) {
                                    println!("⚠️  [坐标调试] ROI 裁剪存盘失败: {:?}", e);
                                } else {
                                    println!(
                                        "✅ [坐标调试] 已保存坐标区域截图到 {}，请核对是否刚好框住\"危险 X,Y\"文字",
                                        roi_path
                                    );
                                }

                                let crop_dir = util::get_debug_digit_crop_dir();
                                if let Err(e) = position_reader::debug_crop_digit_boxes(
                                    mat, &pos_cfg, &crop_dir,
                                ) {
                                    println!("⚠️  [坐标调试] 数字字符裁剪存盘失败: {:?}", e);
                                }
                            }
                            dumped_position_debug = true;
                        }

                        // 📍 读取角色实时坐标(数字模板库为空时恒为 None)
                        let current_position = bgr_mat.as_ref().ok().and_then(|mat| {
                            position_reader::read_position(mat, &pos_cfg, &digit_templates, 0.7)
                        });

                        if let Some((px, py)) = current_position {
                            live_status.update_position(px, py);
                            println!("📍 [坐标] 当前位置: ({}, {})", px, py);
                        }

                        // 🗺️ 识别当前地图名字(跟坐标/怪物识别同一帧、同一频率，
                        // 避免"旧地图名字 + 新怪物列表"这种状态不一致的情况)
                        //
                        // 🐛 调试图导出不依赖模板库是否存在:哪怕
                        // templates/map_names/ 还是空目录(还没开始建模板)，
                        // 也应该正常存出 DEBUG_MAP_ROI.png 方便你核对 ROI 范围。
                        if let Ok(mat) = &bgr_mat {
                            if !dumped_map_debug {
                                let map_roi_path = util::get_debug_map_roi_path();
                                if let Err(e) =
                                    map_matcher::debug_dump_map_roi(mat, &map_cfg, &map_roi_path)
                                {
                                    println!("⚠️  [地图调试] ROI 裁剪存盘失败: {:?}", e);
                                } else {
                                    println!(
                                        "✅ [地图调试] 已保存地图名字区域截图到 {}，请核对是否刚好框住地图名字",
                                        map_roi_path
                                    );
                                }
                                dumped_map_debug = true;
                            }

                            if !map_templates.is_empty() {
                                match map_matcher::identify_map(mat, &map_cfg, &map_templates, 0.8)
                                {
                                    Ok(Some((name, score))) => {
                                        println!(
                                            "🗺️  [地图识别] 当前地图: {} | 置信度: {:.2}%",
                                            name,
                                            score * 100.0
                                        );
                                        live_status.update_map_name(name);
                                    }
                                    Ok(None) => {
                                        println!("⚠️  [地图识别] 未能匹配到任何已知地图名字模板");
                                    }
                                    Err(e) => println!("⚠️  [地图识别] 识别失败: {:?}", e),
                                }
                            }
                        }

                        // 💰 检测地面掉落物品:改用 OCR 识别文字(不再依赖颜色阈值+
                        // 模板匹配,因为物品种类太多没法逐个建模板库)。
                        // 复用同一帧 bgr_mat,不需要重新截图。
                        if let (Ok(mat), Some(recognizer)) = (&bgr_mat, &item_ocr_recognizer) {
                            match recognizer.detect_items(
                                mat,
                                &item_ocr_cfg,
                                &live_status.allowed_item_set,
                            ) {
                                Ok(matched) => {
                                    if !matched.is_empty() {
                                        println!("💰 [拾取] 发现 {} 个白名单物品", matched.len());
                                        for (item_name, ocr_text, conf) in &matched {
                                            println!(
                                                "   └── 💎 {} (OCR原文: {} | 置信度: {:.2}%)",
                                                item_name,
                                                ocr_text,
                                                conf * 100.0
                                            );
                                        }

                                        // ⚠️ OCR 目前只返回识别到的文字内容,没有解析具体
                                        // 屏幕坐标(拾取本身也不需要精确坐标,点固定的
                                        // "拾取"按钮就行),这里 screen_x/y 先占位成 0。
                                        let entities: Vec<game_status::EntityInfo> = matched
                                            .iter()
                                            .map(|(name, _, conf)| game_status::EntityInfo {
                                                name: name.clone(),
                                                screen_x: 0,
                                                screen_y: 0,
                                                confidence: *conf,
                                            })
                                            .collect();
                                        live_status.update_items(entities);

                                        if let Some(btn_pick_up) = coordinates_cache.get("pick_up")
                                        {
                                            if let (Some(x), Some(y)) =
                                                (btn_pick_up.screen_x, btn_pick_up.screen_y)
                                            {
                                                mouse_action::click_at(
                                                    &mut enigo,
                                                    x,
                                                    y,
                                                    "【自动拾取】发现白名单物品",
                                                );

                                                // 🎯 拾取优先级 > 自动攻击 > 自动寻路移动。
                                                // 点击拾取按钮后固定等待2秒,让角色有时间跑
                                                // 过去把物品捡起来,这段时间内不做怪物攻击/
                                                // 移动判断(不会打断已经在进行的自动战斗,
                                                // 只是这一轮循环不再额外发出攻击/移动指令),
                                                // 直接跳到下一轮循环重新截图判断。
                                                println!(
                                                    "⏳ [拾取] 等待角色跑过去拾取(约2秒)，本轮跳过打怪/寻路判断..."
                                                );
                                                sleep(Duration::from_secs(2));
                                                continue;
                                            }
                                        }
                                    } else {
                                        live_status.update_items(Vec::new());
                                    }
                                }
                                Err(e) => println!("⚠️  [物品OCR] 识别失败: {:?}", e),
                            }
                        }

                        if !templates.is_empty() {
                            match &bgr_mat {
                                Ok(mat) => match monster_matcher::identify_monsters(
                                    mat, &boxes, &templates, 0.75,
                                ) {
                                    Ok(identified) => {
                                        let allowed_monsters: Vec<_> = identified
                                            .into_iter()
                                            .filter(|(name, _, _)| {
                                                live_status.allowed_monster_set.contains(name)
                                            })
                                            .collect();

                                        if !allowed_monsters.is_empty() {
                                            println!(
                                                "🎯 [自动攻击] 发现 {} 个白名单怪物，准备攻击...",
                                                allowed_monsters.len()
                                            );
                                            for (name, b, score) in &allowed_monsters {
                                                println!(
                                                    "   └── 👾 {} | 置信度: {:.2}% | 位置: ({}, {})",
                                                    name,
                                                    score * 100.0,
                                                    b.x,
                                                    b.y
                                                );
                                            }

                                            // 📦 写入 game_status 缓存,而不是识别完就扔
                                            let entities: Vec<game_status::EntityInfo> =
                                                allowed_monsters
                                                    .iter()
                                                    .map(|(name, b, score)| {
                                                        game_status::EntityInfo {
                                                            name: name.clone(),
                                                            screen_x: b.x,
                                                            screen_y: b.y,
                                                            confidence: *score,
                                                        }
                                                    })
                                                    .collect();
                                            live_status.update_monsters(entities);

                                            should_attack = true;
                                        } else {
                                            println!(
                                                "⏳ [自动攻击] 本次识别到的怪物不在白名单内，继续轮询..."
                                            );
                                            live_status.update_monsters(Vec::new());
                                        }
                                    }
                                    Err(e) => println!("⚠️  [怪物识别] 识别失败: {:?}", e),
                                },
                                Err(e) => println!("⚠️  [调试] RGBA 转 Mat 失败: {:?}", e),
                            }
                        } else if !boxes.is_empty() {
                            println!(
                                "⚠️  [自动攻击] 模板库为空，检测到候选文字框，先尝试点击攻击按钮。"
                            );
                            should_attack = true;
                        } else {
                            println!("⏳ [自动攻击] 当前未检测到怪物候选文字框，继续轮询...");
                        }

                        if should_attack {
                            // 🎯 发现目标:清空卡住检测器状态,避免战斗结束后用旧的
                            // 基准坐标误判"没动"。
                            live_status.reset_movement_check();

                            if let Some(btn_attack) = coordinates_cache.get("attack") {
                                if let (Some(x), Some(y)) =
                                    (btn_attack.screen_x, btn_attack.screen_y)
                                {
                                    println!("⚔️  [自动攻击] 点击攻击按钮: {}", btn_attack.name);
                                    mouse_action::click_at(
                                        &mut enigo,
                                        x,
                                        y,
                                        "【自动攻击】大剑普通攻击",
                                    );
                                } else {
                                    println!("⚠️  [自动攻击] 攻击按钮坐标未缓存，无法点击。");
                                }
                            } else {
                                println!("⚠️  [自动攻击] 未能在资产缓存中找到攻击按钮。");
                            }
                        } else {
                            // 🚶 没发现目标,需要判断要不要移动。
                            //
                            // 完全根据角色真实坐标有没有变化来判断:
                            // - 坐标在变 -> 角色正在自动走路(可能是上一次点地图
                            //   目标点触发的自动寻路还没走完),什么都不做,继续等
                            // - 坐标没变 -> 已经走到目标点停下了,或者压根没在动
                            //   (刚进入这个状态/卡住了),该打开大地图选一个新
                            //   目标点,点击后靠引擎自动寻路把角色带过去
                            if live_status.has_position_stalled(
                                current_position,
                                STUCK_MOVE_EPSILON,
                                STUCK_CHECK_INTERVAL,
                            ) {
                                if map_nav_supported {
                                    println!("🗺️  [移动] 角色坐标未变化,打开大地图选取新目标点...");
                                    match map_nav::navigate_to_random_point(
                                        &window, &mut enigo, &nav_cfg,
                                    ) {
                                        Ok(true) => {
                                            println!(
                                                "✅ [移动] 已触发引擎自动寻路,交给引擎接管这段路"
                                            );
                                        }
                                        Ok(false) => {
                                            println!(
                                                "⚠️  [移动] 当前地图不支持该流程,本次会话后续不再尝试自动寻路移动"
                                            );
                                            map_nav_supported = false;
                                        }
                                        Err(e) => {
                                            println!("⚠️  [移动] 执行出错: {:?}", e);
                                        }
                                    }
                                } else {
                                    println!(
                                        "⚠️  [移动] 地图导航不可用,当前没有可用的移动方式,原地等待..."
                                    );
                                }
                            } else {
                                println!("🚶 [移动] 角色坐标正在变化或仍在观察窗口内,继续等待...");
                            }
                        }
                    }
                    Err(e) => println!("⚠️  [怪物检测] 失败: {:?}", e),
                }
            }
            None => println!("⚠️  [截图] 失败，跳过本次循环"),
        }

        sleep(Duration::from_millis(1200));
    }
}
