// src/app.rs
//! 应用状态与启动初始化。
//!
//! 把原来 main() 里"只跑一次"的准备工作(找窗口 -> 定位图标资产 ->
//! 激活鼠标焦点 -> 加载各类模板库/OCR引擎)全部收敛到 `App::init()`
//! 里,main() 只需要 `let mut app = App::init();` 然后把 app 交给
//! bot_loop::run() 就行,循环体也不用再顶着一堆局部变量到处传参,
//! 直接 `app.xxx` 取用/更新即可。

use enigo::{Enigo, Settings};
use std::collections::HashMap;
use std::thread::sleep;
use std::time::Duration;
use xcap::Window;

use crate::asset_loader::{self, IconButton};
use crate::game_status::GameStatusCache;
use crate::item_ocr::{ItemOcrConfig, ItemOcrRecognizer};
use crate::map_matcher::{self, MapReaderConfig, MapTemplate};
use crate::map_nav::NavConfig;
use crate::monster_detector::DetectorConfig;
use crate::monster_matcher::{self, MonsterTemplate};
use crate::mouse_action;
use crate::position_reader::{self, DigitTemplate, PositionReaderConfig};
use crate::util;

/// 🚶 判断"角色坐标是否停滞"用的两个阈值,原来是 main() 里的局部
/// const,现在跟着 App 一起放这里,bot_loop 里直接引用。
pub const STUCK_CHECK_INTERVAL: Duration = Duration::from_secs(4);
pub const STUCK_MOVE_EPSILON: f64 = 5.0;
/// 🗺️ 大地图导航失败后的冷却时长,冷却结束会自动重试,而不是永久禁用。
pub const MAP_NAV_RETRY_COOLDOWN: Duration = Duration::from_secs(10);

/// 持有整场挂机会话需要跨循环轮次共享的一切状态:
/// 窗口句柄、鼠标控制器、各类模板库/识别配置、以及若干"只做一次"的
/// 调试导出/兼容性开关。
pub struct App {
    pub window: Window,
    pub enigo: Enigo,
    pub coordinates_cache: HashMap<String, IconButton>,
    pub live_status: GameStatusCache,

    pub cfg: DetectorConfig,
    pub pos_cfg: PositionReaderConfig,
    pub map_cfg: MapReaderConfig,
    pub nav_cfg: NavConfig,
    pub item_ocr_cfg: ItemOcrConfig,

    pub templates: Vec<MonsterTemplate>,
    pub digit_templates: Vec<DigitTemplate>,
    pub map_templates: Vec<MapTemplate>,
    pub item_ocr_recognizer: Option<ItemOcrRecognizer>,

    /// 🐛 地图名字 ROI 位置目前是目测估算,还没有 DEBUG 图核实过,第一轮先存一张调试图
    pub dumped_map_debug: bool,
    /// 数字模板库还没建好的话,第一次跑先把坐标区域裁出来,方便你校准 ROI + 建模板
    pub dumped_position_debug: bool,
    /// 🗺️ 大地图导航上一次失败后的冷却截止时间。为 None 表示当前没有
    /// 冷却中,可以随时尝试;为 Some(t) 表示要等到 t 之后才重新尝试,
    /// 避免一次失败(比如动画卡顿/没找到可行走点)就整场会话永久放弃。
    pub map_nav_retry_after: Option<std::time::Instant>,
    /// 🐛 支持通过环境变量持续导出怪物候选框调试图,方便建怪物名字模板库。
    /// 用法: DEBUG_MONSTER=1 cargo run
    pub debug_monster: bool,

    /// 🔘 大地图"关闭按钮"上一次成功识别到的绝对屏幕坐标缓存。按钮在
    /// 屏幕上的物理位置是固定的,一旦识别成功过一次就把坐标存起来,
    /// 以后哪怕某一轮模板匹配偶然失误(比如面板动画没播完/截图花屏),
    /// 也能直接用缓存坐标兜底点击,不至于让整场会话因为一次识别失误
    /// 就判定"这张地图不支持"。
    pub close_map_button_cache: Option<(i32, i32)>,
}

impl App {
    /// 一次性完成整场会话的启动准备工作。任何"基础资产定位失败"这种
    /// 致命错误会直接打印原因并 `std::process::exit(1)`,跟原来 main()
    /// 里的行为完全一致。
    pub fn init() -> Self {
        // 1. 寻找游戏窗口
        let game_title = "24luling";
        let window = crate::find_window::require_game_window(game_title);

        // 2. 📸 静态资产定位
        // ⚠️ 必须在做任何鼠标点击之前完成:这一步只依赖纯净的截图,
        // 不需要也不能碰鼠标。之前把"窗口焦点点击"挪到这一步前面,
        // 导致点击本身改变了游戏画面(角色转向/移动/UI变化等副作用),
        // 紧接着截的图就跟模板对不上了,相似度大幅下降。
        let coordinates_cache = match asset_loader::load_and_cache_assets(&window) {
            Some(cache) => cache,
            None => {
                println!(
                    "❌ [严重错误] 基础图标资产未能全部定位，无法进行后续挂机测试，程序退出！"
                );
                std::process::exit(1);
            }
        };

        // 3. 🖱️ 资产匹配完成后,再初始化鼠标并激活游戏窗口的输入焦点。
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

        // 4. 初始化识别环境
        let live_status = GameStatusCache::new();
        let cfg = DetectorConfig::default();
        let pos_cfg = PositionReaderConfig::default();

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

        let map_cfg = MapReaderConfig::default();
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

        let item_ocr_cfg = ItemOcrConfig::default();
        // 🎯 OCR 引擎只在启动时加载一次(det/rec 模型 + 字典),不要每轮循环
        // 都重新加载,模型文件需要你提前下载放到 CARGO_MANIFEST_DIR/models/
        let item_ocr_dir = format!("{}/models", env!("CARGO_MANIFEST_DIR"));
        let item_ocr_recognizer = match ItemOcrRecognizer::new(
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

        // 🗺️ 大地图导航(复用引擎自带寻路能力)相关状态。
        let nav_cfg = NavConfig::default();

        // 🐛 支持通过环境变量持续导出怪物候选框调试图,方便建怪物名字模板库。
        // 跟坐标/地图不同,建怪物模板往往需要站在目标怪物附近多跑几轮循环、
        // 让画面里出现你要的怪物,所以这里做成"开关打开就每轮都导出",而不是
        // 只导出一次,你可以跑一会儿再从 DEBUG_MONSTER_CROPS/ 目录里挑图。
        let debug_monster = std::env::var("DEBUG_MONSTER").is_ok();

        // 数字模板库还没建好的话才需要导出一次调试图,已经有模板库时
        // 直接标记成"已导出",跟原来 main() 里
        // `!digit_templates.is_empty()` 的初始值保持一致。
        let dumped_position_debug = !digit_templates.is_empty();

        App {
            window,
            enigo,
            coordinates_cache,
            live_status,
            cfg,
            pos_cfg,
            map_cfg,
            nav_cfg,
            item_ocr_cfg,
            templates,
            digit_templates,
            map_templates,
            item_ocr_recognizer,
            dumped_map_debug: false,
            dumped_position_debug,
            map_nav_retry_after: None,
            debug_monster,
            close_map_button_cache: None,
        }
    }
}
