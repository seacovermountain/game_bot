// src/main.rs
mod app;
mod asset_loader;
mod bot_loop;
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

    // 2. 找窗口、定位图标资产、激活鼠标焦点、加载各类模板库/OCR引擎。
    // 详见 app::App::init() 内的分步注释。
    let mut app = app::App::init();

    // 3. 循环检测怪物:发现白名单怪物就攻击，没发现就移动寻路。
    bot_loop::run(&mut app);
}
