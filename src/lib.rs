// src/lib.rs
//! 把原本只挂在 main.rs 下的模块对外暴露成一个库 crate,
//! 这样 src/bin/ 下的独立小工具(比如怪物模板裁剪工具、识别率测试工具)
//! 也能直接 `use game_bot::monster_detector;` 复用同一套检测/匹配逻辑,
//! 不用复制代码,也不需要启动完整的 app::App::init() / bot_loop::run()。
//!
//! 主程序 main.rs 现在也通过这个 lib 来引用各模块,行为和之前完全一致,
//! 只是模块的"物理位置"从 main.rs 的内联 `mod` 声明改成了这里。

pub mod app;
pub mod asset_loader;
pub mod bot_loop;
pub mod find_window;
pub mod game_status;
pub mod map_matcher;
pub mod map_nav;
pub mod match_icon;
pub mod monster_detector;
pub mod monster_matcher;
pub mod mouse_action;
pub mod position_reader;
pub mod quit_game_bot;
pub mod test_modes;
pub mod text_ocr;
pub mod util;
