// src/main.rs
//! 用法:
//!   cargo run                          跑完整挂机流程(找怪 -> 攻击/拾取/移动)
//!
//!   怪物识别拆成 3 步(参考 monster_matcher.rs 顶部注释里的建库流程):
//!   TEST_MODULE=monster_capture cargo run   第1步:只截取候选框存盘到
//!                                            DEBUG_MONSTER_CROPS/,不做模板匹配
//!                                            (然后你自己人工确认+改名+复制到
//!                                             templates/monster_names/)
//!   TEST_MODULE=monster_verify  cargo run   第3步:只做模板匹配验证,不存盘
//!
//!   TEST_MODULE=item     cargo run     只测物品识别
//!   TEST_MODULE=position cargo run     只测坐标识别
//!   TEST_MODULE=map      cargo run     只测地图名字识别
//!
//! 不管跑哪种模式,"前提"(1.找游戏窗口 2.截图定位按钮资产并缓存)
//! 都只在 `app::App::init()` 里做这一次,跟原来完全一样。

use game_bot::{app, bot_loop, quit_game_bot, test_modes};

fn main() {
    println!("🚀 游戏自动化辅助主程序已启动...");
    println!("⌨️  随时长按 ESC 键满 3 秒即可强制退出...\n");

    // 1. 启动并行后台键盘强杀模块
    quit_game_bot::QuitWatchdog::start_async_loop();

    // 2. 找窗口、定位图标资产、激活鼠标焦点、加载各类模板库/OCR引擎。
    // 详见 app::App::init() 内的分步注释。这一步是所有模式共同的前提,
    // 不管接下来跑完整流程还是单模块测试,都只做这一次。
    let mut app = app::App::init();

    // 3. 根据 TEST_MODULE 环境变量决定接下来跑什么:
    //    不设置 -> 完整挂机循环;设置了 -> 只跑对应的单模块识别测试循环。
    match std::env::var("TEST_MODULE").ok().as_deref() {
        Some("monster_capture") => test_modes::run_monster_capture(&mut app),
        Some("monster_verify") => test_modes::run_monster_verify(&mut app),
        Some("item") => test_modes::run_item_test(&mut app),
        Some("position") => test_modes::run_position_test(&mut app),
        Some("map") => test_modes::run_map_test(&mut app),
        Some(other) => {
            println!(
                "⚠️  未知的 TEST_MODULE 值: \"{}\"，可选: monster_capture / monster_verify / item / position / map。将按完整流程运行。",
                other
            );
            bot_loop::run(&mut app);
        }
        None => bot_loop::run(&mut app),
    }
}
