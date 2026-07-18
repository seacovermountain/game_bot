// src/quit_game_bot.rs

use device_query::{DeviceQuery, DeviceState, Keycode};
use std::thread;
use std::time::{Duration, Instant};

pub struct QuitWatchdog;

impl QuitWatchdog {
    /// 🚀 启动异步并行看门狗守护线程
    pub fn start_async_loop() {
        // 创建一个独立的新线程，专门用于监听键盘，不占用主业务线程
        thread::spawn(move || {
            let device_state = DeviceState::new();
            let mut esc_pressed_start: Option<Instant> = None;

            // 守护线程自己的死循环，独立于你的游戏逻辑
            loop {
                let keys: Vec<Keycode> = device_state.get_keys();

                if keys.contains(&Keycode::Escape) {
                    match esc_pressed_start {
                        Some(start_time) => {
                            let elapsed = start_time.elapsed();

                            if elapsed.as_millis() % 500 < 20 {
                                println!(
                                    "⏳ [独立守护线程] 检测到长按 ESC... 已按住: {:.1} 秒",
                                    elapsed.as_secs_f32()
                                );
                            }

                            if elapsed >= Duration::from_secs(2) {
                                println!(
                                    "\n🛑 [绝对安全退出] 收到强制退出指令！正在强杀所有并行线程..."
                                );
                                // 💡 核心：std::process::exit 会直接结束整个进程
                                // 哪怕你的游戏识别逻辑此刻卡死在死循环里，也会被瞬间秒杀退出！
                                std::process::exit(0);
                            }
                        }
                        None => {
                            println!("🎵 [独立守护线程] 检测到按下 ESC，启动 3 秒安全计时...");
                            esc_pressed_start = Some(Instant::now());
                        }
                    }
                } else {
                    if esc_pressed_start.is_some() {
                        println!("💤 [独立守护线程] 用户松开 ESC，计时重置。");
                        esc_pressed_start = None;
                    }
                }

                // 守护线程保持极高的检测频率（10毫秒一次）
                thread::sleep(Duration::from_millis(10));
            }
        });
    }
}
