use std::thread::sleep;
use std::time::Duration;
use xcap::Window;

/// 🎯 跨平台锁定游戏窗口函数（已加固最小化过滤）
pub fn find_game_window(title: &str) -> Option<Window> {
    // 转换成小写，实现模糊匹配
    let target_title = title.to_lowercase();

    Window::all().ok()?.into_iter().find(|w| {
        let w_title = w.title().to_lowercase();

        // 💡 核心加固点：
        // 1. 标题必须包含关键字
        // 2. 窗口绝对不能处于最小化状态 (!w.is_minimized())
        w_title.contains(&target_title) && !w.is_minimized()
    })
}

pub fn require_game_window(title: &str) -> Window {
    println!("🔄 正在全系统检索包含关键字 [{}] 的活动游戏窗口...", title);

    for attempt in 1..=5 {
        if let Some(window) = find_game_window(title) {
            println!(
                "🎯 [窗口锁定成功] 标题: \"{}\" | 坐标: ({}, {}) | 分辨率: {}x{}",
                window.title(),
                window.x(),
                window.y(),
                window.width(),
                window.height()
            );
            return window; // 成功找到，直接把窗口作为战利品返回出去
        }

        println!(
            "⚠️  第 {} / 5 次尝试：未检测到游戏窗口，2秒后重试...",
            attempt
        );
        sleep(Duration::from_secs(2));
    }

    // 最终审判：5次都找不到，直接拔掉电源，终结整个进程
    println!("\n❌ [严重错误] 连续 5 次未检测到游戏窗口，程序失去运行基础，强制闪退！");
    std::process::exit(1);
}
