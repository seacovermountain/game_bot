// src/mouse_action.rs

use std::{thread, time::Duration};
// 💥 引入 Keyboard 和 Key 用于模拟键盘大招
use enigo::{Button, Coordinate::Abs, Direction, Enigo, Mouse};

/// 🎯 现有的鼠标点击函数（保持不变，供拾取按钮使用）
pub fn click_at(enigo: &mut Enigo, x: i32, y: i32, action: &str) {
    let _ = enigo.move_mouse(x, y, Abs);
    thread::sleep(Duration::from_millis(80));
    let _ = enigo.button(Button::Left, Direction::Press);
    thread::sleep(Duration::from_millis(100));
    let _ = enigo.button(Button::Left, Direction::Release);
    println!("鼠标动作: {} -> ({}, {})", action, x, y);
}
