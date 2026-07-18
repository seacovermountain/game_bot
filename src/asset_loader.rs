// src/asset_loader.rs

use crate::match_icon;
use crate::util;
use std::collections::HashMap;
use xcap::Window; // 💥 引入你的工具模块

#[derive(Debug, Clone)]
pub struct IconButton {
    pub name: String,
    pub template_path: String,
    pub min_confidence: f32,
    pub screen_x: Option<i32>,
    pub screen_y: Option<i32>,
}

/// 🛠️ 初始化并构建你需要匹配的图标资产池
fn create_button_manifest() -> HashMap<String, IconButton> {
    let mut manifest = HashMap::new();

    // 1. 录入：普通攻击按钮（利用你的 util 工具类自动生成安全的绝对路径）
    manifest.insert(
        "attack".to_string(),
        IconButton {
            name: "大剑普通攻击".to_string(),
            template_path: util::get_secure_template_path("attack.png"),
            min_confidence: 0.55,
            screen_x: None,
            screen_y: None,
        },
    );

    // 2. 录入：自动拾取按钮
    manifest.insert(
        "pick_up".to_string(),
        IconButton {
            name: "自动拾取".to_string(),
            template_path: util::get_secure_template_path("pick_up.png"),
            min_confidence: 0.55,
            screen_x: None,
            screen_y: None,
        },
    );

    manifest
}

/// 🎯 静态资产匹配与漂亮的账单打印
pub fn load_and_cache_assets(window: &Window) -> Option<HashMap<String, IconButton>> {
    println!("\n🔄 [资产模块] 开始对游戏运行画面进行 OpenCV 静态资产定位...");

    let mut button_manager = create_button_manifest();
    let (raw_rgba, width, height) = util::capture_window(window)?;
    let (scale_x, scale_y) = compute_scale_factor(window, width, height);
    println!(
        "   🌍 [跨平台缩放] 物理分辨率: {}x{} | 窗口逻辑尺寸: {}x{} | 缩放系数: ({:.2}, {:.2})",
        width,
        height,
        window.width(),
        window.height(),
        scale_x,
        scale_y
    );
    let mut all_success = true;

    for (_id, btn) in button_manager.iter_mut() {
        println!("   👉 OpenCV 正在检索资产: {} ...", btn.name);
        println!("      路径: [{}]", btn.template_path);

        // 💡 这里我们把 conf 提取出来打印，完美消除变量未使用的编译告警
        if let Some(((orig_x, orig_y), conf)) = match_icon::find_icon_opencv(
            &raw_rgba,
            width,
            height,
            &btn.template_path,
            btn.min_confidence,
            "test1",
        ) {
            let scr_x = window.x() + (orig_x as f32 / scale_x) as i32;
            let scr_y = window.y() + (orig_y as f32 / scale_y) as i32;

            btn.screen_x = Some(scr_x);
            btn.screen_y = Some(scr_y);
            println!("      🎯 [锁定成功] 置信度: {:.2}%", conf * 100.0);
        } else {
            println!("      ❌ [锁定失败] 画面中未找到该资产！");
            all_success = false;
        }
    }

    if all_success {
        println!("\n==========================================================================");
        println!("👑 [🎉 资产全量缓存成功] 基础坐标已锁死在内存中！全量资产账单如下：");
        println!("--------------------------------------------------------------------------");
        for (key, btn) in &button_manager {
            println!(
                " 🏷️  资产KEY: [{:<8}] | 名称: {:<6} | 绝对屏幕点击坐标: (X: {:<5}, Y: {:<5})",
                key,
                btn.name,
                btn.screen_x.unwrap(),
                btn.screen_y.unwrap()
            );
        }
        println!("==========================================================================\n");

        Some(button_manager)
    } else {
        println!("⚠️  [警告] 部分初始化资产丢失，程序无法安全挂机。\n");
        None
    }
}

fn compute_scale_factor(window: &Window, physical_w: u32, physical_h: u32) -> (f32, f32) {
    let logical_w = window.width().max(1) as f32;
    let logical_h = window.height().max(1) as f32;

    let scale_x = physical_w as f32 / logical_w;
    let scale_y = physical_h as f32 / logical_h;

    // 兜底:万一算出诡异值(比如窗口最小化瞬间宽高为 0 导致的极端比例),
    // 退回旧的按 OS 猜测的静态值,保证程序不会直接崩掉。
    if scale_x.is_finite() && scale_x > 0.1 && scale_y.is_finite() && scale_y > 0.1 {
        (scale_x, scale_y)
    } else {
        let fallback = match_icon::get_screen_scale_factor();
        println!(
            "   ⚠️ [缩放计算异常] 回退到静态缩放系数兜底: {:.2}",
            fallback
        );
        (fallback, fallback)
    }
}
