use crate::match_icon;
use image::{ImageBuffer, Rgb, RgbImage};
use std::path::{Path, PathBuf};
use xcap::Window;

/// 🌍 算物理像素与窗口逻辑尺寸之间的缩放系数,方便把"截图里量出来的像素坐标"
/// 换算成"鼠标点击需要的绝对屏幕坐标"。
/// asset_loader.rs 和 map_nav.rs 原来各自维护了一份完全相同的实现,现在统一到这里。
pub fn compute_scale_factor(window: &Window, physical_w: u32, physical_h: u32) -> (f32, f32) {
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

pub fn capture_window(window: &Window) -> Option<(Vec<u8>, u32, u32)> {
    let xcap_image = window.capture_image().ok()?;
    let width = xcap_image.width();
    let height = xcap_image.height();

    // 将 xcap 内部的图片对象转换成标准的通用 Vec<u8> 原始像素数据
    let raw_pixels = xcap_image.into_raw();

    Some((raw_pixels, width, height))
}

pub fn save_debug_image<P: AsRef<Path>>(img: &RgbImage, path: P) -> bool {
    img.save(path).is_ok()
}

pub fn rgba_to_rgb(raw_rgba: &[u8], width: u32, height: u32) -> RgbImage {
    let mut rgb_pixels = Vec::with_capacity((width * height * 3) as usize);

    // 每 4 个字节代表一个 RGBA 像素，我们只要前 3 个 (RGB)
    for chunk in raw_rgba.chunks_exact(4) {
        rgb_pixels.push(chunk[0]); // R
        rgb_pixels.push(chunk[1]); // G
        rgb_pixels.push(chunk[2]); // B
    }

    ImageBuffer::<Rgb<u8>, Vec<u8>>::from_raw(width, height, rgb_pixels)
        .expect("无法构建标准 RgbImage")
}

pub fn get_secure_template_path(filename: &str) -> String {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("templates");
    path.push("icons");
    path.push(filename);

    // 它在 Mac 上会自动输出: /Users/xxx/project/templates/attack.png
    // 在 Win 上会自动输出: C:\xxx\project\templates\attack.png
    path.to_string_lossy().into_owned()
}

/// 🗂️ 怪物名字模板库目录:templates/monster_names/
/// 里面每张 png 的文件名(不含扩展名)就是对应的怪物名字,
/// 跟 config.toml 里 target_monsters 的写法保持完全一致。
pub fn get_monster_name_template_dir() -> String {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("templates");
    path.push("monster_names");
    path.to_string_lossy().into_owned()
}

/// 🗂️ 调试用:每次跑检测时,把候选框裁剪结果存到哪个目录，
/// 方便你从里面挑出真正的怪物名字去建模板库。
pub fn get_debug_crop_dir() -> String {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("DEBUG_MONSTER_CROPS");
    path.to_string_lossy().into_owned()
}

/// 🗂️ 坐标数字模板库目录:templates/digits/
/// 里面每张 png 的文件名(不含扩展名)就是对应的数字/符号,
/// 例如 0.png -> "0"，逗号请存成 comma.png(加载时自动转成 ",")。
pub fn get_digit_template_dir() -> String {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("templates");
    path.push("digits");
    path.to_string_lossy().into_owned()
}

/// 🗂️ 调试用:坐标区域整块裁剪存到哪个文件。
pub fn get_debug_position_roi_path() -> String {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("DEBUG_POSITION_ROI.png");
    path.to_string_lossy().into_owned()
}

/// 🗂️ 调试用:坐标区域里单个数字字符裁剪结果存到哪个目录。
pub fn get_debug_digit_crop_dir() -> String {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("DEBUG_DIGIT_CROPS");
    path.to_string_lossy().into_owned()
}

/// 🗂️ 调试用:地图名字 ROI 区域整块裁剪存到哪个文件,方便核对 ROI 范围。
pub fn get_debug_map_roi_path() -> String {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("DEBUG_MAP_ROI.png");
    path.to_string_lossy().into_owned()
}
