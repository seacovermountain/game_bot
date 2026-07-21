// src/bin/test_monster_recognition.rs
//! 🎯 怪物识别率测试工具(独立小工具,不依赖游戏窗口/不需要开着游戏跑)
//!
//! 用法:
//!   cargo run --bin test_monster_recognition
//!   cargo run --bin test_monster_recognition -- test_data/monster_names 0.75
//!
//! 📋 测试数据怎么准备(跟你现有的两个目录接起来):
//! 1. 正常挂机时用 `DEBUG_MONSTER=1 cargo run`,候选框裁剪图会持续存到
//!    `DEBUG_MONSTER_CROPS/`(文件名是 box_序号_x_y.png,还不知道具体是哪个怪)。
//! 2. 肉眼过一遍,把"确定是哪个怪物名字"的裁剪图复制一份出来,按怪物名字
//!    分文件夹存放,组成一份"已标注测试集"(跟建模板库那份 templates/monster_names/
//!    是两回事——那份是"每个怪物挑 1~2 张最干净的当模板",这份是"尽量多攒几张
//!    不同状态/不同光照的裁剪图当测试样本",两者不需要是同一批图):
//!
//!     test_data/monster_names/
//!       ├── 地火兽骑将/
//!       │     ├── box_03_120_400.png
//!       │     └── box_07_115_820.png
//!       ├── 炎魔/
//!       │     └── box_01_88_300.png
//!       └── ...
//!
//! 3. 跑这个工具,会拿 `templates/monster_names/` 里已经建好的模板库,
//!    对测试集里每一张裁剪图做一次模板匹配,汇总出:
//!    - 总体识别率(总数/命中数/命中率)
//!    - 按每个怪物名字分别统计的命中率,方便定位哪个模板不行
//!    - 具体识别错误/未命中的文件列表(文件路径 + 期望名字 + 实际匹配结果 + 匹配分数)
//!
//! ⚠️ 注意:这里用的是 `monster_matcher::identify_crop`,直接对"已经裁好的
//! 单张候选框图片"做模板匹配,不套用整帧检测(`identify_monsters`)里那套
//! 按整帧宽度换算缩放系数的逻辑——因为这里的输入本来就是单独裁出来的小图,
//! 没有"整帧宽度"这个概念,跟建模板库时用的是同一套物理分辨率。

use game_bot::monster_matcher::{self, MonsterTemplate};
use game_bot::util;
use opencv::{
    imgcodecs::{IMREAD_COLOR, imread},
    prelude::*,
};
use std::collections::BTreeMap;
use std::path::Path;

/// 单条测试样本的判定结果
enum Verdict {
    /// 匹配上了正确的怪物名字,且分数达到阈值
    Correct,
    /// 匹配上了某个模板,但分数没到阈值(生产环境里会被当成"没识别到")
    BelowThreshold { predicted: String, score: f32 },
    /// 匹配到了别的怪物名字(且分数达到阈值)——真正的误判
    Misidentified { predicted: String, score: f32 },
    /// 模板库里没有任何一个模板尺寸小于等于这张图,压根没法比较
    NoComparableTemplate,
}

/// 单个怪物名字下的统计
#[derive(Default)]
struct MonsterStat {
    total: usize,
    correct: usize,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let test_dir = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| {
            let mut p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
            p.push("test_data");
            p.push("monster_names");
            p.to_string_lossy().into_owned()
        });

    // 跟 bot_loop.rs 里 identify_monsters 用的阈值(0.75)保持一致,
    // 这样测出来的识别率才是"跟挂机时实际表现一致"的数字。
    let min_confidence: f32 = args
        .get(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.75);

    println!("🧪 [怪物识别率测试]");
    println!("   测试集目录: {}", test_dir);
    println!("   置信度阈值: {:.2}\n", min_confidence);

    if !Path::new(&test_dir).exists() {
        println!("❌ 测试集目录不存在: {}", test_dir);
        println!("   请先按上面文件头注释的说明准备测试集(按怪物名字分文件夹存放裁剪图)。");
        std::process::exit(1);
    }

    let template_dir = util::get_monster_name_template_dir();
    let templates = match monster_matcher::load_monster_templates(&template_dir) {
        Ok(t) => t,
        Err(e) => {
            println!("❌ 加载模板库失败: {:?}", e);
            std::process::exit(1);
        }
    };
    if templates.is_empty() {
        println!("❌ 模板库是空的({}),先建好 templates/monster_names/ 再测。", template_dir);
        std::process::exit(1);
    }
    println!("📎 已加载 {} 个怪物模板\n", templates.len());

    let mut per_monster: BTreeMap<String, MonsterStat> = BTreeMap::new();
    let mut failures: Vec<String> = Vec::new();
    let mut total = 0usize;
    let mut total_correct = 0usize;

    let monster_dirs = match std::fs::read_dir(&test_dir) {
        Ok(d) => d,
        Err(e) => {
            println!("❌ 读取测试集目录失败: {:?}", e);
            std::process::exit(1);
        }
    };

    for entry in monster_dirs.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue; // 忽略不是按怪物名字分类的散落文件
        }
        let expected_name = match path.file_name().and_then(|n| n.to_str()) {
            Some(n) => n.to_string(),
            None => continue,
        };

        let stat = per_monster.entry(expected_name.clone()).or_default();

        let files = match std::fs::read_dir(&path) {
            Ok(f) => f,
            Err(_) => continue,
        };

        for file_entry in files.flatten() {
            let file_path = file_entry.path();
            if file_path.extension().and_then(|e| e.to_str()) != Some("png") {
                continue;
            }

            let crop = match imread(&file_path.to_string_lossy(), IMREAD_COLOR) {
                Ok(m) if !m.empty() => m,
                _ => {
                    println!("   ⚠️ 无法读取图片,跳过: {}", file_path.display());
                    continue;
                }
            };

            total += 1;
            stat.total += 1;

            let verdict = judge(&crop, &expected_name, &templates, min_confidence);
            match verdict {
                Verdict::Correct => {
                    total_correct += 1;
                    stat.correct += 1;
                }
                Verdict::BelowThreshold { predicted, score } => {
                    failures.push(format!(
                        "   ⚠️ [分数不够] {} | 期望: {} | 最接近: {} (分数 {:.3} < 阈值 {:.2})",
                        file_path.display(),
                        expected_name,
                        predicted,
                        score,
                        min_confidence
                    ));
                }
                Verdict::Misidentified { predicted, score } => {
                    failures.push(format!(
                        "   ❌ [认错] {} | 期望: {} | 实际匹配到: {} (分数 {:.3})",
                        file_path.display(),
                        expected_name,
                        predicted,
                        score
                    ));
                }
                Verdict::NoComparableTemplate => {
                    failures.push(format!(
                        "   ⚠️ [无可比较模板] {} | 期望: {} | 模板库里没有尺寸能匹配的模板(裁剪图可能比模板还小)",
                        file_path.display(),
                        expected_name
                    ));
                }
            }
        }
    }

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("📊 按怪物名字分别统计:\n");
    for (name, stat) in &per_monster {
        if stat.total == 0 {
            continue;
        }
        let rate = stat.correct as f64 / stat.total as f64 * 100.0;
        let mark = if stat.correct == stat.total { "✅" } else { "⚠️" };
        println!(
            "   {mark} {:<12} {}/{}  ({:.1}%)",
            name, stat.correct, stat.total, rate
        );
    }

    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    if !failures.is_empty() {
        println!("🔎 失败明细({} 条):\n", failures.len());
        for f in &failures {
            println!("{}", f);
        }
        println!();
    }

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    if total == 0 {
        println!("⚠️ 测试集里一张图都没读到,检查目录结构是否正确。");
    } else {
        let overall_rate = total_correct as f64 / total as f64 * 100.0;
        println!(
            "🏁 总体识别率: {}/{}  ({:.1}%)",
            total_correct, total, overall_rate
        );
    }
}

/// 对单张裁剪图做一次识别判定
fn judge(
    crop: &Mat,
    expected_name: &str,
    templates: &[MonsterTemplate],
    min_confidence: f32,
) -> Verdict {
    match monster_matcher::identify_crop(crop, templates) {
        Ok(Some((predicted, score))) => {
            if score < min_confidence {
                Verdict::BelowThreshold { predicted, score }
            } else if predicted == expected_name {
                Verdict::Correct
            } else {
                Verdict::Misidentified { predicted, score }
            }
        }
        Ok(None) => Verdict::NoComparableTemplate,
        Err(e) => {
            println!("   ⚠️ 模板匹配出错: {:?}", e);
            Verdict::NoComparableTemplate
        }
    }
}
