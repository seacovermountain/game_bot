// src/game_status.rs

use chrono::Local;
use serde::Deserialize;
use std::collections::HashSet;
use std::fs;
use std::time::{Duration, Instant};

#[derive(Debug, Deserialize, Clone)]
pub struct HuntingConfig {
    pub target_monsters: Vec<String>,
    pub hp_protect_line: u8,
}

#[derive(Debug, Deserialize, Clone)]
pub struct LootConfig {
    pub target_items: Vec<String>,
}

/// 🔊 日志开关:每一项对应 config.toml 里 [logging] 下的一个开关。
/// 关掉某一项只会让"正常流程信息"不再打印;真正的报错/存盘失败
/// (println 前缀是 ⚠️/❌ 的那类失败分支)不受这里控制，一直会打印，
/// 免得关掉某个类别之后连它出的错都看不见。
#[derive(Debug, Deserialize, Clone)]
pub struct LoggingConfig {
    #[serde(default = "default_true")]
    pub capture: bool,
    #[serde(default = "default_true")]
    pub position: bool,
    #[serde(default = "default_true")]
    pub map: bool,
    #[serde(default = "default_true")]
    pub monster: bool,
    #[serde(default = "default_true")]
    pub item: bool,
    // "move" 是 Rust 关键字，字段名换成 movement，toml 里的键名还是 move。
    #[serde(default = "default_true", rename = "move")]
    pub movement: bool,
    #[serde(default = "default_true")]
    pub status: bool,
}

fn default_true() -> bool {
    true
}

impl Default for LoggingConfig {
    fn default() -> Self {
        LoggingConfig {
            capture: true,
            position: true,
            map: true,
            monster: true,
            item: true,
            movement: true,
            status: true,
        }
    }
}

#[derive(Debug, Deserialize, Clone)]
pub struct AppConfig {
    pub hunting: HuntingConfig,
    pub loot: LootConfig,
    // config.toml 里没有 [logging] 这一段也没关系，全部退化成默认开(true)。
    #[serde(default)]
    pub logging: LoggingConfig,
}

/// 💾 统一的实体快照情报（怪物和物品共用坐标实体）
#[derive(Debug, Clone)]
pub struct EntityInfo {
    pub name: String,    // 实体中文名
    pub screen_x: i32,   // 绝对屏幕 X
    pub screen_y: i32,   // 绝对屏幕 Y
    pub confidence: f32, // 识别置信度(模板匹配分数,0.0~1.0)
}

/// 🎯 本轮"移动状态"判断结果,细分成五种含义不同的情况,方便调用方
/// 打印准确的日志,而不是笼统地都算作"没有卡住"。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MovementStatus {
    /// 这一轮没能读到坐标(数字模板库为空,或本轮识别失败)
    NoPosition,
    /// 第一次记录基准坐标,还没开始判断
    FirstObservation,
    /// 还没到 check_interval 这个检查点,先观察
    Cooling,
    /// 到了检查点,判定角色确实在移动
    Moving,
    /// 到了检查点,判定角色卡住了(坐标基本没变)
    Stalled,
}

#[derive(Debug, Clone, Default)]
pub struct GameStatusCache {
    // 1. 地图与位置监控
    pub map_name: String,
    pub player_x: i32,
    pub player_y: i32,

    // 2. 生命体征
    pub hp_percent: u8,

    // 3. 🎯 【分流缓存】实时通过配置白名单过滤后的存活怪与地上物品
    pub active_monsters: Vec<EntityInfo>,
    pub active_items: Vec<EntityInfo>,

    // 4. 🗃️ 运行时加载的哈希过滤器
    pub allowed_monster_set: HashSet<String>,
    pub allowed_item_set: HashSet<String>,

    // 4.5 🔊 从 config.toml [logging] 段加载的日志开关
    pub logging: LoggingConfig,

    // 5. 🎯 移动状态判断:角色坐标是否"长时间没有变化"(卡住/空闲),
    // 用于决定要不要打开大地图重新选择目标点。
    // 复用 update_position 已经缓存的实时坐标做判断,而不是在别的模块
    // (比如 map_nav.rs)里再单独维护一套平行状态 —— 跟 target.md 里
    // "先把实时信息统一缓存起来,再根据缓存信息做业务处理"的设计保持一致。
    last_movement_check_position: Option<(i32, i32)>,
    last_movement_check_at: Option<Instant>,
}

impl GameStatusCache {
    pub fn new() -> Self {
        let mut cache = Self::default();
        cache.load_all_config_whitelists();
        cache
    }

    /// 📂 从外部 config.toml 动态加载怪物和物品的白名单
    pub fn load_all_config_whitelists(&mut self) {
        if let Ok(content) = fs::read_to_string("config.toml") {
            if let Ok(config_data) = toml::from_str::<AppConfig>(&content) {
                // 加载怪物
                let mut monster_set = HashSet::new();
                for monster in config_data.hunting.target_monsters {
                    monster_set.insert(monster.trim().to_string());
                }
                self.allowed_monster_set = monster_set;

                // 加载物品
                let mut item_set = HashSet::new();
                for item in config_data.loot.target_items {
                    item_set.insert(item.trim().to_string());
                }
                println!(
                    "✅ [配置中心] 加载成功！当前怪物目标: {}个 | 物品捡取目标: {}个",
                    self.allowed_monster_set.len(),
                    item_set.len()
                );
                self.allowed_item_set = item_set;
                self.logging = config_data.logging;
                return;
            }
        }
        println!("⚠️  [配置中心] 未能正确解析 config.toml 文件！");
    }

    /// 📍 用 position_reader 识别出的坐标更新玩家实时位置。
    pub fn update_position(&mut self, x: i32, y: i32) {
        self.player_x = x;
        self.player_y = y;
    }

    /// 🗺️ 用 map_matcher 识别出的地图名字更新缓存。
    pub fn update_map_name(&mut self, name: String) {
        self.map_name = name;
    }
    /// 🎯 判断角色本轮的"移动状态"细分成五种含义不同的情况,而不是
    /// 笼统地返回一个 bool——不然调用方没法区分"真的在移动"和"还没到
    /// 检查时间点/没读到坐标"这几种完全不同的场景,日志也会因此写得
    /// 不准确。
    ///
    /// 每隔 check_interval 才会真正做一次"动没动"的判断;这段时间内
    /// 直接返回 `Cooling`,给角色留出时间真正走出位移,避免拿两次间隔
    /// 太短的坐标比较导致误判。判断完(不管结果是 Moving 还是 Stalled)
    /// 都会重新记录一次基准坐标和时间,自然形成两次判断之间的冷却间隔,
    /// 不需要再额外单独维护一个冷却计时器。
    ///
    /// `current_position` 传入这一轮 position_reader 实际读到的坐标
    /// (读取失败传 None),而不是直接读 self.player_x/player_y —— 因为
    /// 读取失败的轮次里这两个字段还停留在上一次成功读取的旧值,直接用
    /// 会掩盖"这一轮根本没读到坐标"这个事实。
    pub fn check_movement_status(
        &mut self,
        current_position: Option<(i32, i32)>,
        move_epsilon: f64,
        check_interval: Duration,
    ) -> MovementStatus {
        let current = match current_position {
            Some(p) => p,
            None => return MovementStatus::NoPosition, // 坐标没读到,没法判断
        };

        match (
            self.last_movement_check_position,
            self.last_movement_check_at,
        ) {
            (None, _) | (_, None) => {
                // 第一次记录基准坐标,先观察一轮,不立即判断
                self.last_movement_check_position = Some(current);
                self.last_movement_check_at = Some(Instant::now());
                MovementStatus::FirstObservation
            }
            (Some(prev), Some(t)) => {
                if t.elapsed() < check_interval {
                    return MovementStatus::Cooling;
                }

                let dx = (current.0 - prev.0) as f64;
                let dy = (current.1 - prev.1) as f64;
                let dist = (dx * dx + dy * dy).sqrt();

                self.last_movement_check_position = Some(current);
                self.last_movement_check_at = Some(Instant::now());

                if dist < move_epsilon {
                    MovementStatus::Stalled
                } else {
                    MovementStatus::Moving
                }
            }
        }
    }

    /// 发现怪物准备攻击时调用,清空卡住判断状态,避免战斗结束后用旧的
    /// 基准坐标误判"没动"。
    pub fn reset_movement_check(&mut self) {
        self.last_movement_check_position = None;
        self.last_movement_check_at = None;
    }

    /// 👾 用本轮识别出的白名单怪物列表,整体替换掉缓存里的旧数据。
    /// 怪物识别本身就是每轮对画面全量重新扫描一遍,不存在"增量更新"
    /// 这一说,所以这里是直接整体替换,不是合并。
    pub fn update_monsters(&mut self, monsters: Vec<EntityInfo>) {
        self.active_monsters = monsters;
    }

    /// 💰 同上,地上掉落物列表整体替换。
    pub fn update_items(&mut self, items: Vec<EntityInfo>) {
        self.active_items = items;
    }

    /// 📥 漂亮打印，把怪和物品清清楚楚分开展示
    pub fn print_debug_status(&self) {
        let now = Local::now().format("%H:%M:%S");
        println!("====== 💾 [{}] 实时动态数据状态机快照 ======", now);
        println!(
            "🗺️  当前地图: {} | 坐标: ({}, {}) | ❤️ 血量: {}%",
            self.map_name, self.player_x, self.player_y, self.hp_percent
        );

        println!("⚔️  【合法战区怪】数量: {} 只", self.active_monsters.len());
        for (i, monster) in self.active_monsters.iter().enumerate() {
            println!(
                "   └── 👾 [怪 {:02}] 名称: {}({:.2}%) -> 物理点击坐标: ({}, {})",
                i + 1,
                monster.name,
                monster.confidence * 100.0,
                monster.screen_x,
                monster.screen_y
            );
        }

        println!("💰 【地上掉落物】数量: {} 个", self.active_items.len());
        for (i, item) in self.active_items.iter().enumerate() {
            println!(
                "   └── 💎 [物 {:02}] 名称: {}({:.2}%) -> 物理点击坐标: ({}, {})",
                i + 1,
                item.name,
                item.confidence * 100.0,
                item.screen_x,
                item.screen_y
            );
        }
        println!("======================================\n");
    }
}
