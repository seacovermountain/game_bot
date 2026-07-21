名称  手机模拟器游戏脚本
需求  实现自动打怪，拾取，跑到指定地图，主要是windows平台和mac平台
业务拆解
1寻找游戏窗口
2截取游戏画面，获取指定按钮的位置，并缓存起来方便后续使用，只需要一次
3 实施获取关键信息 人物血量，当前坐标，地图名字，当前怪物信息，并缓存起来
4 根据玩家实际需求进行作业 ，比如 跑图。打怪和拾取等挂机复杂业务
模块部分
asset_oader.rs  缓存按钮位置信息
find_windows.rs 寻找游戏窗口
game_status.rs 加载配置文件，缓存游戏实时获取到的信息，比如 人物血量，坐标，怪物，物品等信息
match_icon.rs 按钮匹配部分
monster_detector.rs 怪物识别部分
monster_matcher.rs 怪物匹配部分
mouse_action.rs 鼠标操作部分
movement.rs 人物移动部分
quit_gmae_bot.rs  脚本退出部分
util.rs 工具类
