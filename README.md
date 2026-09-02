# Tennis Simulation — MuJoCo + Unitree G1

逼真的 ITF 标准网球场 + Unitree G1 人形机器人 + 发球机的 MuJoCo 仿真环境，
用于训练与部署 G1 的网球击球 policy（Serve / Forehand / Backhand / Volley / Smash）。

## 环境要求

- Linux + NVIDIA GPU（离屏渲染用 EGL，无需显示器）
- 系统即可，所有依赖装在项目内 `.venv`

## 安装

```bash
bash setup_env.sh
source .venv/bin/activate
```

`setup_env.sh` 会：创建 `.venv`、安装 `mujoco/numpy/imageio/imageio-ffmpeg`、
从 MuJoCo Menagerie 下载官方 Unitree G1 模型到 `assets/g1/`、生成球场、验证 EGL 渲染。
临时文件与缓存在 `~/tmp`。

## 快速开始

```bash
# 物理验证（28 项断言：弹跳/摩擦/阻力/Magnus/球机精度/防穿透）
python -m tennis_sim.physics_check

# 冒烟测试（站立喂球 + 脚本化正手回球过网）
python -m tennis_sim.smoke_test

# 录制演示视频（1080p50 mp4）
python scripts/record_video.py --out outputs/videos/demo.mp4 --camera rally
```

## 场景与坐标系

- 球网位于 `x=0`，底线 `x=±11.885`，单打边线 `y=±4.115`，双打边线 `y=±5.485`，发球线 `x=±6.4`
- G1 默认站在 `+x` 半区面向球网，发球机默认在 `−x` 半区
- 网高：中央 0.914 m，网柱 1.07 m（网柱位于双打边线外 0.914 m）

## 物理真实性（已验证）

| 项目 | 数值 | 依据 |
|---|---|---|
| 球质量/半径 | 57.7 g / 33.5 mm | ITF |
| 2.54 m 落地回弹 | 1.407 m（标准 1.35–1.47） | ITF |
| 恢复系数随速度 | e≈0.76（慢）→0.71（快） | 硬地实测规律 |
| 空气阻力 | Cd≈0.55，无风，F=½ρCdAv² | 与解析 RK4 误差 <2 cm |
| Magnus 力 | Cl=1/(2+v/(rω))，上旋下坠/下旋飘升 | 与解析 RK4 误差 <2 cm |
| 自旋衰减 | dω/dt=−0.04ω | 自由飞行验证 |
| 50 m/s 撞地穿透 | 6.5 mm | 数值稳定 |
| 网拦截 | 40–50 m/s 球全部被网挡下（无穿透） | — |

实现方式：刚性稳定接触（`solref=0.001 1.0`）+ 显式反弹层
（法向恢复系数 + 切向摩擦/上旋踢球模型，`tennis_sim/bounce.py`），
因为 MuJoCo 内置软接触无法同时满足 ITF 精度、高速稳定与低穿透。

## 发球机

```python
from tennis_sim.env import TennisEnv

env = TennisEnv()
env.reset(machine_pos=(-12.7, -3.0, 0.0), machine_yaw=0.0)

# 按落点/球速/转速发球（轨迹规划含空气阻力与过网净空校验）
plan = env.machine.serve(target_xy=(8.0, 1.5), mode="topspin")   # flat/topspin/slice/lob
# 或直接瞄准一个空间点
plan = env.machine.serve_direct(point=(0.0, -0.3, 0.55), speed=45.0)
```

模式：`flat` 24–33 m/s、`topspin` 16–27 m/s +2200~3200 rpm、
`slice` 19–29 m/s 负旋、`lob` 高弧线。落点精度 <0.15 m（20/20 实测）。

## Policy 部署接口

```python
from tennis_sim.env import TennisEnv

env = TennisEnv()
obs = env.reset()
env.set_policy(my_policy)          # fn(obs)->action(29,) 或实现 .reset(obs)/.act(obs) 的对象

obs, reward, done, info = env.step(action)
```

- 控制 50 Hz，物理 2000 Hz（`env.step` 内部走 40 个物理子步）
- **action**（29 维，位置目标，弧度）：12 腿 + 3 腰 + 7 左臂 + 7 右臂，
  顺序 = `env.act_joint_names`；无策略时可用 `obs["qpos"]` 冻结保持站立
- **obs["flat"]**（86 维）：qpos29 | qvel29 | 骨盆位姿7 | 骨盆线/角速度6 |
  球位置3 | 球速度3 | 球角速度3 | 拍头位置3 | 拍头速度3
- **info["events"]**：`bounce`（含落点与 in/out 判定）、`racket_hit`（含击球前后球速）、
  `net_hit` —— 用于自建 reward（如：击球后球落在对方单打区 = +1）

技能注册（含发球机配球与成功判定）见 `tennis_sim/skills.py`：
`forehand / backhand / volley / smash / serve`，
`skills.evaluate_skill(events, duration)` 返回回合成功率等指标。

## 录制视频

```bash
python scripts/record_video.py --camera rally --seconds 4.5 --out outputs/videos/demo.mp4
```

机位：`rally`（默认近景）/ `robot_front`（机器人正前方跟随机位）/ `broadcast`
（整体俯拍，全场含底线均在画面内）/ `side` / `behind_robot` / `robot_close` / `track_ball`。
也可用 `tennis_sim.render.EpisodeRecorder` 包住任何 `env.run_episode(..., on_step=rec.on_step)`。

脚本化正手按网球规则接发：喂球先在机器人半场落地一次，机器人不截击、
落地后再击球；`scripts/record_video.py` 会持续录制到回球完全静止后结束。

## 场景装饰与可视化（纯渲染层，零物理影响）

- 天空：court 项目 kloppenheim_06 HDRI 转立方体贴图（`assets/kloppenheim_06_puresky_4k.hdr`），
  主光方向对齐 court 项目太阳灯；地面纹理为 court 项目 painted concrete 4K 贴图
  （`assets/textures/`，颜色保持原 RGBA 不变）；球网为 court2 样式细网格
  （长度/高度与原一致，碰撞盒不变）；网球配色同 court 项目

- 程序化天空立方体贴图（渐变 + 云层 + 太阳光晕，`tennis_sim/textures.py` 自动生成
  `assets/sky/skybox.png`，失败时回退纯渐变）
- 场地装饰（全部 contype=0 非碰撞）：围网 + 挡风背板、裁判椅、球员座椅、
  捡球车、四角灯杆、外围大地面；场地/外围地面带程序化颗粒纹理
- 网球拍：真实 23 英寸青少年拍（总长 0.584 m），黑色拍框 + 白色拍柄 + 线床，
  腕部 +25° 固定偏移 mount（绕前臂/腕滚轴）；质量与碰撞参数保持不变
- 球轨迹彗星尾（`render.BallTrail`，仅注入渲染 scene）：去球（发球机→机器人）
  青色，回球（机器人击回）橙色，宽度/透明度渐隐；`Recorder(trail=False)` 可关闭

## 无 policy 的冒烟演示

`tennis_sim/smoke_test.py`：
1) 站立抗 6 次多角度喂球（物理断言全过）；
2) 脚本化正手（IK 轨迹 + 全场景试演自动校准）真实击球并回球过网落入对方场地。

## 项目结构

```
tennis_sim/
  constants.py        物理常量（标定后自动回写）
  aero.py             空气阻力 + Magnus + 自旋衰减（仿真与规划共用）
  bounce.py           反弹控制器：ITF 恢复系数/切向模型 + 球/网/拍事件检测
  build_court.py      程序化生成 ITF 球场 + 场景装饰
  textures.py         程序化贴图（天空立方体贴图/场地颗粒纹理）
  world.py            场景组装（球场+G1+球拍+发球机+球 → assets/scene.xml）
  ball_machine.py     发球机（轨迹规划/过网净空/直接瞄准/穿越点喂球）
  env.py              Gym 风格环境 + policy 接口 + 事件流
  skills.py           五技能注册与成功判定
  scripted_forehand.py 脚本化正手（IK + 试演自动校准，冒烟演示用）
  physics_check.py    物理标定与 28 项验证
  smoke_test.py       端到端冒烟测试
  render.py           EGL 离屏渲染 + mp4 录制
scripts/
  setup/download_g1/record_video
assets/               court_preview.xml scene.xml g1/ ball_seam.png
outputs/              videos/ qa/
```

## 注意

- 首次 `TennisEnv()` 会自动生成 `assets/scene.xml`（可用 `world.write_scene(...)` 自定义机器人/球机初始位）
- 训练 policy 后直接 `env.set_policy(your_policy)` 即可部署测试，无需改动环境代码
- 沙箱外有 4×3090 可用：渲染用 `EGL_DEVICE_ID` 选卡，训练可并行多个 env 进程
