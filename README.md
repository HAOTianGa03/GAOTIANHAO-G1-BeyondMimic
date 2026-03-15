# BeyondMimic — G1 Whole-Body Motion Tracking

<p align="center">
  <img src="docs/assets/banner.png" alt="BeyondMimic Banner" width="800">
</p>

> **BeyondMimic** 是一个基于 [Isaac Lab](https://github.com/isaac-sim/IsaacLab) 的全身运动追踪（Whole-Body Motion Tracking）框架，  
> 使用 **DeepMimic** 风格的奖励函数 + **RSL-RL PPO** 算法，让 Unitree G1 人形机器人在仿真中复现真实人体运动。
>
> - 论文：[arXiv:2508.08241](https://arxiv.org/abs/2508.08241)
> - 项目主页：[beyondmimic.github.io](https://beyondmimic.github.io)
> - 原始仓库：[HybridRobotics/whole_body_tracking](https://github.com/HybridRobotics/whole_body_tracking)

---

## 📌 本仓库与原始 BeyondMimic 的区别

本仓库在 [原始 BeyondMimic 框架](https://github.com/HybridRobotics/whole_body_tracking) 的基础上进行了以下扩展，以便于开箱即用：

| 新增内容 | 说明 |
|---------|------|
| **Dockerfile** | 提供了基于 `nvcr.io/nvidia/isaac-sim:5.0.0` 的完整 Docker 镜像构建文件，一键安装 Isaac Lab + pytorch3d + 全部依赖 |
| **Docker 文档** | 详细的 [Docker 环境搭建指南](docs/docker_setup.md)，涵盖构建、启动、验证全流程 |
| **ASAP 数据适配** | 新增 `asap_to_npz.py` 和 `batch_asap_to_npz.sh`，支持将 [ASAP](https://github.com/) 重定向数据（joblib pkl）转换为训练所需的 npz 格式 |
| **本地训练支持** | train.py 新增 `--motion_file` 参数，支持直接使用本地 `.npz` 文件训练，**无需 WandB 账号** |
| **本地推理支持** | play.py 新增 `--checkpoint_path` 参数，支持从本地 `.pt` 文件加载模型，无需 WandB |
| **ONNX 导出修复** | play.py 新增 `--no_export` 参数，解决 headless + camera 模式下 `torch.onnx.export` 与渲染线程死锁的问题 |
| **完整录制支持** | play.py 新增 `--no_termination` 参数，禁用 failure termination 以录制完整运动视频 |
| **离线可视化** | 新增 `visualize_npz.py`，支持纯 Matplotlib 骨架动画输出（无需 GPU 渲染） |

> 原始 BeyondMimic 框架强依赖 WandB Registry 管理数据和模型。本仓库的所有修改均保持了对 WandB 的可选兼容——如果你有 WandB 账号，原有的 `--registry_name` / `--wandb_path` 工作流仍然可用。

---

## 🚀 环境配置（必读）

**请先按照以下两个文档配置运行环境**：

1. **[Dockerfile](Dockerfile)**：基于 Isaac Sim 5.0.0 官方镜像，自动安装 Isaac Lab v2.2.0、pytorch3d、smplx 及全部依赖
2. **[Docker 环境搭建指南](docs/docker_setup.md)**：Step-by-step 的构建、启动、验证流程

```bash
# 快速开始（2 条命令）
docker image build --network host . -t mimic-model:local
docker run --gpus all -dit --restart=always --privileged \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $(pwd):/workspace/mimic_model \
    --network host -e DISPLAY=$DISPLAY \
    --name mimic-model -w /workspace/mimic_model \
    mimic-model:local /bin/bash
```

> 如果你已有 Isaac Sim 5.0.0 + Isaac Lab 环境，也可以直接安装 Extension：
> ```bash
> /isaac-sim/python.sh source/whole_body_tracking/setup.py develop
> ```

---

## 📁 项目结构

```
mimic_model/
├── Dockerfile                  # Docker 镜像构建文件
├── README.md                   # 本文件
├── .gitignore
├── docs/
│   └── docker_setup.md         # Docker 环境搭建详细指南
├── source/
│   └── whole_body_tracking/    # Isaac Lab Extension（核心代码）
│       ├── setup.py            # Python 包安装脚本
│       ├── config/
│       │   └── extension.toml  # Isaac Lab 扩展元数据
│       ├── whole_body_tracking/
│       │   ├── tasks/          # 任务定义（环境、MDP、奖励等）
│       │   ├── robots/         # 机器人配置（G1、SMPL）
│       │   └── assets/         # URDF / Mesh 资源
│       └── data/               # ⚠️ 运动数据（需额外下载，见下方）
│           ├── ASAP/           # ASAP 重定向原始数据（pkl）
│           └── motion_npz/     # 转换后的运动数据（npz）
└── scripts/
    ├── rsl_rl/
    │   ├── train.py            # 训练入口
    │   ├── play.py             # 推理 / 评估入口
    │   └── cli_args.py         # CLI 参数定义
    ├── asap_to_npz.py          # ASAP pkl → npz 转换
    ├── csv_to_npz.py           # LAFAN1 CSV → npz 转换
    ├── batch_asap_to_npz.sh    # 批量 ASAP 转换
    ├── replay_npz.py           # Isaac Sim 中回放 npz
    ├── visualize_npz.py        # 离线骨架可视化 → MP4
    └── upload_npz.py           # 上传 npz 到 WandB Registry
```

---

## ⚠️ 数据下载说明

**运动数据不包含在代码仓库中**，需要额外下载并放置到正确路径。

### 选项 A：ASAP 重定向数据（推荐，无需 WandB）

ASAP 数据包含 52 个 pkl 文件（约 5.6 MB），每个文件对应一个运动片段。

1. **下载**：从 [Release 页面]() 或数据提供方获取 ASAP pkl 文件
2. **放置**：将所有 `.pkl` 文件放入 `source/whole_body_tracking/data/ASAP/`
3. **转换为 npz**（训练所需格式）：

```bash
# 单个文件转换（以 Kobe_level1 为例）
/isaac-sim/python.sh scripts/asap_to_npz.py \
    --input_file source/whole_body_tracking/data/ASAP/0-motions_raw_tairantestbed_smpl_video_Kobe_level1_amass.pkl \
    --output_name Kobe_level1 \
    --output_dir source/whole_body_tracking/data/motion_npz/Kobe_level1 \
    --no_upload --headless

# 批量转换所有 52 个文件
bash scripts/batch_asap_to_npz.sh
```

转换结果存放在 `source/whole_body_tracking/data/motion_npz/<motion_name>/motion.npz`。

### 选项 B：LAFAN1 数据集（需要 HuggingFace）

1. **下载**：[LAFAN1 Retargeting Dataset (HuggingFace)](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset)
2. **转换**：

```bash
/isaac-sim/python.sh scripts/csv_to_npz.py \
    --input_file {motion}.csv \
    --input_fps 30 \
    --output_name {motion_name} \
    --headless
```

### 数据大小参考

| 数据 | 大小 | 说明 |
|------|------|------|
| `data/ASAP/` | ~5.6 MB | 52 个 ASAP pkl 文件（原始重定向数据） |
| `data/motion_npz/` | ~22 MB | 转换后的 npz 文件（可由 ASAP 转换生成） |
| `assets/unitree_description/` | ~173 MB | URDF + Mesh（随代码仓库发布） |

---

## 🚀 快速开始

### 前提条件

- NVIDIA GPU（≥6 GB VRAM，推荐 RTX 3060 及以上）
- 已按上方 [环境配置](#-环境配置必读) 完成 Docker 镜像构建和容器启动
- 容器内已执行 `setup.py develop` 安装 Extension（Dockerfile 已自动完成）

### 1. 准备运动数据

参见上方 [⚠️ 数据下载说明](#️-数据下载说明)，下载 ASAP pkl 文件并转换为 npz。

### 2. 训练

> **显存建议**：
> | GPU | VRAM | 推荐 `--num_envs` | 说明 |
> |-----|------|-------------------|------|
> | RTX 3050 / 同级 | 4 GB | `32` | 接近极限，建议无头模式 |
> | RTX 3060 / 同级 | 6 GB | `64` | 显存极限，**必须**使用无头模式 |
> | RTX 3080 / 同级 | 10 GB | `256` | 可开启 GUI，仍建议无头模式录制 |
> | RTX 4090 / A100 | 24 GB+ | `1024`+ | 无限制 |
>
> ⚠️ **低于 8 GB 显存的 GPU 请务必使用 `--headless` 无头模式**，并通过 `--video` 录制视频查看训练效果（GUI 渲染会额外占用约 2 GB 显存）。

```bash
# 基础训练（本地 npz，无需 WandB）
# RTX 3060 (6GB): --num_envs 64   RTX 3050 (4GB): --num_envs 32
/isaac-sim/python.sh scripts/rsl_rl/train.py \
    --task=Tracking-Flat-G1-v0 \
    --motion_file source/whole_body_tracking/data/motion_npz/Kobe_level1/motion.npz \
    --num_envs 64 \
    --headless

# 训练并录制过程视频（低显存 GPU 查看训练效果的推荐方式）
/isaac-sim/python.sh scripts/rsl_rl/train.py \
    --task=Tracking-Flat-G1-v0 \
    --motion_file source/whole_body_tracking/data/motion_npz/Kobe_level1/motion.npz \
    --num_envs 64 \
    --video --video_interval 3072 --video_length 96 \
    --headless --enable_cameras
# 视频保存至 logs/rsl_rl/g1_flat/<timestamp>/videos/train/rl-video-step-*.mp4
```

训练日志保存至 `logs/rsl_rl/g1_flat/<timestamp>/`，包含：
- `model_*.pt`：模型检查点（每 500 iter）
- `events.out.tfevents.*`：TensorBoard 日志
- `videos/train/`：训练视频（需 `--video` 参数）

**查看训练曲线**：

```bash
tensorboard --logdir logs/rsl_rl/g1_flat/ --port 6007
```

### 3. 推理 / 评估

> ⚠️ **低于 8 GB 显存的 GPU 请使用 `--headless --enable_cameras` 无头模式 + `--video` 录制视频**，不要开启 GUI（Isaac Sim GUI 本身占用约 2 GB 显存）。

```bash
# 从本地检查点回放 + 录制视频（低显存 GPU 推荐方式）
/isaac-sim/python.sh scripts/rsl_rl/play.py \
    --task=Tracking-Flat-G1-v0 \
    --num_envs 2 \
    --checkpoint_path logs/rsl_rl/g1_flat/<timestamp>/model_29000.pt \
    --motion_file source/whole_body_tracking/data/motion_npz/Kobe_level1/motion.npz \
    --video --video_length 500 \
    --headless --enable_cameras \
    --no_export --no_termination
# 视频保存至 logs/rsl_rl/g1_flat/<timestamp>/videos/play/rl-video-step-0.mp4
```

**参数说明**：

| 参数 | 说明 |
|------|------|
| `--checkpoint_path` | 直接指定 `.pt` 模型文件路径 |
| `--motion_file` | 本地 `.npz` 运动数据路径 |
| `--no_export` | 跳过 ONNX 导出（避免 headless+camera 死锁） |
| `--no_termination` | 禁用 failure termination，录制完整视频 |
| `--video_length` | 录制帧数（200Hz，500 步 ≈ 2.5s） |

### 4. 可视化运动数据

```bash
# Isaac Sim 中回放
/isaac-sim/python.sh scripts/replay_npz.py \
    --motion_file source/whole_body_tracking/data/motion_npz/Kobe_level1/motion.npz

# 无头录制 MP4
/isaac-sim/python.sh scripts/replay_npz.py \
    --motion_file source/whole_body_tracking/data/motion_npz/Kobe_level1/motion.npz \
    --video --video_output video/Kobe_level1.mp4 \
    --headless --enable_cameras

# 离线骨架可视化（无需 GPU 渲染）
python scripts/visualize_npz.py \
    --motion_file source/whole_body_tracking/data/motion_npz/Kobe_level1/motion.npz \
    --output video/Kobe_level1_skeleton.mp4
```

---

## 🏷️ 注册的 Gym 环境

| 环境 ID | 说明 |
|---------|------|
| `Tracking-Flat-G1-v0` | G1 全身追踪（标准 200Hz） |
| `Tracking-Flat-G1-Wo-State-Estimation-v0` | 不依赖线速度估计（部署友好） |
| `Tracking-Flat-G1-Low-Freq-v0` | 低频控制版本（100Hz） |

---

## 🔧 技术细节

- **算法**：RSL-RL PPO（Proximal Policy Optimization）
- **观测空间**：关节相位 + anchor 姿态误差 + 本体感知（含噪声）
- **Critic 额外输入**：全身 body 相对位姿（特权信息）
- **奖励**：DeepMimic 风格 — 关节位置 / 姿态 / 线速度 / 角速度跟踪 + 正则化
- **终止条件**：anchor 高度 / 姿态偏差 > 阈值，或末端 body 高度异常
- **动作空间**：29D 关节位置（全身所有关节）
- **控制频率**：200Hz（decimation=1）

---

## 📦 依赖

| 依赖 | 版本 | 说明 |
|------|------|------|
| Isaac Sim | 5.0.0 | NVIDIA 物理仿真引擎 |
| Isaac Lab | 1.4.1 | 机器人学习框架 |
| PyTorch | 2.7.0+cu128 | 深度学习框架 |
| pytorch3d | 0.7.9 | 3D 变换（需从源码编译） |
| smplx | - | SMPL 人体模型 |
| RSL-RL | - | PPO 实现（Isaac Lab 内置） |
| CUDA Toolkit | 12.8 | GPU 计算 |

---

## 📄 License

MIT License. 详见 [LICENSE](LICENSE)。

---

## 🙏 致谢

- [Isaac Lab](https://github.com/isaac-sim/IsaacLab) — NVIDIA 机器人学习框架
- [DeepMimic](https://xbpeng.github.io/projects/DeepMimic/index.html) — 运动模仿学习方法论
- [Unitree G1](https://www.unitree.com/g1/) — 人形机器人平台
- [ASAP](https://github.com/) — 人体运动重定向数据
