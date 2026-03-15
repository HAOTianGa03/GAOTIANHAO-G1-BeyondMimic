# BeyondMimic Motion Tracking — Scripts

本目录包含 BeyondMimic 运动追踪框架的训练、评估和数据处理脚本。

对应的扩展包位于：`/workspace/Humandex/source/whole_body_tracking/`

---

## 目录结构

```
scripts/beyond_mimic/
├── README.md                  # 本文件
├── asap_to_npz.py             # 将 ASAP joblib pkl 转换为 npz，并上传至 WandB Registry
├── batch_asap_to_npz.sh       # 批量转换 ASAP 目录下所有 pkl 文件
├── csv_to_npz.py              # 将 CSV 格式的动作捕捉数据转换为 npz，并上传至 WandB Registry
├── replay_npz.py              # 在 Isaac Sim 中回放 npz 动作文件（需要 WandB Registry）
├── upload_npz.py              # 单独上传 npz 文件到 WandB Registry
└── rsl_rl/
    ├── cli_args.py            # RSL-RL 命令行参数定义
    ├── train.py               # 训练入口脚本
    └── play.py                # 推理/评估脚本（支持从 WandB 加载模型）
```

---

## 前置条件

### 1. 安装扩展包

```bash
cd /workspace/Humandex
/isaac-sim/python.sh source/whole_body_tracking/setup.py develop
```

### 2. 下载 URDF 资源（已完成，资源在）

```
source/whole_body_tracking/whole_body_tracking/assets/unitree_description/
```

### 3. 配置 WandB

BeyondMimic 使用 WandB Registry 管理动作文件，训练前需要登录：

```bash
/isaac-sim/python.sh -c "import wandb; wandb.login()"
```

---

## 数据准备：ASAP → npz（推荐）

ASAP 数据位于 `source/whole_body_tracking/data/ASAP/`，为 joblib pkl 格式。

### 关节映射说明

ASAP 重定向的 G1 数据含 **23 个 dof**（G1 完整 29 关节去掉手腕 6 个）：

| ASAP dof 索引 | G1 关节名 |
|---|---|
| 0–5 | 左腿 (hip_pitch/roll/yaw, knee, ankle_pitch/roll) |
| 6–11 | 右腿 (hip_pitch/roll/yaw, knee, ankle_pitch/roll) |
| 12–14 | 腰部 (waist_yaw/roll/pitch) |
| 15–18 | 左臂 (shoulder_pitch/roll/yaw, elbow) |
| 19–22 | 右臂 (shoulder_pitch/roll/yaw, elbow) |
| — | 左/右 wrist_roll/pitch/yaw（固定为 0） |

### 转换单个文件

```bash
cd /workspace/Humandex
/isaac-sim/python.sh scripts/beyond_mimic/asap_to_npz.py \
    --input_file source/whole_body_tracking/data/ASAP/0-motions_raw_tairantestbed_smpl_video_walk_level1_filter_amass.pkl \
    --output_name walk_level1 \
    --output_fps 50 \
    --headless
```

不上传 WandB，仅保存本地：

```bash
/isaac-sim/python.sh scripts/beyond_mimic/asap_to_npz.py \
    --input_file ... --output_name walk_level1 --no_upload --headless
```

### 批量转换所有 ASAP 文件

```bash
cd /workspace/Humandex
bash scripts/beyond_mimic/batch_asap_to_npz.sh
```

常用选项：

```bash
# 只转换 walk 相关，不上传
bash scripts/beyond_mimic/batch_asap_to_npz.sh --filter walk --no_upload

# 预览要处理的文件（不实际运行）
bash scripts/beyond_mimic/batch_asap_to_npz.sh --dry_run

# 自定义输出目录和 WandB 项目
bash scripts/beyond_mimic/batch_asap_to_npz.sh \
    --output_dir /data/asap_npz \
    --wandb_project my_project
```

---

## 数据准备：CSV → npz

下载 Unitree 重定向的 LAFAN1 数据集：
- [HuggingFace: LAFAN1_Retargeting_Dataset](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset)

转换并上传运动文件：

```bash
cd /workspace/Humandex
/isaac-sim/python.sh scripts/beyond_mimic/csv_to_npz.py \
    --input_file {motion_name}.csv \
    --input_fps 30 \
    --output_name {motion_name} \
    --headless
```

回放验证：

```bash
/isaac-sim/python.sh scripts/beyond_mimic/replay_npz.py \
    --registry_name={your-org}-org/wandb-registry-motions/{motion_name}
```

---

## 训练

```bash
cd /workspace/Humandex
/isaac-sim/python.sh scripts/beyond_mimic/rsl_rl/train.py \
    --task=Tracking-Flat-G1-v0 \
    --registry_name {your-org}-org/wandb-registry-motions/{motion_name} \
    --headless \
    --logger wandb \
    --log_project_name {project_name} \
    --run_name {run_name}
```

### 可用任务 ID

| 任务 ID | 说明 |
|--------|------|
| `Tracking-Flat-G1-v0` | G1 全身运动追踪（标准频率 200Hz） |
| `Tracking-Flat-G1-Wo-State-Estimation-v0` | 不依赖线速度估计的版本 |
| `Tracking-Flat-G1-Low-Freq-v0` | 低频控制版本（100Hz） |
| `Tracking-Flat-Humanoid-v0` | 通用人形机器人版本 |

---

## 推理/评估

```bash
cd /workspace/Humandex
/isaac-sim/python.sh scripts/beyond_mimic/rsl_rl/play.py \
    --task=Tracking-Flat-G1-v0 \
    --num_envs=2 \
    --wandb_path={your-org}/{project_name}/{run_id}
```

---

## 代码结构（扩展包）

```
source/whole_body_tracking/whole_body_tracking/
├── assets/
│   └── unitree_description/        # G1/SMPL URDF 文件
├── robots/
│   ├── g1.py                       # G1 ArticulationCfg + 物理参数
│   └── smpl.py                     # SMPL 人体模型配置
├── tasks/
│   └── tracking/
│       ├── tracking_env_cfg.py     # 主 MDP 配置
│       ├── mdp/
│       │   ├── commands.py         # MotionCommand（运动加载+自适应采样）
│       │   ├── observations.py     # 观测函数
│       │   ├── rewards.py          # DeepMimic 风格奖励
│       │   ├── events.py           # 域随机化
│       │   └── terminations.py    # 终止条件
│       └── config/
│           ├── g1/                 # G1 特定配置 + gym 注册
│           └── humanoid/           # 通用人形配置
└── utils/
    ├── my_on_policy_runner.py      # 带 ONNX 导出的 PPO Runner
    └── exporter.py                 # ONNX 模型导出工具
```

---

## 参考

- 论文：[arXiv:2508.08241](https://arxiv.org/abs/2508.08241)
- 项目主页：[beyondmimic.github.io](https://beyondmimic.github.io)
- 原始仓库：[HybridRobotics/whole_body_tracking](https://github.com/HybridRobotics/whole_body_tracking)
