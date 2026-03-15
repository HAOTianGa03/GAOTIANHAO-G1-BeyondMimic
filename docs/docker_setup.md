# Docker 环境搭建指南

本文档介绍如何使用 Docker 构建和运行 BeyondMimic 框架的开发环境。

---

## 前提条件

### 硬件要求

| 项目 | 最低要求 | 推荐 |
|------|----------|------|
| GPU | NVIDIA RTX 2080 (6 GB VRAM) | RTX 3090 / 4090 (24 GB) |
| RAM | 16 GB | 32 GB |
| 磁盘 | 50 GB（Docker 镜像约 28 GB） | 100 GB |

### 软件要求

- Docker Engine ≥ 20.10
- NVIDIA Driver ≥ 535（支持 CUDA 12.x）
- [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

### 安装 nvidia-container-toolkit（仅需执行一次）

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
    sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

---

## Step 1：构建 Docker 镜像

```bash
cd /path/to/mimic_model

# 标准构建
docker image build --network host . -t mimic-model:local

# 如需代理：
docker image build --network host \
    --build-arg HTTP_PROXY=http://127.0.0.1:7897 \
    --build-arg HTTPS_PROXY=http://127.0.0.1:7897 \
    . -t mimic-model:local
```

> ⏱ 构建耗时约 30–60 分钟。其中 pytorch3d 源码编译约 15 分钟。

### GPU 架构选择

pytorch3d 需要针对你的 GPU 的 SM 版本编译。默认值 `"7.5 8.0 8.6 8.9"` 覆盖了 RTX 2000–4000 系列。
如果你的 GPU 不在其中，通过 `--build-arg` 覆盖：

| GPU 系列 | SM 版本 | 构建参数 |
|----------|---------|----------|
| RTX 2080 Ti | 7.5 | `--build-arg TORCH_CUDA_ARCH_LIST="7.5"` |
| A100 | 8.0 | `--build-arg TORCH_CUDA_ARCH_LIST="8.0"` |
| RTX 3060 / 3090 | 8.6 | `--build-arg TORCH_CUDA_ARCH_LIST="8.6"` |
| RTX 4090 / 4080 | 8.9 | `--build-arg TORCH_CUDA_ARCH_LIST="8.9"` |
| H100 | 9.0 | `--build-arg TORCH_CUDA_ARCH_LIST="9.0"` |

---

## Step 2：启动容器

```bash
docker run --gpus all -dit \
    --restart=always \
    --privileged \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $(pwd):/workspace/mimic_model \
    -v /dev:/dev \
    --network host \
    -e DISPLAY=$DISPLAY \
    --name mimic-model \
    -w /workspace/mimic_model \
    mimic-model:local /bin/bash
```

验证 GPU 可访问：

```bash
docker exec mimic-model nvidia-smi
```

---

## Step 3：安装 Extension（容器内）

进入容器后安装 `whole_body_tracking` 扩展包：

```bash
docker exec -it mimic-model bash

# 在容器内执行：
cd /workspace/mimic_model
/isaac-sim/python.sh source/whole_body_tracking/setup.py develop
```

> **说明**：`setup.py develop` 以 editable 模式安装，代码修改立即生效，无需重新安装。

---

## Step 4：验证安装

```bash
# 检查核心依赖
docker exec mimic-model \
    /isaac-sim/python.sh -c \
    "import torch, pytorch3d; print('torch:', torch.__version__); print('pytorch3d:', pytorch3d.__version__)"

# 检查 Extension 安装
docker exec mimic-model \
    /isaac-sim/python.sh -m pip show whole_body_tracking

# 检查 Gym 环境注册
docker exec mimic-model \
    /isaac-sim/python.sh -c "import whole_body_tracking; import gymnasium as gym; print([e.id for e in gym.registry.values() if 'Tracking' in e.id])"
```

预期输出：

```
torch: 2.7.0+cu128
pytorch3d: 0.7.9
['Tracking-Flat-G1-v0', 'Tracking-Flat-G1-Wo-State-Estimation-v0', 'Tracking-Flat-G1-Low-Freq-v0']
```

---

## Step 5：准备数据 & 训练

详见 [README.md](../README.md) 中的数据下载说明和训练命令。

快速验证（容器内）：

```bash
cd /workspace/mimic_model

# 1. 转换一个 ASAP 数据（确保已将 pkl 放入 data/ASAP/）
/isaac-sim/python.sh scripts/asap_to_npz.py \
    --input_file source/whole_body_tracking/data/ASAP/0-motions_raw_tairantestbed_smpl_video_Kobe_level1_amass.pkl \
    --output_name Kobe_level1 \
    --output_dir source/whole_body_tracking/data/motion_npz/Kobe_level1 \
    --no_upload --headless

# 2. 启动训练
/isaac-sim/python.sh scripts/rsl_rl/train.py \
    --task=Tracking-Flat-G1-v0 \
    --motion_file source/whole_body_tracking/data/motion_npz/Kobe_level1/motion.npz \
    --num_envs 64 --headless
```

---

## Python 可执行文件

在容器内，**始终使用 Isaac Sim 的 Python**：

```bash
# 直接调用
/isaac-sim/python.sh <script.py>

# pip 安装
/isaac-sim/python.sh -m pip install <package>

# 或使用 Isaac Lab 包装器
/app/IsaacLab/isaaclab.sh -p <script.py>
```

> ⚠️ **不要**使用系统 `python` 或 conda 环境的 Python —— 它们缺少 Isaac Sim / Omniverse 的运行时绑定。

---

## 关键镜像参数

| 项目 | 值 |
|------|-----|
| 基础镜像 | `nvcr.io/nvidia/isaac-sim:5.0.0` |
| 构建镜像 | `mimic-model:local`（约 28 GB） |
| Python | 3.11 |
| PyTorch | 2.7.0+cu128 |
| pytorch3d | 0.7.9（源码编译） |
| CUDA Toolkit | 12.8 |
| Isaac Lab | 1.4.1 |

---

## 常见问题

### Q: 构建 pytorch3d 时报错 `CUDA_HOME not found`

确保 Dockerfile 中 `CUDA_HOME=/usr/local/cuda-12.8` 设置正确。如果使用自定义 Dockerfile，需要先安装 `cuda-toolkit-12-8`。

### Q: 训练报 `PhysX GPU CUDA error`

降低并行环境数：`--num_envs 64`（6 GB VRAM）或 `--num_envs 256`（12 GB VRAM）。

### Q: 容器内 `nvidia-smi` 报错

检查 host 机器的 `nvidia-container-toolkit` 是否正确安装，以及 Docker daemon 是否重启。

### Q: `import whole_body_tracking` 报错

确保已执行 `setup.py develop`，且使用 `/isaac-sim/python.sh` 而非系统 Python。
