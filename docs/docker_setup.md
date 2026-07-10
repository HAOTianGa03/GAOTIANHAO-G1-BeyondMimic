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
cd /workspace/mimic_model/source/whole_body_tracking
/isaac-sim/python.sh -m pip install -e .
```

> **说明**：以 editable 模式安装，代码修改立即生效，无需重新安装。
>
> ⚠️ **不要**使用 `setup.py develop`（legacy easy_install 解析器）——在解析 `onnxscript` 的 `typing_extensions` 依赖时会报
> `error: [Errno 2] No such file or directory` 读取 `setuptools/_vendor/typing_extensions-*.dist-info/METADATA`。
> 用 `pip install -e .`（pip 现代 resolver）可以避免这个问题。另外必须在 `source/whole_body_tracking` 目录下执行
> （不能带相对路径从别的目录调用 `setup.py`），因为 `setuptools` 的 `packages=[...]` 是相对 CWD 解析的。

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

## 调试记录：RTX 50 系（Blackwell）环境的已知问题与修复（2026-07-10）

以下问题是在下述硬件/驱动环境上从零搭建本环境时实测发现的，均已修复并合并进 Dockerfile。记录在此供以后遇到类似 GPU（sm_120，即 RTX 50 系列 Blackwell 架构）或复现构建问题时参考。

### 调试所用硬件 / 驱动环境

| 项目 | 值 |
|------|-----|
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU（8 GB VRAM，笔记本版，本文档"硬件要求"表格里的最低配置） |
| GPU 架构 | Blackwell，Compute Capability / SM 12.0（`sm_120`） |
| NVIDIA 驱动版本 | 580.159.03 |
| 宿主机 OS | Ubuntu 24.04.4 LTS（内核 6.17.0-35-generic） |
| Docker 版本 | 29.1.3 |
| nvidia-container-toolkit 版本 | 1.19.1-1 |
| 容器内 torch | 2.7.0+cu128（`torch.cuda.get_device_capability(0)` 返回 `(12, 0)`） |

> sm_120 不在 torch 2.7.0+cu128 预编译的 SM 列表里，但通过 CUDA 12.8 的 PTX JIT 前向兼容，运行时能自动编译到 sm_120，无需额外升级 CUDA 版本或自行编译 torch。`TORCH_CUDA_ARCH_LIST` 构建参数（用于 pytorch3d 源码编译）需要显式传 `"12.0"`：
> ```bash
> --build-arg TORCH_CUDA_ARCH_LIST="12.0"
> ```

### 已修复的问题

**1. torch 版本未锁定导致拉到不兼容版本**
`isaaclab_rl`/`rsl-rl-lib` 对 `torch` 的依赖没有锁版本，`isaaclab.sh --install` 会让 pip 解析到最新版（实测拉到 2.13.0+cu130），这个版本和 Isaac Sim 5.0.0 内置的 Kit Python 不兼容，加载 `isaacsim.core.*`/`isaaclab_assets`/`isaaclab_tasks` 等 Kit 扩展时报 `AttributeError: module 'torch' has no attribute 'Tensor'/'jit'`。Dockerfile 现已显式锁定 `torch==2.7.0`、`torchvision==0.22.0`（`cu128` 版）、`typing_extensions==4.12.2`。

**2. torch==2.7.0+cu128 wheel 自带的 cusparselt 库 RPATH 错误**
该 wheel 里 `libtorch_cuda.so` 等文件的 RPATH 少写了一段路径（`$ORIGIN/../../cusparselt/lib`，应为 `$ORIGIN/../../nvidia/cusparselt/lib`），导致编译 pytorch3d 时因 `import torch` 报 `ImportError: libcusparseLt.so.0: cannot open shared object file`。修复：在 site-packages 下建一个符号链接 `cusparselt -> nvidia/cusparselt`。

**3.（本次最主要的根因）Isaac Sim 自带扩展里一个占位空目录会遮蔽真实 torch 安装**
`omni.isaac.ml_archive` 扩展的 `pip_prebundle/torch` 目录只剩一个无关紧要的遗留文件，没有 `__init__.py`，因为该扩展设置了 `order = -1000`（最先加载），这个空目录会先于真实 site-packages 出现在 `sys.path` 上。Python 会把它当作一个空的命名空间包。只要脚本是先 `AppLauncher()` 后 `import torch`（`scripts/rsl_rl/train.py`、`play.py` 都是这个顺序），Kit 扩展系统触发的第一次 `import torch` 就会被这个空壳顶替，之后整个进程里 `sys.modules['torch']` 都是这个坏掉的模块，报 `AttributeError: module 'torch' has no attribute 'Tensor'/'jit'`。修复：构建时直接删除这个占位目录（`rm -rf .../omni.isaac.ml_archive/pip_prebundle/torch`）。

**4. `setup.py develop` 的 legacy easy_install 解析器 bug**
见上方 Step 3 的说明，已改用 `pip install -e .`。

**5. 容器缺少 `openssh-client`**
基础镜像 `nvcr.io/nvidia/isaac-sim:5.0.0` 没有装 `openssh-client`，仅挂载 `~/.ssh` 不足以在容器内 `git push` / `ssh -T git@github.com`，已加入 Dockerfile 的 apt 依赖列表。若要在容器内直接提交代码，启动容器时还需加上只读挂载：
```bash
-v ~/.ssh:/root/.ssh:ro
```

### 已知但无害、未修复的残留问题

即使以上问题都修复了，Kit 首次自动加载 `isaaclab_tasks` 扩展时日志里仍会出现一条：
```
TypeError: Type parameter +RV without a default follows type parameter with a default
```
根因是 IsaacLab v2.2.0 的 `isaaclab_tasks` 会无差别 import 仓库内所有示例任务包，其中一个和 G1 无关的 Franka 机械臂任务配置 `import torchvision.utils`，顺带触发了 `torch._inductor.utils.py` 里一段本身就有 PEP 696 顺序问题的代码（`class CachedMethod(Protocol, Generic[P, RV])`，属于 torch 2.7.0+cu128 自身瑕疵）。这只影响 Kit 扩展系统"第一次"自动加载的记录，不影响后续任何脚本正常 `import isaaclab_tasks` 或 gym 环境注册，无需处理。

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
