# Docker 环境搭建问题总结

## 概述

本文档记录了在构建和运行 BeyondMimic Docker 环境过程中遇到的所有问题及其解决方案。

---

## 问题列表

### 1. ❌ pytorch3d 构建失败

**错误信息**：
```
ModuleNotFoundError: No module named 'torch._vendor.packaging._structures'
```

**原因**：torch 的 vendor 目录中的 packaging 模块不完整，缺少 `_structures.py` 文件

**解决方案**：

在 Dockerfile 中添加以下步骤：

```dockerfile
# 安装最新版 packaging 并复制到 torch 的 vendor 目录
RUN ${ISAACLAB_PATH}/_isaac_sim/python.sh -m pip install "packaging>=24.0" && \
    rm -f /app/IsaacLab/_isaac_sim/exts/omni.isaac.ml_archive/pip_prebundle/torch/_vendor/packaging/_structures.py && \
    cp /app/IsaacLab/_isaac_sim/kit/python/lib/python3.11/site-packages/packaging/_structures.py \
       /app/IsaacLab/_isaac_sim/exts/omni.isaac.ml_archive/pip_prebundle/torch/_vendor/packaging/_structures.py

# 使用 --no-build-isolation 参数安装 pytorch3d
RUN export FORCE_CUDA=1 && \
    export CUDA_HOME=/usr/local/cuda-12.8 && \
    ${ISAACLAB_PATH}/_isaac_sim/python.sh -m pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git" && \
    ${ISAACLAB_PATH}/_isaac_sim/python.sh -m pip install --no-build-isolation "git+https://github.com/mattloper/chumpy@9b045ff5d6588a24a0bab52c83f032e2ba433e17"
```

**关键点**：
- 先安装最新版 `packaging>=24.0`
- 删除 torch vendor 目录中的旧文件（可能是符号链接）
- 复制新文件到正确位置
- 使用 `--no-build-isolation` 参数避免构建隔离问题

---

### 2. ❌ whole_body_tracking 扩展安装失败

**错误信息**：
```
error: package directory 'whole_body_tracking' does not exist
```

**原因**：setup.py 的工作目录路径错误，setup.py 在错误的目录下寻找 `whole_body_tracking` 包目录

**解决方案**：

修改 Dockerfile 中的安装命令：

```dockerfile
# 修改前（错误）
RUN cd ${MIMIC_MODEL_PATH} && \
    ${ISAACLAB_PATH}/_isaac_sim/python.sh source/whole_body_tracking/setup.py develop

# 修改后（正确）
RUN cd ${MIMIC_MODEL_PATH}/source/whole_body_tracking && \
    ${ISAACLAB_PATH}/_isaac_sim/python.sh setup.py develop
```

**关键点**：
- 切换到 `source/whole_body_tracking/` 目录
- 直接运行 `setup.py develop`，不需要指定完整路径

---

### 3. ❌ flatdict 缺失

**错误信息**：
```
ModuleNotFoundError: No module named 'flatdict'
```

**原因**：Isaac Lab 依赖的 `flatdict` 包未在 Dockerfile 中安装

**解决方案**：

**方案 A：在容器中安装（快速修复）**
```bash
docker exec mimic-model /isaac-sim/python.sh -m pip install flatdict
```

**方案 B：在 Dockerfile 中添加（永久修复）**
```dockerfile
RUN ${ISAACLAB_PATH}/_isaac_sim/python.sh -m pip install smplx termcolor dm_tree prettytable flatdict && \
    ${ISAACLAB_PATH}/_isaac_sim/python.sh -m pip install "git+https://github.com/otaheri/chamfer_distance.git" && \
    ${ISAACLAB_PATH}/_isaac_sim/python.sh -m pip install "git+https://github.com/lixiny/manotorch.git" && \
    ${ISAACLAB_PATH}/_isaac_sim/python.sh -m pip install "git+https://github.com/otaheri/bps_torch"
```

**关键点**：
- `flatdict` 是 Isaac Lab 的必需依赖
- 应该与其他基础依赖一起安装

---

#