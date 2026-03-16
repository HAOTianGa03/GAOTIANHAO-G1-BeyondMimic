FROM nvcr.io/nvidia/isaac-sim:5.0.0 AS base
ENV ISAACSIM_VERSION=5.0.0
ENV ISAACSIM_ROOT_PATH=/isaac-sim
ENV ISAACLAB_PATH=/app/IsaacLab
ENV DOCKER_USER_HOME=/root


# TORCH_CUDA_ARCH_LIST: space-separated SM versions for pytorch3d compilation.
# Override at build time for your GPU:
#   --build-arg TORCH_CUDA_ARCH_LIST="8.6"   (RTX 3060)
#   --build-arg TORCH_CUDA_ARCH_LIST="8.9"   (RTX 4090)
#   --build-arg TORCH_CUDA_ARCH_LIST="9.0"   (H100)
ARG TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6 8.9"
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}

SHELL ["/bin/bash", "-c"]

LABEL version="1.0.0"
LABEL description="BeyondMimic: Whole-Body Motion Tracking for Unitree G1"

ENV LANG=C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive

USER root

# Accept proxy as build-arg
ARG HTTP_PROXY=""
ARG HTTPS_PROXY=""
ENV HTTP_PROXY=${HTTP_PROXY}
ENV HTTPS_PROXY=${HTTPS_PROXY}
ENV http_proxy=${HTTP_PROXY}
ENV https_proxy=${HTTPS_PROXY}
RUN git config --global http.version HTTP/1.1 && \
    git config --global https.version HTTP/1.1 || true

# ==========================================================================
# Step 1: System dependencies
# ==========================================================================
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    libglib2.0-0 \
    ncurses-term \
    wget \
    tini && \
    apt -y autoremove && apt clean autoclean && \
    rm -rf /var/lib/apt/lists/*

# ==========================================================================
# Step 2: Install Isaac Lab
# ==========================================================================
# Clone Isaac Lab v2.2.0 (matching Isaac Sim 5.0.0)
# If you have a local copy, replace this with: COPY IsaacLab/ ${ISAACLAB_PATH}/
RUN git clone --depth 1 --branch v2.2.0 \
    https://github.com/isaac-sim/IsaacLab.git ${ISAACLAB_PATH}

RUN chmod +x ${ISAACLAB_PATH}/isaaclab.sh && \
    ln -sf ${ISAACSIM_ROOT_PATH} ${ISAACLAB_PATH}/_isaac_sim

# Install toml for setup.py parsing
RUN ${ISAACLAB_PATH}/isaaclab.sh -p -m pip install toml

# Install apt deps declared in extension.toml files
RUN --mount=type=cache,target=/var/cache/apt \
    ${ISAACLAB_PATH}/isaaclab.sh -p ${ISAACLAB_PATH}/tools/install_deps.py apt ${ISAACLAB_PATH}/source && \
    apt -y autoremove && apt clean autoclean && \
    rm -rf /var/lib/apt/lists/*

# Create cache directories for Isaac Sim
RUN mkdir -p ${ISAACSIM_ROOT_PATH}/kit/cache && \
    mkdir -p ${DOCKER_USER_HOME}/.cache/ov && \
    mkdir -p ${DOCKER_USER_HOME}/.cache/pip && \
    mkdir -p ${DOCKER_USER_HOME}/.cache/nvidia/GLCache && \
    mkdir -p ${DOCKER_USER_HOME}/.nv/ComputeCache && \
    mkdir -p ${DOCKER_USER_HOME}/.nvidia-omniverse/logs && \
    mkdir -p ${DOCKER_USER_HOME}/.local/share/ov/data && \
    mkdir -p ${DOCKER_USER_HOME}/Documents

# Install Isaac Lab core
RUN --mount=type=cache,target=${DOCKER_USER_HOME}/.cache/pip \
    ${ISAACLAB_PATH}/_isaac_sim/kit/python/bin/python3 -m pip install --upgrade pip && \
    ${ISAACLAB_PATH}/_isaac_sim/kit/python/bin/python3 -m pip uninstall -y packaging 2>/dev/null || true && \
    ${ISAACLAB_PATH}/isaaclab.sh --install && \
    ${ISAACLAB_PATH}/_isaac_sim/kit/python/bin/python3 -m pip uninstall -y packaging 2>/dev/null || true && \
    ${ISAACLAB_PATH}/_isaac_sim/kit/python/bin/python3 -m pip install "packaging==24.2" --no-deps

RUN ${ISAACLAB_PATH}/isaaclab.sh -p -m pip uninstall -y quadprog

# Shell aliases
RUN echo "export ISAACLAB_PATH=${ISAACLAB_PATH}" >> ${DOCKER_USER_HOME}/.bashrc && \
    echo "alias isaaclab=${ISAACLAB_PATH}/isaaclab.sh" >> ${DOCKER_USER_HOME}/.bashrc && \
    echo "alias python=${ISAACLAB_PATH}/_isaac_sim/python.sh" >> ${DOCKER_USER_HOME}/.bashrc && \
    echo "alias python3=${ISAACLAB_PATH}/_isaac_sim/python.sh" >> ${DOCKER_USER_HOME}/.bashrc && \
    echo "alias pip='${ISAACLAB_PATH}/_isaac_sim/python.sh -m pip'" >> ${DOCKER_USER_HOME}/.bashrc && \
    echo "alias tensorboard='${ISAACLAB_PATH}/_isaac_sim/python.sh ${ISAACLAB_PATH}/_isaac_sim/tensorboard'" >> ${DOCKER_USER_HOME}/.bashrc

WORKDIR ${ISAACLAB_PATH}

# ==========================================================================
# Step 3: CUDA Toolkit 12.8 (for pytorch3d compilation)
# ==========================================================================
RUN apt-get update && \
    apt-get install -y --no-install-recommends wget gnupg && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb -O /tmp/cuda-keyring.deb && \
    dpkg -i /tmp/cuda-keyring.deb && \
    rm /tmp/cuda-keyring.deb && \
    apt-get update && \
    apt-get install -y --no-install-recommends cuda-toolkit-12-8 && \
    rm -rf /var/lib/apt/lists/*

ENV CUDA_HOME=/usr/local/cuda-12.8
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}

RUN echo 'export CUDA_HOME=/usr/local/cuda-12.8' >> /root/.bashrc && \
    echo 'export PATH=${CUDA_HOME}/bin:${PATH}' >> /root/.bashrc && \
    echo 'export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}' >> /root/.bashrc

# ==========================================================================
# Step 4: Python dependencies (smplx, pytorch3d, etc.)
# ==========================================================================
RUN ${ISAACLAB_PATH}/_isaac_sim/python.sh -m pip install smplx termcolor dm_tree prettytable && \
    ${ISAACLAB_PATH}/_isaac_sim/python.sh -m pip install "git+https://github.com/otaheri/chamfer_distance.git" && \
    ${ISAACLAB_PATH}/_isaac_sim/python.sh -m pip install "git+https://github.com/lixiny/manotorch.git" && \
    ${ISAACLAB_PATH}/_isaac_sim/python.sh -m pip install "git+https://github.com/otaheri/bps_torch"

# pytorch3d from source (requires FORCE_CUDA + _structures.py fix)
RUN export FORCE_CUDA=1 && \
    export CUDA_HOME=/usr/local/cuda-12.8 && \
    STRUCTS="/app/IsaacLab/_isaac_sim/exts/omni.isaac.ml_archive/pip_prebundle/torch/_vendor/packaging/_structures.py" && \
    rm -f "${STRUCTS}" && \
    printf '%s\n' \
        'class InfinityType:' \
        '    def __repr__(self): return "Infinity"' \
        '    def __hash__(self): return hash(repr(self))' \
        '    def __lt__(self, other): return False' \
        '    def __le__(self, other): return False' \
        '    def __eq__(self, other): return isinstance(other, self.__class__)' \
        '    def __gt__(self, other): return True' \
        '    def __ge__(self, other): return True' \
        '    def __neg__(self): return NegativeInfinity' \
        'Infinity = InfinityType()' \
        'class NegativeInfinityType:' \
        '    def __repr__(self): return "-Infinity"' \
        '    def __hash__(self): return hash(repr(self))' \
        '    def __lt__(self, other): return True' \
        '    def __le__(self, other): return True' \
        '    def __eq__(self, other): return isinstance(other, self.__class__)' \
        '    def __gt__(self, other): return False' \
        '    def __ge__(self, other): return False' \
        '    def __neg__(self): return Infinity' \
        'NegativeInfinity = NegativeInfinityType()' \
        > "${STRUCTS}" && \
    ${ISAACLAB_PATH}/_isaac_sim/python.sh -m pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git" && \
    ${ISAACLAB_PATH}/_isaac_sim/python.sh -m pip install --no-build-isolation "git+https://github.com/mattloper/chumpy@9b045ff5d6588a24a0bab52c83f032e2ba433e17" && \
    ${ISAACLAB_PATH}/_isaac_sim/kit/python/bin/python3 -m pip uninstall -y packaging 2>/dev/null || true

# ==========================================================================
# Step 5: Fix Python path registrations (warp + isaaclab editable install)
# ==========================================================================
RUN SITE="/app/IsaacLab/_isaac_sim/kit/python/lib/python3.11/site-packages" && \
    echo '/app/IsaacLab/_isaac_sim/extscache/omni.warp.core-1.7.1+lx64' \
        > "${SITE}/warp-1.7.1.pth" && \
    cat > "${SITE}/__editable___isaaclab_1_4_1_finder.py" << 'FINDER_EOF'
from __future__ import annotations
import sys
from importlib.machinery import ModuleSpec, PathFinder
from importlib.machinery import all_suffixes as module_suffixes
from importlib.util import spec_from_file_location
from pathlib import Path

MAPPING: dict[str, str] = {'isaaclab': '/app/IsaacLab/source/isaaclab/isaaclab'}
NAMESPACES: dict[str, list[str]] = {}
PATH_PLACEHOLDER = '__editable__.isaaclab-1.4.1.finder.__path_hook__'

class _EditableFinder:
    @classmethod
    def find_spec(cls, fullname, path=None, target=None):
        if fullname in MAPPING:
            return cls._find_spec(fullname, Path(MAPPING[fullname]))
        if fullname in NAMESPACES:
            spec = ModuleSpec(fullname, None)
            spec.submodule_search_locations = NAMESPACES[fullname]
            return spec
        return None
    @classmethod
    def _find_spec(cls, fullname, candidate_path):
        init = candidate_path / '__init__.py'
        if init.is_file():
            return spec_from_file_location(fullname, init,
                submodule_search_locations=[str(candidate_path)])
        for suffix in module_suffixes():
            mod = candidate_path.with_suffix(suffix)
            if mod.is_file():
                return spec_from_file_location(fullname, mod)
        return None

class _EditableNamespaceFinder:
    @classmethod
    def _install_hook(cls, finder):
        if PATH_PLACEHOLDER not in sys.path:
            sys.path.insert(0, PATH_PLACEHOLDER)
        sys.path_hooks.insert(0, finder)
        sys.path_importer_cache.pop(PATH_PLACEHOLDER, None)
    @classmethod
    def find_spec(cls, fullname, path=None, target=None):
        return None
    def __call__(self, path):
        if path == PATH_PLACEHOLDER:
            return self
        raise ImportError

def install():
    if not MAPPING and not NAMESPACES:
        return
    sys.meta_path.append(_EditableFinder())
    if NAMESPACES:
        ns = _EditableNamespaceFinder()
        _EditableNamespaceFinder._install_hook(ns)
FINDER_EOF
    echo 'import __editable___isaaclab_1_4_1_finder; __editable___isaaclab_1_4_1_finder.install()' \
        > "${SITE}/__editable__.isaaclab-1.4.1.pth" && \
    echo "warp + isaaclab path registration done"

# ==========================================================================
# Step 6: Install whole_body_tracking Extension
# ==========================================================================
ENV MIMIC_MODEL_PATH=/workspace/mimic_model

# Copy project source into the image
COPY source/ ${MIMIC_MODEL_PATH}/source/
COPY scripts/ ${MIMIC_MODEL_PATH}/scripts/
COPY README.md ${MIMIC_MODEL_PATH}/

# Install Extension in editable mode
RUN cd ${MIMIC_MODEL_PATH} && \
    ${ISAACLAB_PATH}/_isaac_sim/python.sh source/whole_body_tracking/setup.py develop && \
    echo "whole_body_tracking installed successfully"

# Create data placeholder directories
RUN mkdir -p ${MIMIC_MODEL_PATH}/source/whole_body_tracking/data/ASAP \
             ${MIMIC_MODEL_PATH}/source/whole_body_tracking/data/motion_npz \
             ${MIMIC_MODEL_PATH}/logs \
             ${MIMIC_MODEL_PATH}/video

WORKDIR ${MIMIC_MODEL_PATH}

ENTRYPOINT ["tini", "-s", "--"]
CMD ["/bin/bash"]
