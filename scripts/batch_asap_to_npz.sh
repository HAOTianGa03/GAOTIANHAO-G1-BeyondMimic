#!/usr/bin/env bash
# batch_asap_to_npz.sh
# 批量将 ASAP data 目录下所有 pkl 文件转换为 BeyondMimic npz 并上传到 WandB Registry
#
# Usage:
#   bash scripts/beyond_mimic/batch_asap_to_npz.sh [OPTIONS]
#
# Options:
#   --asap_dir DIR      ASAP pkl 文件目录（默认: source/whole_body_tracking/data/ASAP）
#   --output_dir DIR    本地 npz 临时输出目录（默认: /tmp/asap_npz）
#   --output_fps N      目标帧率（默认: 50）
#   --no_upload         只保存本地 npz，不上传 WandB
#   --wandb_project P   WandB 项目名（默认: asap_to_npz）
#   --filter PATTERN    只处理文件名包含 PATTERN 的 pkl（例如: walk）
#   --dry_run           只打印要处理的文件，不实际运行
#
# Example:
#   # 转换所有文件并上传
#   bash scripts/beyond_mimic/batch_asap_to_npz.sh
#
#   # 只转换 walk 相关文件，不上传
#   bash scripts/beyond_mimic/batch_asap_to_npz.sh --filter walk --no_upload
#
#   # 使用自定义目录
#   bash scripts/beyond_mimic/batch_asap_to_npz.sh \
#       --asap_dir /data/ASAP --output_dir /data/npz --output_fps 50
#
# 注意: 必须从项目根目录执行，且使用 Isaac Sim Python:
#   cd /workspace/Humandex
#   bash scripts/beyond_mimic/batch_asap_to_npz.sh

set -euo pipefail

# ---- 默认参数 ----
ASAP_DIR="source/whole_body_tracking/data/ASAP"
OUTPUT_DIR="/tmp/asap_npz"
OUTPUT_FPS=50
NO_UPLOAD=""
WANDB_PROJECT="asap_to_npz"
FILTER=""
DRY_RUN=0

# ---- 解析参数 ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --asap_dir)     ASAP_DIR="$2";       shift 2 ;;
        --output_dir)   OUTPUT_DIR="$2";     shift 2 ;;
        --output_fps)   OUTPUT_FPS="$2";     shift 2 ;;
        --no_upload)    NO_UPLOAD="--no_upload"; shift ;;
        --wandb_project) WANDB_PROJECT="$2"; shift 2 ;;
        --filter)       FILTER="$2";         shift 2 ;;
        --dry_run)      DRY_RUN=1;           shift ;;
        *) echo "[ERROR] Unknown argument: $1"; exit 1 ;;
    esac
done

PYTHON="/isaac-sim/python.sh"
SCRIPT="scripts/beyond_mimic/asap_to_npz.py"

echo "========================================"
echo " ASAP → BeyondMimic Batch Converter"
echo "========================================"
echo "  ASAP dir    : $ASAP_DIR"
echo "  Output dir  : $OUTPUT_DIR"
echo "  Output FPS  : $OUTPUT_FPS"
echo "  WandB proj  : $WANDB_PROJECT"
echo "  No upload   : ${NO_UPLOAD:-false}"
echo "  Filter      : ${FILTER:-<all>}"
echo "========================================"

# ---- 收集 pkl 文件 ----
mapfile -t PKL_FILES < <(find "$ASAP_DIR" -name "*.pkl" | sort)

if [[ ${#PKL_FILES[@]} -eq 0 ]]; then
    echo "[ERROR] No pkl files found in: $ASAP_DIR"
    exit 1
fi

# 过滤
if [[ -n "$FILTER" ]]; then
    FILTERED=()
    for f in "${PKL_FILES[@]}"; do
        if [[ "$f" == *"$FILTER"* ]]; then
            FILTERED+=("$f")
        fi
    done
    PKL_FILES=("${FILTERED[@]}")
fi

echo "[INFO] Found ${#PKL_FILES[@]} pkl file(s) to process"
echo ""

TOTAL=${#PKL_FILES[@]}
SUCCESS=0
FAIL=0
FAIL_LIST=()

for i in "${!PKL_FILES[@]}"; do
    PKL="${PKL_FILES[$i]}"
    BASENAME=$(basename "$PKL" .pkl)

    # 提取运动名称（去掉前缀 "0-motions_raw_tairantestbed_smpl_video_" 或 "0-TairanTestbed_TairanTestbed_"）
    MOTION_NAME=$(echo "$BASENAME" \
        | sed 's/^0-motions_raw_tairantestbed_smpl_video_//' \
        | sed 's/^0-TairanTestbed_TairanTestbed_//' \
        | sed 's/_amass$//' \
        | sed 's/_filter_amass$//' \
        | sed 's/_filter$//')

    # 单次 npz 保存路径（每个 motion 独立子目录）
    MOTION_OUT_DIR="$OUTPUT_DIR/$MOTION_NAME"

    NUM=$((i + 1))
    echo "------------------------------------------------------------"
    echo "[$NUM/$TOTAL] $BASENAME"
    echo "  motion_name : $MOTION_NAME"
    echo "  output_dir  : $MOTION_OUT_DIR"

    if [[ $DRY_RUN -eq 1 ]]; then
        echo "  [DRY RUN] Skipping actual conversion."
        continue
    fi

    # 跳过已存在的（断点续传）
    if [[ -f "$MOTION_OUT_DIR/motion.npz" && -z "$NO_UPLOAD" ]]; then
        echo "  [SKIP] $MOTION_OUT_DIR/motion.npz already exists."
        ((SUCCESS++)) || true
        continue
    fi

    CMD="$PYTHON $SCRIPT \
        --input_file $PKL \
        --output_name $MOTION_NAME \
        --output_fps $OUTPUT_FPS \
        --output_dir $MOTION_OUT_DIR \
        --wandb_project $WANDB_PROJECT \
        $NO_UPLOAD \
        --headless"

    echo "  Running..."
    if eval "$CMD"; then
        echo "  [OK] $MOTION_NAME"
        ((SUCCESS++)) || true
    else
        echo "  [FAIL] $MOTION_NAME"
        ((FAIL++)) || true
        FAIL_LIST+=("$MOTION_NAME")
    fi
done

echo ""
echo "========================================"
echo " Batch conversion complete"
echo "  Total   : $TOTAL"
echo "  Success : $SUCCESS"
echo "  Failed  : $FAIL"
if [[ $FAIL -gt 0 ]]; then
    echo "  Failed motions:"
    for name in "${FAIL_LIST[@]}"; do
        echo "    - $name"
    done
fi
echo "========================================"
