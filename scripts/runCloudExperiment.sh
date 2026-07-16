#!/usr/bin/env bash
# 一键配置 Linux/CUDA 环境，校验数据，训练并评估 SpecLoc 基线。

set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ORIGINAL_PWD="$PWD"
DATASET="${SPECLOC_DATASET:-rsod}"
CONFIG=""
CANONICAL_REL=""
CANONICAL_ROOT=""

ENV_NAME="specloc"
DATA_ROOT=""
DATA_SOURCE="${AITOD_SOURCE:-}"
LICENSE_NOTE="${AITOD_LICENSE_NOTE:-}"
WORK_DIR=""
MODE="all"
SKIP_SETUP=0
DRY_RUN=0
GPUS=1
BATCH_SIZE=4
USE_AMP=1
RESUME=0
MASTER_PORT=29500

usage() {
    cat <<'EOF'
用法：
  bash scripts/runCloudExperiment.sh [选项]

默认执行：配置/更新 Conda 环境 → 安装依赖 → 数据校验 → 测试与环境检查
          → 训练标准基线 → 选择最佳 checkpoint → 评估并导出预测。

默认数据集为 RSOD。仅 AI-TOD-v2 模式要求：
  --data-source TEXT       实际授权下载地址或来源说明
  --license-note TEXT      已核实的数据许可或访问条款

也可通过环境变量提供：
  SPECLOC_DATASET、RSOD_ROOT、AITOD_ROOT、AITOD_SOURCE、AITOD_LICENSE_NOTE

常用选项：
  --dataset rsod|aitod     数据集；默认 rsod
  --data-root PATH         数据根目录；默认 data/RSOD 或 data/AI-TOD
  --work-dir PATH          输出目录；默认按数据集命名
  --env-name NAME          Conda 环境名；默认 specloc
  --gpus N                 GPU 数量；默认 1
  --batch-size 2|4         每卡 batch size；2 时自动累积两步
  --resume                 从工作目录最近 checkpoint 自动恢复
  --no-amp                 关闭 AMP
  --master-port PORT       多卡通信端口；默认 29500
  --skip-setup             使用已有 Conda 环境，不更新依赖
  --setup-only             只配置环境，不要求数据和 GPU
  --check-only             配置环境并完成数据/代码/GPU 验收，不训练
  --dry-run                只显示计划，不执行
  -h, --help               显示帮助

RSOD 最短示例：
  bash scripts/runCloudExperiment.sh --dataset rsod

AI-TOD-v2 示例：
  AITOD_SOURCE='<实际来源>' \
  AITOD_LICENSE_NOTE='<已核实条款>' \
  bash scripts/runCloudExperiment.sh --dataset aitod

外部数据盘示例：
  bash scripts/runCloudExperiment.sh \
    --dataset rsod \
    --data-root /mnt/data/RSOD

AI-TOD-v2 外部数据盘示例：
  bash scripts/runCloudExperiment.sh \
    --dataset aitod \
    --data-root /mnt/data/AI-TOD \
    --data-source '<实际来源>' \
    --license-note '<已核实条款>'
EOF
}

fail() {
    echo "[失败] $*" >&2
    exit 1
}

step() {
    echo
    echo "========================================================================"
    echo "[步骤] $*"
    echo "========================================================================"
}

on_error() {
    local exit_code=$?
    echo "[失败] 脚本在第 ${BASH_LINENO[0]} 行停止，退出码 ${exit_code}。" >&2
    exit "$exit_code"
}
trap on_error ERR

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)
            [[ $# -ge 2 ]] || fail '--dataset 缺少参数'
            DATASET=$2
            shift 2
            ;;
        --data-root)
            [[ $# -ge 2 ]] || fail '--data-root 缺少参数'
            DATA_ROOT=$2
            shift 2
            ;;
        --data-source)
            [[ $# -ge 2 ]] || fail '--data-source 缺少参数'
            DATA_SOURCE=$2
            shift 2
            ;;
        --license-note)
            [[ $# -ge 2 ]] || fail '--license-note 缺少参数'
            LICENSE_NOTE=$2
            shift 2
            ;;
        --work-dir)
            [[ $# -ge 2 ]] || fail '--work-dir 缺少参数'
            WORK_DIR=$2
            shift 2
            ;;
        --env-name)
            [[ $# -ge 2 ]] || fail '--env-name 缺少参数'
            ENV_NAME=$2
            shift 2
            ;;
        --gpus)
            [[ $# -ge 2 ]] || fail '--gpus 缺少参数'
            GPUS=$2
            shift 2
            ;;
        --batch-size)
            [[ $# -ge 2 ]] || fail '--batch-size 缺少参数'
            BATCH_SIZE=$2
            shift 2
            ;;
        --master-port)
            [[ $# -ge 2 ]] || fail '--master-port 缺少参数'
            MASTER_PORT=$2
            shift 2
            ;;
        --resume)
            RESUME=1
            shift
            ;;
        --no-amp)
            USE_AMP=0
            shift
            ;;
        --skip-setup)
            SKIP_SETUP=1
            shift
            ;;
        --setup-only)
            [[ "$MODE" == 'all' ]] || fail '--setup-only 与其他模式参数互斥'
            MODE='setup'
            shift
            ;;
        --check-only)
            [[ "$MODE" == 'all' ]] || fail '--check-only 与其他模式参数互斥'
            MODE='check'
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            fail "未知参数：$1"
            ;;
    esac
done

[[ "$GPUS" =~ ^[1-9][0-9]*$ ]] || fail '--gpus 必须为正整数'
[[ "$MASTER_PORT" =~ ^[1-9][0-9]*$ ]] || fail '--master-port 必须为正整数'
[[ "$BATCH_SIZE" == '2' || "$BATCH_SIZE" == '4' ]] || fail '--batch-size 仅支持 2 或 4'

case "$DATASET" in
    rsod)
        CONFIG='configs/rsodCenternetR50.py'
        CANONICAL_REL='data/RSOD'
        DATA_ROOT="${DATA_ROOT:-${RSOD_ROOT:-$REPO_ROOT/data/RSOD}}"
        WORK_DIR="${WORK_DIR:-work_dirs/rsod_centernet_r50}"
        ;;
    aitod)
        CONFIG='configs/aitodCenternetR50.py'
        CANONICAL_REL='data/AI-TOD'
        DATA_ROOT="${DATA_ROOT:-${AITOD_ROOT:-$REPO_ROOT/data/AI-TOD}}"
        WORK_DIR="${WORK_DIR:-work_dirs/aitod_centernet_r50}"
        ;;
    *)
        fail '--dataset 仅支持 rsod 或 aitod'
        ;;
esac
CANONICAL_ROOT="$REPO_ROOT/$CANONICAL_REL"

if [[ "$DATA_ROOT" != /* ]]; then
    DATA_ROOT="$ORIGINAL_PWD/$DATA_ROOT"
fi
if [[ "$WORK_DIR" != /* ]]; then
    WORK_DIR="$REPO_ROOT/$WORK_DIR"
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
    cat <<EOF
[DRY RUN]
模式          : $MODE
数据集        : $DATASET
配置          : $CONFIG
Conda 环境    : $ENV_NAME
跳过环境配置  : $SKIP_SETUP
数据目录      : $DATA_ROOT
工作目录      : $WORK_DIR
GPU 数量      : $GPUS
每卡 batch    : $BATCH_SIZE
AMP           : $USE_AMP
断点续训      : $RESUME
AI-TOD 来源   : $([[ -n "$DATA_SOURCE" ]] && echo provided || echo not-required-or-missing)
AI-TOD 许可   : $([[ -n "$LICENSE_NOTE" ]] && echo provided || echo not-required-or-missing)
EOF
    exit 0
fi

[[ "$(uname -s)" == 'Linux' ]] || fail '本脚本只支持 Linux 云服务器'
[[ "$(uname -m)" == 'x86_64' ]] || fail '当前环境文件只验证过 Linux x86_64'
command -v git >/dev/null 2>&1 || fail '未找到 git'

cd "$REPO_ROOT"

CONDA_BIN="${CONDA_EXE:-}"
if [[ -z "$CONDA_BIN" ]]; then
    CONDA_BIN="$(command -v conda || true)"
fi
[[ -n "$CONDA_BIN" && -x "$CONDA_BIN" ]] || fail '未找到 conda，请先安装 Miniconda/Anaconda'

run_env() {
    "$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" "$@"
}

step '检查 Git 版本'
REVISION="$(git rev-parse HEAD)"
[[ -z "$(git status --porcelain)" ]] || fail 'Git 工作树有未提交改动，请提交后再运行正式实验'
echo "Git commit: $REVISION"

if [[ "$SKIP_SETUP" -eq 0 ]]; then
    step '创建或更新 Conda/CUDA 环境'
    if "$CONDA_BIN" run -n "$ENV_NAME" python -V >/dev/null 2>&1; then
        "$CONDA_BIN" env update -n "$ENV_NAME" -f environment.yml
    else
        "$CONDA_BIN" env create -n "$ENV_NAME" -f environment.yml
    fi

    step '按锁定依赖安装 MMCV 与当前项目'
    run_env python -m pip install -r requirements-cu118.txt
    run_env python -m pip install -r requirements/dev.txt
else
    step '使用已有 Conda 环境'
    "$CONDA_BIN" run -n "$ENV_NAME" python -V >/dev/null 2>&1 \
        || fail "Conda 环境不存在：$ENV_NAME"
fi

if [[ "$MODE" == 'setup' ]]; then
    echo '[完成] 环境配置完成。未检查数据或启动训练。'
    exit 0
fi

if [[ "$DATASET" == 'aitod' ]]; then
    [[ -n "$DATA_SOURCE" ]] || fail 'AI-TOD-v2 缺少 --data-source 或 AITOD_SOURCE'
    [[ -n "$LICENSE_NOTE" ]] || fail 'AI-TOD-v2 缺少 --license-note 或 AITOD_LICENSE_NOTE'
fi
[[ -d "$DATA_ROOT" ]] || fail "数据目录不存在：$DATA_ROOT"

step '连接数据目录'
mkdir -p "$REPO_ROOT/data"
DATA_ROOT_REAL="$(readlink -f "$DATA_ROOT")"
if [[ "$DATA_ROOT_REAL" != "$(readlink -m "$CANONICAL_ROOT")" ]]; then
    if [[ -e "$CANONICAL_ROOT" || -L "$CANONICAL_ROOT" ]]; then
        EXISTING_REAL="$(readlink -f "$CANONICAL_ROOT" || true)"
        [[ "$EXISTING_REAL" == "$DATA_ROOT_REAL" ]] \
            || fail "$CANONICAL_ROOT 已指向其他数据，请人工确认后处理"
    else
        ln -s "$DATA_ROOT_REAL" "$CANONICAL_ROOT"
    fi
fi
echo "$DATASET: $(readlink -f "$CANONICAL_ROOT")"

step "验证 $DATASET 数据"
if [[ "$DATASET" == 'rsod' ]]; then
    run_env python scripts/validateRsod.py --root "$CANONICAL_REL"
else
    run_env python scripts/validateAitod.py \
        --root "$CANONICAL_REL" \
        --source "$DATA_SOURCE" \
        --license-note "$LICENSE_NOTE"
fi

step '运行单元测试与正式环境检查'
run_env python -m pytest -q
run_env python scripts/checkEnv.py \
    --dataset "$DATASET" \
    --require-gpu \
    --strict-versions

AVAILABLE_GPUS="$(
    run_env python -c 'import torch; print(torch.cuda.device_count())' \
        | tail -n 1 \
        | tr -d '[:space:]'
)"
[[ "$AVAILABLE_GPUS" =~ ^[0-9]+$ ]] || fail '无法读取 CUDA GPU 数量'
(( AVAILABLE_GPUS >= GPUS )) \
    || fail "请求 $GPUS 张 GPU，但 PyTorch 只检测到 $AVAILABLE_GPUS 张"

if [[ "$MODE" == 'check' ]]; then
    echo '[完成] 环境、数据、测试、配置、模型和 GPU 验收通过。未启动训练。'
    exit 0
fi

step "训练 $DATASET 标准基线"
TRAIN_ARGS=(--work-dir "$WORK_DIR")
if [[ "$USE_AMP" -eq 1 ]]; then
    TRAIN_ARGS+=(--amp)
fi
if [[ "$RESUME" -eq 1 ]]; then
    TRAIN_ARGS+=(--resume)
fi
if [[ "$BATCH_SIZE" == '2' ]]; then
    TRAIN_ARGS+=(
        --cfg-options
        train_dataloader.batch_size=2
        train_dataloader.num_workers=2
        optim_wrapper.accumulative_counts=2
    )
fi

if [[ "$GPUS" -eq 1 ]]; then
    run_env python tools/train.py "$CONFIG" "${TRAIN_ARGS[@]}"
else
    echo "[提示] $GPUS 卡会改变全局 batch size；确认性比较必须保持训练合同一致。"
    run_env bash tools/distTrain.sh "$CONFIG" "$GPUS" \
        --master-port "$MASTER_PORT" "${TRAIN_ARGS[@]}"
fi

step '选择最佳 checkpoint 并运行正式评估'
CHECKPOINT="$(
    find "$WORK_DIR" -maxdepth 1 -type f -name 'best_*.pth' \
        -printf '%T@ %p\n' 2>/dev/null \
        | sort -nr \
        | head -n 1 \
        | cut -d' ' -f2-
)"
if [[ -z "$CHECKPOINT" ]]; then
    CHECKPOINT="$(
        find "$WORK_DIR" -maxdepth 1 -type f -name 'epoch_*.pth' \
            -printf '%T@ %p\n' 2>/dev/null \
            | sort -nr \
            | head -n 1 \
            | cut -d' ' -f2-
    )"
fi
[[ -n "$CHECKPOINT" && -f "$CHECKPOINT" ]] \
    || fail "训练结束但未在 $WORK_DIR 找到 checkpoint"

TEST_WORK_DIR="${WORK_DIR}_test"
PREDICTIONS="$WORK_DIR/predictions.bbox.json"
run_env python tools/test.py \
    "$CONFIG" \
    "$CHECKPOINT" \
    --work-dir "$TEST_WORK_DIR" \
    --out "$PREDICTIONS"

cat <<EOF

[完成] $DATASET 基线训练与评估完成。
Git commit : $REVISION
Checkpoint : $CHECKPOINT
预测文件   : $PREDICTIONS
测试目录   : $TEST_WORK_DIR

EOF

if [[ "$DATASET" == 'rsod' ]]; then
    echo '说明：RSOD 结果仅用于工程流程验收，不作为 SpecLoc 微小目标机制结论。'
else
    echo '下一步：人工确认 scene_id 或场景文件名规则后，再执行 Gate 1。'
fi
