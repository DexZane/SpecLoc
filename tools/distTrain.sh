#!/usr/bin/env bash
# Distributed training launcher.
#
# Usage:
#   bash tools/distTrain.sh <config> <num_gpus> [training arguments]

set -euo pipefail

usage() {
    echo "Usage: bash tools/distTrain.sh <config> <num_gpus> [training arguments]"
    exit 1
}

[[ $# -ge 2 ]] || usage

CONFIG=$1
GPUS=$2
shift 2

PORT=29500
TRAIN_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --master-port|--master_port)
            PORT=$2
            shift 2
            ;;
        *)
            TRAIN_ARGS+=("$1")
            shift
            ;;
    esac
done

[[ -f "$CONFIG" ]] || { echo "Error: config not found: $CONFIG"; exit 1; }

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "Config : $CONFIG"
echo "GPUs   : $GPUS"
echo "Port   : $PORT"

if command -v torchrun &>/dev/null; then
    LAUNCHER=(torchrun --nproc_per_node="$GPUS" --master_port="$PORT")
else
    LAUNCHER=(python -m torch.distributed.launch --nproc_per_node="$GPUS" --master_port="$PORT")
fi

"${LAUNCHER[@]}" tools/train.py "$CONFIG" --launcher pytorch "${TRAIN_ARGS[@]}"
