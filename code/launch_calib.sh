#!/bin/bash
# 校准 pilot：qwen GPU0 → healthgpt GPU1（显存不足则排队 GPU0），完成后自动出指标
set -uo pipefail
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate merlin
cd "$(dirname "$0")/.."

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY=$HOME/miniconda3/envs/merlin/bin/python

mkdir -p logs results/raw results/summary data
rm -f logs/CALIB_DONE
OPTS="--seed data/seed100.jsonl"

CUDA_VISIBLE_DEVICES=0 $PY code/run_calib.py --model qwen $OPTS \
    --out results/raw/qwen.jsonl > logs/qwen.log 2>&1 &
P0=$!

FREE1=$($PY -c "import torch; f,t=torch.cuda.mem_get_info(1); print(int(f/2**30))" 2>/dev/null || echo 0)
if [ "$FREE1" -ge 17 ]; then
    echo "healthgpt on GPU1 (free ${FREE1} GiB)" > logs/calib_status.txt
    CUDA_VISIBLE_DEVICES=1 $PY code/run_calib.py --model healthgpt $OPTS \
        --out results/raw/healthgpt.jsonl > logs/healthgpt.log 2>&1 &
    P1=$!
    wait $P0; wait $P1
else
    echo "GPU1 insufficient (${FREE1} GiB); healthgpt queued on GPU0" > logs/calib_status.txt
    wait $P0
    CUDA_VISIBLE_DEVICES=0 $PY code/run_calib.py --model healthgpt $OPTS \
        --out results/raw/healthgpt.jsonl > logs/healthgpt.log 2>&1
fi

$PY code/calib_metrics.py --raw-dir results/raw --out results/summary >> logs/metrics.log 2>&1
touch logs/CALIB_DONE
echo "CALIB PILOT DONE"
