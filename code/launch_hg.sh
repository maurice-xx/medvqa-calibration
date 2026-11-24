#!/bin/bash
cd "$(dirname "$0")/.."
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate merlin
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
CUDA_VISIBLE_DEVICES=0 $HOME/miniconda3/envs/merlin/bin/python code/run_calib.py --model healthgpt --seed data/seed100.jsonl --out results/raw/healthgpt.jsonl > logs/healthgpt.log 2>&1
while ! grep -q "FINISHED healthgpt" logs/healthgpt.log 2>/dev/null; do sleep 120; done
$HOME/miniconda3/envs/merlin/bin/python code/calib_metrics.py --raw-dir results/raw --out results/summary >> logs/metrics.log 2>&1
touch logs/CALIB_HG_DONE
