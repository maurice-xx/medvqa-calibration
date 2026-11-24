#!/bin/bash
set -uo pipefail
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate merlin
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY=$HOME/miniconda3/envs/merlin/bin/python
PY5=$HOME/venvs/tf5/bin/python
G=assets/models_cache/google/gemma-4-E4B-it
rm -f logs/GEMMA_P0_DONE
echo "warm start $(date '+%T')" > logs/gemma_warm_status.txt
S=$(date +%s)
for f in "$G"/*.safetensors; do dd if="$f" of=/dev/null bs=8M 2>/dev/null; done
E=$(date +%s)
echo "warm done $((E-S))s $(date '+%T')" >> logs/gemma_warm_status.txt

CUDA_VISIBLE_DEVICES=0 $PY5 code/rescore_3x3.py --model gemma4 --skip-aprime > logs/p0_gemma_retry.log 2>&1
while ! grep -q "FINISHED gemma4" logs/p0_gemma_retry.log 2>/dev/null; do sleep 60; done
$PY code/rescore_analysis.py > logs/p1_rescore_analysis.log 2>&1
$PY code/aprime_analysis.py > logs/p1_aprime.log 2>&1
$PY code/format_cond_calib.py > logs/p1_format_cond.log 2>&1
touch logs/GEMMA_P0_DONE
echo "all done $(date '+%T')" >> logs/gemma_warm_status.txt
