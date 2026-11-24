#!/bin/bash
set -uo pipefail
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate merlin
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY=$HOME/miniconda3/envs/merlin/bin/python
PY5=$HOME/venvs/tf5/bin/python

while [ ! -f logs/CALIB_500_DONE ]; do sleep 300; done

PICK_GPU() {
$PY -c "
import torch
best, bf = -1, 0
for i in range(3):
    f,t = torch.cuda.mem_get_info(i)
    if f > bf: best, bf = i, f
print(best if bf/2**30 >= 16 else -1)" 2>/dev/null || echo -1
}

BEST=$(PICK_GPU)
for i in $(seq 1 72); do
  [ "$BEST" != "-1" ] && break
  sleep 300
  BEST=$(PICK_GPU)
done
if [ "$BEST" = "-1" ]; then echo "no GPU free after 6h" > logs/gemma500_status.txt; exit 1; fi
echo "gemma on GPU$BEST" > logs/gemma500_status.txt

CUDA_VISIBLE_DEVICES=$BEST $PY5 code/run_calib.py --model gemma4 --seed data/seed500.jsonl \
    --out results/raw/gemma500.jsonl > logs/gemma500.log 2>&1
while ! grep -q "FINISHED gemma4" logs/gemma500.log 2>/dev/null; do sleep 120; done
$PY code/calib_metrics.py --raw-dir results/raw --out results/summary >> logs/metrics.log 2>&1
touch logs/GEMMA_500_DONE
