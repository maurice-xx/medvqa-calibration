#!/bin/bash
set -uo pipefail
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate merlin
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY=$HOME/miniconda3/envs/merlin/bin/python
PY5=$HOME/venvs/tf5/bin/python
mkdir -p logs results/rescore_actual
rm -f logs/ACTUAL_DONE
pick() { for i in 0 1 2; do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $i | tr -d ' '); f=$((24576-${u:-24576})); [ "$f" -ge 16384 ] && { echo $i; return; }; done; echo -1; }
run_model() {
  local m=$1 py=$2
  for a in 1 2 3; do
    local gpu=$(pick)
    while [ "$gpu" = "-1" ]; do sleep 120; gpu=$(pick); done
    echo "run $m gpu$gpu $(date '+%T')" >> logs/actual_status.txt
    CUDA_VISIBLE_DEVICES=$gpu $py code/rescore_actual.py --model $m > logs/actual_$m.log 2>&1
    grep -q "FINISHED $m" logs/actual_$m.log && return 0
    echo "retry $m $a" >> logs/actual_status.txt
    sleep 120
  done
  return 1
}
run_model qwen "$PY"
run_model healthgpt "$PY"
if [ -d /dev/shm/gemma4 ]; then export GEMMA_MODEL_PATH=/dev/shm/gemma4; fi
run_model gemma4 "$PY5"
touch logs/ACTUAL_DONE
echo "ALL DONE $(date '+%T')" >> logs/actual_status.txt
