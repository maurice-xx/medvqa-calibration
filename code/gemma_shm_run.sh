#!/bin/bash
set -uo pipefail
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate merlin
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export GEMMA_MODEL_PATH=/dev/shm/gemma4
PY=$HOME/miniconda3/envs/merlin/bin/python
PY5=$HOME/venvs/tf5/bin/python
rm -f logs/GEMMA_P0_DONE
echo "wait $(date '+%T')" > logs/gemma_shm_status.txt

pick_gpu() {
  local best=-1 bf=0 f i
  for i in 0 1 2; do
    local u
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $i | tr -d ' ')
    f=$(( 24576 - ${u:-24576} ))
    if [ "$f" -gt "$bf" ]; then bf=$f; best=$i; fi
  done
  if [ "$bf" -ge 16384 ]; then echo "$best"; else echo "-1"; fi
}

GPU=$(pick_gpu)
for i in $(seq 1 240); do
  [ "$GPU" != "-1" ] && break
  sleep 60
  GPU=$(pick_gpu)
done
if [ "$GPU" = "-1" ]; then echo "no gpu" >> logs/gemma_shm_status.txt; exit 1; fi
echo "gpu=$GPU $(date '+%T')" >> logs/gemma_shm_status.txt

CUDA_VISIBLE_DEVICES=$GPU $PY5 code/rescore_3x3.py --model gemma4 --skip-aprime > logs/p0_gemma_shm.log 2>&1
while ! grep -q "FINISHED gemma4" logs/p0_gemma_shm.log 2>/dev/null; do sleep 60; done
$PY code/rescore_analysis.py > logs/p1_rescore_analysis.log 2>&1
$PY code/aprime_analysis.py > logs/p1_aprime.log 2>&1
$PY code/format_cond_calib.py > logs/p1_format_cond.log 2>&1
touch logs/GEMMA_P0_DONE
echo "done $(date '+%T')" >> logs/gemma_shm_status.txt
