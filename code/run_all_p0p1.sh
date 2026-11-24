#!/bin/bash
set -uo pipefail
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate merlin
cd "$(dirname "$0")/.."
exec 9>logs/p0p1.lock
flock -n 9 || { echo "another driver running"; exit 1; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY=$HOME/miniconda3/envs/merlin/bin/python
PY5=$HOME/venvs/tf5/bin/python
QDIR=assets/Qwen3-VL-8B-Instruct
HDIR=assets/.runtime/modelscope/TianweiLin--HealthGPT-Pro-8B/TianweiLin/HealthGPT-Pro-8B
GDIR=assets/models_cache/google/gemma-4-E4B-it
mkdir -p logs results/rescore results/raw_aprime results/rescore_analysis results/selective_risk results/aprime results/format_cond
rm -f logs/P0P1_DONE
: > logs/p0p1_status.txt

warm() {
  local dir=$1 name=$2 s e
  s=$(date +%s)
  for f in "$dir"/*.safetensors; do dd if="$f" of=/dev/null bs=8M 2>/dev/null; done
  e=$(date +%s)
  echo "warm $name $((e-s))s" >> logs/p0p1_status.txt
}

wait_loaded() {
  local logf=$1 pid=$2 i
  for i in $(seq 1 60); do
    grep -q "model loaded" "$logf" 2>/dev/null && return 0
    kill -0 "$pid" 2>/dev/null || return 1
    sleep 15
  done
  return 1
}

launch() {
  local model=$1 gpu=$2 py=$3 dir=$4 tag=$5 var=$6 pid
  warm "$dir" "$model"
  CUDA_VISIBLE_DEVICES=$gpu $py code/rescore_3x3.py --model "$model" --skip-aprime > "logs/p0_$tag.log" 2>&1 &
  pid=$!
  eval "$var=$pid"
  wait_loaded "logs/p0_$tag.log" "$pid"
  echo "launched $model gpu$gpu pid=$pid $(date '+%T')" >> logs/p0p1_status.txt
}

launch qwen 0 "$PY" "$QDIR" qwen QPID
launch healthgpt 1 "$PY" "$HDIR" healthgpt HPID
launch gemma4 2 "$PY5" "$GDIR" gemma GPID

wait $QPID; echo "qwen rc=$? $(date '+%T')" >> logs/p0p1_status.txt
wait $HPID; echo "healthgpt rc=$? $(date '+%T')" >> logs/p0p1_status.txt
wait $GPID; echo "gemma rc=$? $(date '+%T')" >> logs/p0p1_status.txt

$PY code/rescore_analysis.py > logs/p1_rescore_analysis.log 2>&1
echo "rescore analysis $(date '+%T')" >> logs/p0p1_status.txt
$PY code/selective_risk.py > logs/p1_selective_risk.log 2>&1
echo "selective risk $(date '+%T')" >> logs/p0p1_status.txt
$PY code/aprime_analysis.py > logs/p1_aprime.log 2>&1
echo "aprime $(date '+%T')" >> logs/p0p1_status.txt
$PY code/format_cond_calib.py > logs/p1_format_cond.log 2>&1
echo "format-cond $(date '+%T')" >> logs/p0p1_status.txt

{
  echo "# P0/P1 运行报告"
  echo
  echo "## 状态"
  cat logs/p0p1_status.txt
  echo
  echo "## 重评分"
  tail -40 logs/p1_rescore_analysis.log 2>/dev/null
  echo
  echo "## 选择性风险"
  tail -60 logs/p1_selective_risk.log 2>/dev/null
  echo
  echo "## A' 控制"
  tail -10 logs/p1_aprime.log 2>/dev/null
  echo
  echo "## 格式条件校准"
  tail -20 logs/p1_format_cond.log 2>/dev/null
} > docs/P0P1_REPORT.md

touch logs/P0P1_DONE
echo "ALL DONE $(date '+%T')" >> logs/p0p1_status.txt
