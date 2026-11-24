#!/bin/bash
set -uo pipefail
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate merlin
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY=$HOME/miniconda3/envs/merlin/bin/python
mkdir -p results/raw_vqarad results/summary_vqarad logs
rm -f logs/VQARAD_DONE
OPTS="--seed data/seed_vqarad.jsonl"
CUDA_VISIBLE_DEVICES=0 $PY code/run_calib.py --model qwen $OPTS \
    --out results/raw_vqarad/qwen.jsonl > logs/vqarad_qwen.log 2>&1 &
P0=$!
while ! grep -q "model loaded" logs/vqarad_qwen.log 2>/dev/null; do sleep 15; done
sleep 10
FREE1=$($PY -c "import torch; f,t=torch.cuda.mem_get_info(1); print(int(f/2**30))" 2>/dev/null || echo 0)
if [ "$FREE1" -ge 17 ]; then
    echo "healthgpt on GPU1 (free ${FREE1} GiB)" > logs/vqarad_status.txt
    CUDA_VISIBLE_DEVICES=1 $PY code/run_calib.py --model healthgpt $OPTS \
        --out results/raw_vqarad/healthgpt.jsonl > logs/vqarad_healthgpt.log 2>&1 &
    P1=$!
else
    echo "GPU1 insufficient (${FREE1} GiB); healthgpt queued on GPU0" > logs/vqarad_status.txt
    wait $P0
    CUDA_VISIBLE_DEVICES=0 $PY code/run_calib.py --model healthgpt $OPTS \
        --out results/raw_vqarad/healthgpt.jsonl > logs/vqarad_healthgpt.log 2>&1
    P1=$!
fi
wait $P0; wait $P1
$PY code/calib_metrics.py --raw-dir results/raw_vqarad --out results/summary_vqarad >> logs/vqarad_metrics.log 2>&1
touch logs/VQARAD_DONE
echo "VQARAD DONE"
