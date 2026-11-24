#!/bin/bash
cd "$(dirname "$0")/.."
while ! grep -q "FINISHED gemma4" logs/gemma_retry.log 2>/dev/null; do sleep 120; done
$HOME/miniconda3/envs/merlin/bin/python code/calib_metrics.py --raw-dir results/raw --out results/summary >> logs/metrics.log 2>&1
touch logs/GEMMA_500_DONE
