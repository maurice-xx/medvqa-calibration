#!/bin/bash
cd "$(dirname "$0")/.."
while ! grep -q "FINISHED gemma4" logs/p0_gemma_retry.log 2>/dev/null; do sleep 120; done
$HOME/miniconda3/envs/merlin/bin/python code/rescore_analysis.py > logs/p1_rescore_analysis.log 2>&1
$HOME/miniconda3/envs/merlin/bin/python code/aprime_analysis.py > logs/p1_aprime.log 2>&1
$HOME/miniconda3/envs/merlin/bin/python code/format_cond_calib.py >> logs/p1_format_cond.log 2>&1
touch logs/GEMMA_P0_DONE
