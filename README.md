# Format-Bound Calibration: Answer Format and Scoring Context in Medical VQA

Code and derived results for a study of how answer format and scoring context change the
confidence assigned to a *fixed* answer in medical visual question answering.

The central distinction is between two protocol choices that are usually varied together:

- **generation format** — what the model is asked to produce (direct answer, observation-then-answer,
  answer-plus-confidence);
- **scoring context** — the assistant-side prefix under which the answer tokens are scored.

Each saved answer is rescored under all three prefixes while its text and correctness label are held
fixed, giving a 3 × 3 matrix per model. A separate pass teacher-forces the same saved text under the
original generation prompt.

## Repository layout

```
code/           experiment, scoring, calibration and analysis scripts
data/           question seeds (SLAKE, VQA-RAD, VQA-Med 2019) as JSONL
docs/           protocol notes, experiment ledger, tables (T1-T5, tables.tex)
research/       novelty audit
results/figures/  paper figures (PNG)
```

## Environment

Python 3.11 with PyTorch 2.8 (CUDA 12.8) and Transformers 4.57. Gemma-4 requires a newer
Transformers than the other two models; the launch scripts assume two environments, referenced as
`$HOME/miniconda3/envs/merlin` and `$HOME/venvs/tf5`.

```bash
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate merlin
```

## Data and model assets

Datasets and weights are **not** redistributed here. Place them under `assets/` (or edit the paths in
`code/README.md` and the `launch_*.sh` scripts):

| Asset | Expected path |
| --- | --- |
| SLAKE | `assets/data/Slake/Slake1.0` |
| VQA-RAD | `assets/data/VQA-RED` |
| VQA-Med 2019 | `assets/data/VQAMed2019` |
| Qwen3-VL-8B-Instruct | `assets/Qwen3-VL-8B-Instruct` |
| HealthGPT-Pro-8B | `assets/.runtime/modelscope/.../HealthGPT-Pro-8B` |
| Gemma-4-E4B-it | `assets/models_cache/google/gemma-4-E4B-it` |

SLAKE is available at <https://www.med-vqa.com/slake/>, VQA-RAD at <https://osf.io/89kps/>, and
VQA-Med 2019 through the ImageCLEF/CEUR working notes.

## Reproducing the pipeline

Run everything from the repository root.

```bash
# 1) build a question seed
python code/sample_data.py --out data/seed100.jsonl --n 100

# 2) smoke test (6 items, one GPU)
CUDA_VISIBLE_DEVICES=0 python code/run_calib.py --model qwen \
    --seed data/seed100.jsonl --out results/raw/smoke.jsonl --limit 6

# 3) full generation run (staggered across GPUs)
bash code/launch_calib.sh

# 4) teacher-forced rescoring of the saved outputs
python code/fixed_output_tf.py

# 5) 3x3 crossed rescoring
python code/rescore_3x3.py

# 6) metrics, calibration transfer, selective prediction
python code/calib_metrics.py --raw-dir results/raw --out results/summary
python code/formal_analysis.py
python code/formal_stats.py

# 7) paper tables and figures
python code/make_tables.py
python code/make_paper_figs.py
```

## Conventions

Confidence definitions are kept separate and never mixed:

- `conf_token_geo` — geometric mean of answer-token probabilities within the extracted span;
- `conf_token_first` — probability of the first answer token;
- `conf_verbalized` — the model's own 0–100 report, available only in format C.

Splits are image-disjoint: images are assigned by MD5 parity so that several questions about one
image never straddle the fit and test halves. Calibrators are fitted on the fit half only.

## Citation

```bibtex
@inproceedings{liu2026formatbound,
  title     = {Format-Bound Calibration: Answer Format and Scoring Context in Medical VQA},
  author    = {Liu, Zhuohui and Zhao, Long},
  booktitle = {Proceedings of the International Conference on Computer Supported Cooperative Work in Design (CSCWD)},
  year      = {2026}
}
```

## Contact

Zhuohui Liu, Long Zhao — Qilu University of Technology (Shandong Academy of Sciences).
Correspondence: zhaolong@qlu.edu.cn
