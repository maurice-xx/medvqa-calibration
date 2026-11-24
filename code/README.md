# MedVQA 跨回答格式置信度校准（阶段 1b pilot）

研究问题：同一模型、同一图像、同一问题，仅改变回答格式（A 直接短答 / B 观察+Answer / C 短答+自报置信度）时，置信度校准是否系统性变化？在格式 A 上拟合的温度缩放参数迁移到 B/C 是否失效？

## 环境

```bash
source $HOME/miniconda3/etc/profile.d/conda.sh && conda activate merlin
cd medvqa_calibration
```

- torch 2.8.0+cu128 / transformers 4.57.6 / 3×RTX 3090（共享）
- 数据：SLAKE validate 英文可枚举题 100 题（seed=20261021），路径只读引用 `assets/data/Slake/Slake1.0/`
- 模型（只读引用）：qwen=`assets/Qwen3-VL-8B-Instruct`；healthgpt=`assets/.runtime/modelscope/TianweiLin--HealthGPT-Pro-8B/TianweiLin/HealthGPT-Pro-8B`

## 真实运行命令

```bash
# 1) 抽样
python code/sample_data.py --out data/seed100.jsonl --n 100

# 2) 冒烟（6 条）
CUDA_VISIBLE_DEVICES=0 python code/run_calib.py --model qwen --seed data/seed100.jsonl \
    --out results/raw/smoke.jsonl --limit 6

# 3) 全量 pilot（错峰：qwen GPU0 → healthgpt GPU1，显存不足自动排队）
bash code/launch_calib.sh

# 4) 指标与迁移矩阵
python code/calib_metrics.py --raw-dir results/raw --out results/summary
```

## 置信度定义（互不混用）

- `conf_token_geo`：生成答案 span 内 token logprob 几何均值（exp(mean logp)）
- `conf_token_first`：答案 span 首 token 概率
- `conf_verbalized`：仅格式 C，模型自报 0–100 → [0,1]
- 温度缩放：conf' = conf^(1/T)，T 在 fit 半区按 ECE 最小化网格搜索（0.05–20，×1.15）

## 数据划分与防泄漏

- 100 题来自 SLAKE validate；按图像 md5 奇偶拆 fit/test 半区（同图多题不跨半区）
- 校准器只在 fit 半区拟合，test 半区只评估

## 结果位置

- `results/raw/`：逐条记录（含逐 token 置信度、自报置信度、正确性）
- `results/summary/summary.json` + `summary.md`：每格式指标 + 3×3 迁移矩阵
- `docs/PHASE1B_REPORT.md`：结论
