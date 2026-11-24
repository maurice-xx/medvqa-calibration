# 实验台账（唯一权威版本，2026-11-03）

> 目的：区分「论文可引用」「被取代」「作废」三类数据，避免写作时混用。

## 0. 时间线

| 阶段 | 日期 | 内容 | 状态 |
|---|---|---|---|
| 试点 | 10-21 | SLAKE validate 100 题 × 3 格式 × 2 模型 | 被 E1 取代 |
| 主实验 | 10-21~22 | SLAKE 432 题 × 3 格式 × 3 模型 | **有效（主结果）** |
| 外部验证 | 10-22 | VQA-RAD 150 题 × 3 格式 × 2 模型 | **有效** |
| 第三数据集 | 10-22 | VQA-Med 250 题 × 3 格式 × 2 模型 | **有效（注意变体混入）** |
| 措辞鲁棒性 | 10-25 | 150 题 × 6 变体 × 2 模型 | **有效** |
| P0 重评分 | 11-02~03 | 432 题 × 3 答案 × 3 前缀 × 3 模型 | **有效（v2 设计）** |
| P0 A' 控制 | 11-02~03 | 432 题 × 3 模型 | **有效** |
| P1 派生分析 | 11-03 | 选择性风险 / 格式条件校准 | **有效** |

## 1. 论文可引用的实验（E1–E8）

| ID | 目的 | 数据/模型 | 记录数 | 文件 | 论文位置 |
|---|---|---|---|---|---|
| E1 | 主现象：格式改变 acc/conf/ECE | SLAKE 432 题 × A/B/C × Qwen/HealthGPT/Gemma | 3,888 | `results/raw/{qwen500,healthgpt500,gemma500}.jsonl` | T2, F2 |
| E2 | 外部验证 | VQA-RAD 150 题 × A/B/C × Qwen/HealthGPT | 900 | `results/raw_vqarad/*.jsonl` | T2 延伸 |
| E3 | 第三数据集 | VQA-Med 250 题（plane+modality）× A/B/C × 2 模型 | 1,500（见注） | `results/raw_vqamed/*.jsonl` | 异质性小节 |
| E4 | 措辞鲁棒性 | SLAKE 150 题 × 6 变体（A2/A3/B2/B3/C2/C3）× 2 模型 | 1,800 | `results/raw_wording/*.jsonl` | 5.5 |
| E5 | A' 控制（标签 vs 观察） | SLAKE 432 题 × 3 模型 | 1,296 | `results/raw_aprime/*.jsonl` | 5.4 |
| E6 | 3×3 反事实重评分 | 432 题 × 3 答案 × 3 前缀 × 3 模型 | 11,664 | `results/rescore/*.jsonl` | 5.1 机制 |
| E7 | 选择性风险 | E1 派生（conf/correct） | CPU | `results/selective_risk/selective_risk.json`, `fig5_risk_coverage.png` | 5.3 后果 |
| E8 | 格式条件校准 + few-shot | E1 派生 | CPU | `results/format_cond/format_cond.json`, `fig6_fewshot_calibration.png` | 5.6 修复 |

**E3 注**：`raw_vqamed/healthgpt.jsonl` 原为 2,250 条（A/B/C 750 + 6 个措辞变体 1,500）。**2026-11-03 已删除 1,500 条未使用变体**，现为 750 条；重算汇总与原值完全一致。

## 2. 被取代（不要引用）

| 文件 | 记录数 | 原因 |
|---|---|---|
| ~~`pilot/qwen_100q.jsonl`、`healthgpt_100q.jsonl`~~ | 300 + 300 | 试点（validate 100 题），被 432 题主实验取代。**2026-11-03 已删除**（不再保留审计副本）|

## 3. 作废（仅保留作审计，绝不引用）

| 文件/版本 | 问题 |
|---|---|
| ~~`qwen_buggy_span.jsonl`~~ | span 提取缺少 break，B/C 置信度全空。**2026-11-03 已删除** |
| 重评分 v1（已删除） | `build_text` 漏传图像 → "tokens: 0, features: 1" 全失败 |
| 重评分 v1b（已删除） | B/C 指令以 `Answer:` 结尾又追加标签 → 双标签，置信度退化至 1e-2 以下 |
| gemma 首轮（失败） | 文件名拼接 bug（gemma4+500→gemma4500） |

## 4. 写作引用规则（硬约束）

1. 主表只用 `*500.jsonl` 且 `format ∈ {A,B,C}`；试点文件不得混入。
2. 重评分分析只用 `results/rescore/*.jsonl`（字段 `g`=答案来源格式，`s`=评分前缀），并注明"前缀匹配、非完全复现"（对角相关 0.42/0.43）。
3. 选择性风险与格式条件校准必须从 E1 重新生成，不能手工抄数。
4. 任何新实验先写 run_id + config 快照，再落盘。

## 5. 当前唯一"未完成"

- 无。E6 已完成：11,664 条（qwen/healthgpt/gemma 各 3,888）。

## 6. 目录卫生（2026-11-03）

- `results/raw/` 现仅含 `qwen500.jsonl`、`healthgpt500.jsonl`、`gemma500.jsonl`（汇总脚本 glob 不再受污染）
- `results/archive/` **已于 2026-11-03 整体删除**（试点与作废数据不再保留）
- `results/raw_vqamed/` 现为 750 条 A/B/C（已移除 1,500 条未使用变体）
- `results/summary/summary.json` 仅含三个 `*500` 模型
