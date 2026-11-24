# MedVQA 跨格式校准研究系统：架构与实现说明

> 版本：v1.0（2026-11-03）
> 用途：交付外部审阅（GPT）以发现可改进点。因此本文档刻意保留实现细节、接口约定、失败模式与已知弱点，不做过度抽象。
> 读者：熟悉 Python / PyTorch / 实验方法学，但未接触过本项目的人。

---

## 1. 系统概述

### 1.1 研究问题

在生成式医学视觉问答（MedVQA）中，模型输出的**置信度**是否可靠，取决于两件事：

1. **生成协议**（answer format）：模型被要求怎样作答（直接短答 / 先观察再作答 / 附带自报置信度）；
2. **评分协议**（scoring context）：从哪段前缀下读取答案 token 的概率。

本系统要回答三个可检验问题：

- **Q1（现象）**：改变回答格式，准确率、置信度、ECE 是否系统性变化？强度是否随模型能力分层？
- **Q2（机制）**：这些变化中有多少来自"模型回答内容变了"（generation effect），多少来自"同一答案的读数变了"（scoring effect）？
- **Q3（后果与修复）**：跨协议复用校准阈值会造成多大的决策风险？逐格式校准能否修复？共享参数的"格式条件校准"是否更优？

### 1.2 系统目标

产出一套**可复现、可审计**的实验数据与分析产物，支撑一篇实证论文：

- 原始生成结果（每题 × 每格式 × 每模型一条记录）
- 反事实重评分矩阵（每题 × 3 答案来源 × 3 评分上下文）
- 统计推断（聚类 bootstrap CI、GEE logistic）
- 论文表 T1–T5 与图 F2–F6

### 1.3 规模

| 实验 | 规模 | 记录数 |
|---|---|---|
| E1 SLAKE 主实验 | 432 题 × 3 格式 × 3 模型 | 3,888 |
| E2 VQA-RAD | 150 × 3 × 2 | 900 |
| E3 VQA-Med2019 | 250 × 3 × 2 | 1,500（另有 1,500 条未使用变体）|
| E4 措辞鲁棒性 | 150 × 6 变体 × 2 | 1,800 |
| E5 A′ 控制 | 432 × 3 模型 | 1,296 |
| E6 3×3 重评分 | 432 × 9 × 3 模型 | 11,664 |
| E7/E8 派生分析 | CPU | — |

可引用记录合计 **22,548 条**（含 E3 的 3,000 条）。

---

## 2. 整体架构

### 2.1 三层架构

```
┌─────────────────────────────────────────────────────────────┐
│ L1 数据与协议层（只读、确定性）                                │
│  数据集 (SLAKE / VQA-RAD / VQA-Med)                          │
│    └─ 抽样脚本 → seed*.jsonl（固定 seed、图像级半区）           │
│  协议定义：格式 A/B/C/A′、评分前缀 sA/sB/sC、答案归一化规则      │
└───────────────────────────┬─────────────────────────────────┘
                            │ seed.jsonl + 模型路径
┌───────────────────────────▼─────────────────────────────────┐
│ L2 执行层（GPU、可重入、逐行落盘）                              │
│  run_calib.py       生成 + 置信度提取（主实验 E1–E4）           │
│  rescore_3x3.py     A′ 生成 + 3×3 teacher-forcing 重评分（E5/E6）│
│  调度脚本           预热→加载→运行→收尾，失败重试                 │
└───────────────────────────┬─────────────────────────────────┘
                            │ raw/*.jsonl、rescore/*.jsonl
┌───────────────────────────▼─────────────────────────────────┐
│ L3 分析层（CPU、纯函数式读取）                                  │
│  calib_metrics.py      逐格式指标 + 校准迁移矩阵                │
│  formal_stats.py       聚类 bootstrap CI                      │
│  gee_stats.py          GEE logistic + isotonic 对照            │
│  fit_sensitivity.py    校准器样本量敏感性                       │
│  rescore_analysis.py   生成/评分解耦 + 规范上下文 ECE            │
│  selective_risk.py     risk-coverage + 阈值迁移                │
│  aprime_analysis.py    标签效应 vs 观察效应                     │
│  format_cond_calib.py  四种校准策略 + few-shot                  │
│  make_figs.py / make_fig3.py / make_tables.py  出图出表          │
└─────────────────────────────────────────────────────────────┘
```

**关键分层原则**：L2 只负责"产生记录"，不做任何指标计算；L3 只读 L2 的 JSONL，不加载模型。这样任何指标口径调整都不需要重跑 GPU。

### 2.2 目录结构

```
./
├── code/                      # 16 个 py + 11 个 sh（其中 4 个为历史遗留调度脚本）
├── data/                      # 4 个 seed 文件（唯一数据入口）
│   ├── seed500.jsonl          # E1/E5/E6（432 题）
│   ├── seed_vqarad.jsonl      # E2（150 题）
│   ├── seed_vqamed.jsonl      # E3（250 题）
│   └── seed150_wording.jsonl  # E4（150 题，seed500 子集）
├── results/
│   ├── raw/                   # 主实验原始记录（仅 *500.jsonl）
│   ├── raw_vqarad/ raw_vqamed/ raw_wording/ raw_aprime/
│   ├── rescore/               # 3×3 矩阵（*model*.jsonl）
│   ├── summary/               # calib_metrics 产物
│   ├── summary_vqarad/ summary_vqamed/
│   ├── rescore_analysis/ selective_risk/ aprime/ format_cond/ sensitivity/ formal_stats/
│   ├── figures/
│   └── archive/{pilot,invalid}/   # 禁止引用
├── docs/                      # 台账、预注册、框架、表、运行报告
└── logs/                      # 运行日志 + 完成标记（*DONE）
```

### 2.3 模块清单

| 文件 | 职责 | 入口参数 | 输出 |
|---|---|---|---|
| `sample_data.py` | SLAKE 抽样（多 split、图像不相交）| `--out --n --splits --max-per-image --seed` | `seed500.jsonl` |
| `vqa_rad_sample.py` | VQA-RAD 抽样（100 yes-no + 50 短开放）| 无 | `seed_vqarad.jsonl` |
| `vqamed_sample.py` | VQA-Med 抽样（plane + modality）| 无 | `seed_vqamed.jsonl` |
| `run_calib.py` | 生成 + 置信度提取 | `--model --seed --out --limit --formats` | `raw/*.jsonl` |
| `rescore_3x3.py` | A′ 生成 + 3×3 重评分 | `--model --seed --limit --skip-aprime --skip-rescore --score-aprime --dry-run` | `raw_aprime/`、`rescore/` |
| `calib_metrics.py` | 指标 + 迁移矩阵 | `--raw-dir --out` | `summary.json/md` |
| `formal_stats.py` | 聚类 bootstrap | 无 | `formal_stats/bootstrap_stats.json` |
| `gee_stats.py` | GEE + isotonic | 无 | `gee_isotonic.json` |
| `fit_sensitivity.py` | 样本量敏感性 | 无 | `sensitivity/fit_sensitivity.json` |
| `rescore_analysis.py` | 解耦分析 | `--rescore-dir --raw-dir --out-dir` | `rescore_summary.json/md` |
| `selective_risk.py` | 风险-覆盖 + 阈值迁移 | `--raw-dir --out-dir` | `selective_risk.json/md/png` |
| `aprime_analysis.py` | 标签 vs 观察 | `--raw-dir --aprime-dir --out-dir` | `aprime_summary.json/md` |
| `format_cond_calib.py` | 四策略 + few-shot | `--raw-dir --out-dir` | `format_cond.json/md/png` |
| `make_figs.py` / `make_fig3.py` / `make_tables.py` | 出图出表 | 无 | `figures/`、`docs/tables/` |
| `run_all_p0p1.sh` | P0/P1 调度（flock 防重入、预热、重试）| — | `docs/P0P1_REPORT.md`、`logs/P0P1_DONE` |
| `gemma_shm_run.sh` | gemma 专用：选卡→等显存→运行→收尾 | — | `logs/GEMMA_P0_DONE` |

---

## 3. 核心模块职责

### 3.1 数据准备

**输入**：原始数据集 JSON/CSV + 图像目录
**输出**：`seed*.jsonl`，每行一道题

**SLAKE 过滤链**（`sample_data.py`）：

1. 仅英文（`q_lang == "en"`）
2. 仅四类可枚举题：`Abnormality / Organ / Modality / Position`
3. 排除治疗/病因/下一步等叙事题（正则 `EXCLUDE`）
4. 答案非空、不含逗号
5. `Abnormality` 仅收 CLOSED 且 gold ∈ {yes,no}
6. 其余类别仅收 OPEN；`Modality` 的 gold 限于 ct/mri/xray/ultrasound
7. 图像文件必须存在
8. `validate` 中与 `test` 图像重叠的题被剔除（防泄漏）
9. 配额：异常 40% / 器官 20% / 模态 20% / 位置 20%，每图最多 3 题

**实际产出**：432 题（过滤上限，未凑 500）、173 图。

**VQA-RAD**：只取 CLOSED(yes/no) 与 ≤3 词的 OPEN；100 + 50，116 图。
**VQA-Med**：只取 `plane`（125）与 `modality`（125）；abnormality 是疾病名、organ 是区域短语，均不可枚举，**明确排除**。

### 3.2 生成执行（`run_calib.py`）

**职责**：对 (题目 × 格式) 组合执行一次 greedy 生成，保存原始输出、解析答案、计算三种置信度、判定正确性。

**内部结构**：

```python
MODEL_PATHS = {qwen, healthgpt, gemma4}
FORMATS = {
  "A": {instr, marker=None},
  "B": {instr, marker="Answer:"},
  "C": {instr, marker="Answer:", end_marker="Confidence:"},
}
VARIANTS = {A2,A3,B2,B3,C2,C3}   # 措辞鲁棒性，运行时 FORMATS.update(VARIANTS)
```

**单条处理流程**：

```
build messages (system + image + text)
  → processor.apply_chat_template(tokenize=True, add_generation_prompt=True)
  → model.generate(greedy, max_new_tokens=128)
  → 解码 raw
  → parse_answer(raw, fmt)          # 有 marker 的格式取 marker 后文本
  → normalize(ans, category)        # 见下
  → correct = (ans_norm == gold_norm)
  → conf_token_geo  = exp(mean(log p(token_j)))    # 生成过程中累积
  → conf_token_first = p(第一个答案 token)
  → conf_verbalized  = 从 "Confidence:" 解析的 0–100
  → 写 JSONL（逐行 flush）
```

**`normalize(ans, category)` 规则**（这是全系统最"人工"的部分）：

| category | 规则 |
|---|---|
| `abnormality` | 去标点小写后必须 ∈ {yes,no} |
| `modality` | 关键词表映射到 {ct,mri,xray,ultrasound} |
| `vqamed_modality` | 前缀表映射（cta/ct angiography→ct，us-d/doppler→ultrasound，iv→contrast …）|
| `plane` | transverse→axial，其余原样 |
| `organ` / 其它 | 去标点小写、去冠词、截断 40 字符 |

**容错**：单条异常写入 `status="error"` 后继续；断点续跑按 `qid|format` 去重。

### 3.3 反事实重评分（`rescore_3x3.py`）

**职责**：把"格式效应"分解为生成效应与评分效应。

**3×3 设计**：

- 行 `g`：答案来自哪种生成格式（A/B/C 已生成的答案，**固定不变**）
- 列 `s`：在哪种评分前缀下读取答案 token 概率

```
c[g][s] = exp( mean_j log P( a_g,j | image, question, neutral_instr + prefix_s ) )
```

三种评分前缀（**中性指令固定，只变答案前缀**）：

| s | 前缀 |
|---|---|
| sA | `\n\n` |
| sB | `\n\nObservation: The image shows a medical scan.\nAnswer: ` |
| sC | `\n\nAnswer: ` |

**为什么用中性指令**：若沿用各格式自己的指令，B/C 的指令本身以 `Answer:` 结尾，再拼答案会造成"双标签"，使置信度退化到 1e-2 量级（v1 设计的实测失败）。中性指令保证三种上下文结构平行。

**A′ 控制条件**：只加 `Answer: <answer>` 标签，无观察、无自报，用于把 B 的准确率下降拆成"标签效应 + 观察效应"。

**边界处理**：答案清洗（去引号/句点、截断 80 字）、prompt 长度上限 4096 token、span 为空/下标越界/非有限 logprob 均记原因、OOM 清缓存重试一次、按 `qid|g|s` 断点续跑。

### 3.4 指标计算（`calib_metrics.py`）

对每个模型、每种格式计算：

- `acc`、`mean_conf`
- `ECE`（10 等宽分箱，`|acc_bin − conf_bin|` 加权）
- `Brier`、`AUROC(err)`（用 conf 检测错误，**翻转方向**报告）
- 自报置信度的解析数与 ECE（仅 C）
- **迁移矩阵**：T 在 fit 半区按 ECE 网格搜索；矩阵元素 `ECE(conf^(1/T_src) on tgt)`

**关键约定**：`conf^(1/T)` 是 **power scaling**，不是标准 temperature scaling（标准 TS 作用于 logits）。论文中统一称 power scaling。

### 3.5 统计推断

| 脚本 | 方法 | 关键点 |
|---|---|---|
| `formal_stats.py` | 图像聚类 bootstrap ×1000 | 重采样图像而非题目；迁移惩罚 = ECE_cal(src→tgt) − ECE_cal(tgt→tgt) |
| `gee_stats.py` | GEE logistic（cluster=image, exchangeable）| `correct ~ fmt_B + fmt_C`；同时跑 isotonic 对照 |
| `fit_sensitivity.py` | 拟合样本 10–200，重复 20 次 | 输出收敛曲线；暴露"小样本伤害已校准格式" |

### 3.6 派生分析

| 脚本 | 核心量 | 关键实现 |
|---|---|---|
| `rescore_analysis.py` | scoring spread（同答案换前缀）、generation spread（同前缀换答案）、规范上下文 ECE | 只保留 9 格齐全的题；CI 用图像聚类 bootstrap；rho 零方差保护 |
| `selective_risk.py` | risk@coverage、coverage@risk、阈值迁移 | A 上取 5% 风险阈值 → 套 B/C；每次 bootstrap 重估阈值；**不可达时返回 None** |
| `aprime_analysis.py` | 标签效应（A→A′）、观察效应（A′→B）| 复用 `run_calib.normalize`；A′ 无效记录单独计数 |
| `format_cond_calib.py` | global / cond-intercept / cond-full / independent | IRLS + L2；伪逆兜底；few-shot 空格式回退 global |

### 3.7 出图出表

- `make_figs.py`：F2（ECE 柱状 + CI）、F4（敏感性曲线）
- `make_fig3.py`：F3（三模型 3×3 ECE 热图）
- `make_tables.py`：T1–T5（Markdown + LaTeX booktabs），从 JSON 直出
- **T5 刻意按格式报 ECE，不报 pooled**：池化会让不同格式误差相互抵消（HealthGPT global pooled 0.058，但 B 实为 0.268）

### 3.8 调度与容错

`run_all_p0p1.sh`：

1. `flock` 文件锁防重复启动
2. 逐模型 `warm()`（`dd bs=8M` 预热权重）→ 启动 → 等 `model loaded` → 下一个
3. 每个模型运行失败重试 3 次（间隔 180s）
4. 全部结束后依次跑 4 个 CPU 分析，写 `docs/P0P1_REPORT.md`，touch `logs/P0P1_DONE`

`gemma_shm_run.sh`：在 `/dev/shm` 副本上运行 gemma；循环选卡（≥16 GiB 空闲）直到成功。

---

## 4. 模块间交互

### 4.1 主实验时序

```
sample_data.py ──► data/seed500.jsonl
                      │
                      ├─► run_calib.py --model qwen      ──► results/raw/qwen500.jsonl
                      ├─► run_calib.py --model healthgpt ──► results/raw/healthgpt500.jsonl
                      └─► run_calib.py --model gemma4    ──► results/raw/gemma500.jsonl
                                                                   │
                                          calib_metrics.py ────────┘
                                                 │
                                                 ▼
                                   results/summary/summary.json/.md
```

### 4.2 P0 时序

```
results/raw/*500.jsonl  （读答案 A/B/C）
        │
        ▼
rescore_3x3.py --model M
        ├─ A′ 生成            ──► results/raw_aprime/M.jsonl
        └─ 3×3 teacher-forcing ──► results/rescore/M.jsonl
                                          │
                                          ▼
                              rescore_analysis.py
                                          │
                                          ▼
                        results/rescore_analysis/rescore_summary.json/.md
```

### 4.3 P1 时序

```
summary.json ──► selective_risk.py  ──► selective_risk.json/md/png（T4、F5）
            ──► format_cond_calib.py ──► format_cond.json/md/png（T5、F6）
raw_aprime/ ──► aprime_analysis.py  ──► aprime_summary.json/md（§6.4）
全部产物 ────► make_tables.py ──────► docs/tables/T1–T5.md + tables.tex
```

### 4.4 接口契约

| 契约 | 内容 |
|---|---|
| 文件命名 | 主实验固定 `{model}500.jsonl`；gemma 模型名是 `gemma4` 但数据文件是 `gemma500.jsonl`（历史原因，脚本内用 `BASE_FILES` 映射）|
| 状态字段 | 所有 JSONL 记录含 `status ∈ {ok, error, empty_answer, fail}`；分析脚本只读 `ok` |
| 去重键 | 主实验 `qid\|format`；重评分 `qid\|g\|s`；A′ `qid` |
| 半区划分 | 全系统唯一规则：`md5(image_path) % 2`（0=fit，1=test）|
| 只读约束 | `results/archive/` 禁止被任何分析脚本读取 |
| 完成标记 | `logs/*_DONE`；调度脚本以此为收敛条件 |

---

## 5. 关键数据结构

### 5.1 seed 记录

```json
{
  "dataset": "slake", "split": "test",
  "qid": "slake:test:11934",
  "image_path": "/mnt/.../imgs/xmlab102",
  "question": "What modality is used to take this image?",
  "gold": "CT",
  "category": "modality",
  "content_type": "Modality",
  "answer_type": "OPEN"
}
```

**不变量**：`qid` 全局唯一；`image_path` 存在；同一图像可对应多题（≤3）。

### 5.2 主实验记录（`results/raw/*.jsonl`）

```json
{
  "model": "qwen",
  "qid": "slake:test:11934",
  "category": "modality",
  "gold": "CT",
  "question": "What modality is used to take this image?",
  "format": "A",
  "image_path": "/mnt/.../imgs/xmlab102",
  "raw": "CT scan",
  "ans_raw": "CT scan",
  "ans_norm": "ct",
  "gold_norm": "ct",
  "correct": true,
  "conf_token_geo": 0.9907,
  "conf_token_first": 0.9999,
  "span_len": 2,
  "conf_verbalized": null,
  "status": "ok",
  "error": null
}
```

**字段语义**：

- `conf_token_geo`：答案 span 的 token 概率几何均值（主置信度）
- `conf_token_first`：首 token 概率（辅助）
- `conf_verbalized`：仅 C 格式，解析自 `Confidence:`（0–1）
- `span_len`：答案 token 数；`0` 表示解析失败
- `status`：`ok` / `error`（附 `error` 字段）

**可选字段**：`base_format`（措辞变体记录中由 `fmt[0]` 得出）。

### 5.3 重评分记录（`results/rescore/*.jsonl`）

```json
{
  "model": "qwen", "qid": "slake:test:11934",
  "g": "A", "s": "B",
  "answer": "CT scan",
  "conf": 0.7997,
  "span": 2,
  "image_path": "...", "category": "modality",
  "status": "ok"
}
```

**不变量**：同一 `qid` 恰好 9 条（3×3）；`conf=null` 时必须有 `reason`。

### 5.4 A′ 记录

```json
{"model":"qwen","qid":"...","category":"modality","gold":"CT","question":"...",
 "format":"A_prime","raw":"Answer: CT scan","ans_raw":"CT scan",
 "image_path":"...","status":"ok"}
```

可选 `{model}_conf.jsonl`：`{"qid","conf","span","reason"}`（`--score-aprime` 生成）。

### 5.5 `summary.json`

```
{
  "<model>": {
    "n_records": int,
    "formats": {
      "A": {"format","n","n_invalid","acc","mean_conf","ece10","brier",
            "auroc_error_detection","verbalized"},
      ...
    },
    "transfer": {
      "temperatures": {"A","B","C"},
      "uncalibrated": {3×3},
      "temperature_scaled": {3×3},
      "n_test_per_format": {...}
    }
  }
}
```

### 5.6 分析产物

| 文件 | 顶层键 | 关键内容 |
|---|---|---|
| `bootstrap_stats.json` | `dataset:model` | `ci_ece`、`transfer_penalty_ci` |
| `rescore_summary.json` | model | `scoring_spread_mean/ci95`、`generation_spread_*`、`ece_diagonal_scoring`、`ece_canonical[s][g]`、`diag_vs_gen_pearson` |
| `selective_risk.json` | model | `threshold_from_A_at_5pct`、`threshold_reachable`、`apply_A_thr_to_{A,B,C}`（含 `coverage_ci95`/`risk_ci95`）|
| `format_cond.json` | model | `strategies.{global,cond_intercept,cond_full,independent}.ece_per_format`、`few_shot` |

---

## 6. 数据流转

### 6.1 端到端

```
原始数据集
  │ 过滤 + 配额 + 图像不相交
  ▼
seed*.jsonl（432/150/250/150 题）
  │ 生成（greedy）+ 置信度提取
  ▼
raw/*.jsonl（每题每格式一条）
  ├─► calib_metrics ──► summary.json ──► T2、F2、T5 的 raw 列
  ├─► formal_stats  ──► bootstrap_stats ──► T3/T4 的 CI
  ├─► gee_stats     ──► GEE 系数（论文正文）
  └─► rescore_3x3（固定答案，换前缀）
         ▼
      rescore/*.jsonl（每题 9 条）
         ├─► rescore_analysis  ──► T3、F3
         ├─► selective_risk    ──► T4、F5
         └─► aprime_analysis   ──► §6.4
```

### 6.2 单条记录的生命周期

1. 抽样脚本写入 `qid/question/image_path/gold/category`
2. `run_calib` 生成答案 → 解析 → 归一化 → 判定 → 计算三种置信度 → 落盘（含 `status`）
3. `rescore_3x3` 只读 `status=ok` 的 `ans_raw`，在三种前缀下重算置信度（不改答案、不改正确性）
4. 分析脚本按 `md5(image_path)%2` 拆半区，fit 半区拟合校准器，test 半区算 ECE
5. 出表脚本按模型/格式聚合，保留 CI

---

## 7. 依赖关系

### 7.1 运行时

| 项 | 版本/路径 |
|---|---|
| Python（主环境）| 3.10.0 @ `$HOME/miniconda3/envs/merlin/bin/python` |
| torch | 2.8.0+cu128 |
| transformers | 4.57.6（merlin） |
| statsmodels | 0.15.0 |
| matplotlib | 3.10.8 |
| Python（gemma）| `$HOME/venvs/tf5/bin/python`，transformers 5.20.1 |
| GPU | 3 × RTX 3090 24GB（共享）|
| 内存 | 48GB（共享，`/dev/shm` 63GB）|

### 7.2 模型与数据依赖

| 资源 | 路径 | 备注 |
|---|---|---|
| SLAKE | `assets/data/Slake/Slake1.0` | test/validate/imgs |
| VQA-RAD | `assets/data/VQA-RED` | JSON + Image Folder |
| VQA-Med2019 | `assets/data/VQAMed2019/.../VQAMed2019Test` | txt + images |
| Qwen3-VL-8B | `assets/Qwen3-VL-8B-Instruct` | merlin |
| HealthGPT-Pro-8B | `.../TianweiLin--HealthGPT-Pro-8B/...` | merlin |
| Gemma-4-E4B | `assets/models_cache/google/gemma-4-E4B-it` | tf5；可 `GEMMA_MODEL_PATH` 指向 `/dev/shm` 副本 |

### 7.3 环境隔离原因

- gemma 需要 transformers 5.x（`Gemma4Processor`）；merlin 的 4.57.6 会报 `Unrecognized processing class`
- 两个 venv 互不污染；脚本通过不同解释器路径切换，而不是运行时切包

### 7.4 硬件约束

- 单模型显存 ~15.5–16.5 GiB；24GB 卡可容一个模型 + 少量余量
- 容器内 `nvidia-smi` 不可靠，**必须**用 `torch.cuda.mem_get_info(0)` 判断真实空闲
- 共享 zpool：顺序读 20–200 MB/s，随机读可塌到 0.5 MB/s → 预热或 `/dev/shm`

---

## 8. 设计决策与权衡

| # | 决策 | 替代方案 | 理由 | 代价 |
|---|---|---|---|---|
| D1 | 只变回答格式，图像/问题/金标冻结 | 同时改题目或图像 | 保证可比性，差异只能归因于协议 | 无法研究题目-格式交互 |
| D2 | 图像级 md5 半区划分 | 随机按题划分 | 同图多题会泄漏；图像级划分更保守 | test 集略小（每格式 ~216）|
| D3 | 评分上下文用中性指令 + 平行前缀 | 沿用各格式指令 | 避免"双标签"导致置信度退化（v1 实测失败）| 对角只能"前缀匹配"（ρ≈0.42）|
| D4 | 报告对角相关而非声称完全复现 | 声称重评分复现生成 | 诚实标注局限 | 审稿人可能追问 |
| D5 | 主置信度用 span 几何均值 | 自报置信度 | 三种格式都有 token 概率；自报仅 C 有 | 与自报信号存在系统性差异 |
| D6 | 用 power scaling `c^(1/T)` | 标准 TS（logits 上） | 只有标量置信度，无 logits | 不能与 TS 文献直接数值对比 |
| D7 | 图像聚类 bootstrap | 题目级 bootstrap | 同图多题相关，题目级会低估方差 | 计算量略大 |
| D8 | GEE（population-averaged） | 混合效应 logistic | statsmodels 无稳定二项混合模型；GEE 聚类稳健 | 系数解释为群体平均而非个体 |
| D9 | T5 按格式报 ECE | 报 pooled ECE | 池化会抵消格式差异，得出相反结论 | 表格更长 |
| D10 | 逐行 flush + 断点续跑 | 批量写 | 中断不丢数据；可续跑 | I/O 略增 |
| D11 | 归档而非删除旧数据 | 直接删 | 审计与复盘 | 目录稍乱（已有 README 约束）|
| D12 | gemma 单独 venv + `/dev/shm` 副本 | 统一环境 | 版本冲突无解；I/O 瓶颈 | 运维复杂（需清理副本）|

---

## 9. 约束与已知风险

1. **样本量**：SLAKE 432 题是过滤上限；每格式 test 半区约 216 题，ECE 的 CI 宽约 ±0.03–0.05。
2. **正确性标签噪声**：开放题依赖别名归一化；`organ` 类同义词（liver/hepatic）会误判；VQA-Med modality 映射是手工表。
3. **VQA-Med 题型偏移**：只含 plane/modality（知觉题），B 格式效应在 Qwen 上消失——这是真实异质性，不是 bug。
4. **对角非严格复现**：评分上下文用中性指令，生成用格式指令；ρ=0.42/0.43 是设计代价。
5. **Gemma-4B 的"免疫"**：可能来自能力不足而非稳健，无法区分。
6. **单轮、无多图、无人类评估**。
7. **共享集群**：他人任务会抢占 GPU/IO；任何长任务都必须可重入、可重试。
8. **无单元测试**：目前靠 dry-run、断点续跑与产物核对；缺自动化回归。
9. **口径分散**：指标实现分布在 `calib_metrics` / `formal_stats` / 四个 P1 脚本中，ECE/半区逻辑存在重复实现（未抽公共库）。
10. **A′ 无置信度**：`--score-aprime` 已实现但未跑，A′ 目前只有准确率。

---

## 10. 可改进点 / 开放问题（给审阅者）

1. **公共库抽取**：`ece()`、`half()`、`normalize()`、`bootstrap()` 在 6 个脚本中重复；是否值得抽 `common/metrics.py`？（风险：改动会影响已产出的数字，需要回归验证）
2. **校准器族**：目前只有 power/logistic/isotonic；是否应加 beta calibration、vector scaling（多参数）？在 216 条 test 上是否过拟合？
3. **混合效应**：GEE 是群体平均；是否值得引入 `glmmTMB`/`lme4`（R）或 PyMC 做随机截距，给出 subject-level 结论？
4. **多重比较**：3 模型 × 6 迁移方向 × 2 数据集 = 36 个检验，目前只报 CI；是否应加 Holm/BH 校正？
5. **评分上下文的构造**：中性观察句是固定的；换 2–3 个占位句做敏感性分析，成本约 40 分钟 GPU，是否必要？
6. **对角一致性**：ρ≈0.42 偏低；能否通过"用生成时的实际前缀（含模型自己的观察）"重评分来提升？代价是矩阵不再是平行结构。
7. **正确性判定**：能否用一个小型 LLM judge 替代手工归一化（成本/一致性权衡）？
8. **选择性风险的决策语义**：5% 风险是任意选的；是否应给出 risk-coverage 曲线下的面积（AURC）作为单一指标？
9. **VQA-Med 变体数据**：`raw_vqamed/healthgpt.jsonl` 里有 1,500 条未分析的措辞变体，是否值得纳入第三数据集的鲁棒性分析？
10. **可复现性**：是否应把所有随机种子、模型 commit、数据集 md5 写入 `run_manifest.json`，并在每次运行时校验？
11. **脚本卫生**：`code/` 下有 4 个历史遗留调度脚本（`launch_calib.sh`、`launch_hg.sh`、`chain_gemma.sh`、`finish_gemma_retry.sh`）已不再使用，是否删除或移入 `archive/` 以免误用？

---

## 11. 附录

### A. 复现命令（最小集）

```bash
cd medvqa_calibration
source $HOME/miniconda3/etc/profile.d/conda.sh && conda activate merlin
PY=$HOME/miniconda3/envs/merlin/bin/python

# 1) 数据（已存在，可跳过）
$PY code/sample_data.py --out data/seed500.jsonl --n 500 --splits test,validate

# 2) 主实验（示例：qwen）
CUDA_VISIBLE_DEVICES=0 $PY code/run_calib.py --model qwen --seed data/seed500.jsonl \
  --out results/raw/qwen500.jsonl
$PY code/calib_metrics.py --raw-dir results/raw --out results/summary

# 3) P0
$PY code/rescore_3x3.py --model qwen --dry-run
CUDA_VISIBLE_DEVICES=0 $PY code/rescore_3x3.py --model qwen --skip-aprime
$PY code/rescore_analysis.py

# 4) P1
$PY code/selective_risk.py && $PY code/aprime_analysis.py && $PY code/format_cond_calib.py
$PY code/make_tables.py
```

### B. 术语

| 术语 | 含义 |
|---|---|
| format（生成协议）| A 直接短答 / B 观察+Answer / C 短答+自报 |
| scoring context（评分上下文）| sA/sB/sC 三种答案前缀 |
| generation effect | 同前缀、不同答案来源的置信度跨度 |
| scoring effect | 同答案、不同前缀的置信度跨度 |
| transfer penalty | ECE_cal(src→tgt) − ECE_cal(tgt→tgt) |
| compliance | B/C 输出中含 `Answer:` 标记的比例 |
| selective risk | 按置信度筛选后剩余样本的错误率 |

### C. 关键常量

| 常量 | 值 | 位置 |
|---|---|---|
| `MAX_PIXELS` | 448×448 | run_calib / rescore_3x3 |
| `MAX_NEW_TOKENS` | 128 | run_calib / rescore_3x3 |
| `MAX_ANSWER_CHARS` | 80 | rescore_3x3 |
| `MAX_PROMPT_TOKENS` | 4096 | rescore_3x3 |
| `MIN_FREE_GIB` | 15.5 | rescore_3x3 |
| ECE bins | 10 | 全部指标脚本 |
| bootstrap reps | 1000 | formal_stats / rescore_analysis / selective_risk |
| 半区 | `md5(image_path) % 2` | 全部 |
