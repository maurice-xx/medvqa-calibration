# P0/P1 实验状态（自动更新）

## 目标
- P0-1: 3x3 反事实重评分（固定答案，换评分前缀）→ 分离生成效应 vs 评分效应
- P0-2: A' 控制条件（只加 Answer 标签，无观察）→ 分离标签效应与观察效应
- P0-3: 选择性风险（risk-coverage、高置信错误率、固定风险覆盖率）
- P1-1: 格式条件校准（global / transfer / independent / format-conditioned）+ few-shot 曲线
- P1-2: 数字与表述修正（9/12、置信度 +0.062、Gemma 措辞、power scaling）

## 关键中间发现（前 4 条记录）
同一答案 "CT scan" 在三种评分上下文下的 token 置信度：
- A 上下文: 0.9955
- B 上下文（中性观察占位）: 0.0047
- C 上下文: 0.0235
→ 评分上下文对置信度的影响达两个数量级；这为"校准是协议属性"提供因果证据，同时说明原迁移矩阵中相当部分来自测量上下文而非可靠性变化。

## 运行方式
- 驱动: code/run_all_p0p1.sh（flock 防重入；预热→加载→计算，三卡并行）
- 日志: logs/p0_{qwen,healthgpt,gemma}.log, logs/p1_*.log
- 输出: results/rescore/, results/raw_aprime/, results/rescore_analysis/, results/selective_risk/, results/aprime/, results/format_cond/
- 完成标记: logs/P0P1_DONE；最终报告: docs/P0P1_REPORT.md
