# 预注册 v2：MedVQA 跨回答格式置信度校准（正式实验）

> 冻结：2026-10-22。基于 pilot（2 模型 × 3 格式 × 100 题）的确认结果制定。偏离须记录于 ITERATIONS。

## 1. 研究问题

Q1（现象）：同一医学 VLM、同一图像与问题，仅改变回答格式，置信度校准（ECE/Brier/AUROC）是否系统性变化？
Q2（缺口）：在格式 A 上拟合的 post-hoc 校准器（温度缩放），迁移到格式 B/C 后是否失效？迁移代价如何量化？
Q3（修复）：format-aware temperature scaling（每格式一个标量，拟合集 ≤50% 样本）能否恢复各格式校准？

## 2. 设计（冻结）

- 数据：SLAKE test+validate 英文可枚举题 432 题（filter 上限，不注水），seed=20261022，max-per-image=3；validate 与 test 图像不相交；按图像 md5 奇偶拆 fit/test 半区（同图多题不跨半区）
- 模型：Qwen3-VL-8B-Instruct、HealthGPT-Pro-8B（Gemma-4-E4B-it 与 VQA-RAD 作为外部验证扩展）
- 格式（冻结，不得更改措辞）：
  - A 直接短答
  - B 观察 + `Answer: <answer>`
  - C 短答 + `Answer:` + `Confidence: 0-100`
- 解码：greedy，max_new_tokens=128，max_pixels=448×448，HF offline
- 置信度定义（分开报告，不得混用）：span token 几何均值；span 首 token 概率；自报 0–100（仅 C）

## 3. 指标与主终点

- 每格式：acc、mean conf、ECE(10bin)、Brier、AUROC（错误检测，翻转报告）、格式合规率（输出含标记比例）
- **主终点：温度缩放迁移矩阵（3×3）的非对角线代价** Δ_transfer = ECE_cal(src→tgt) − ECE_cal(tgt→tgt)
- 次终点：自报置信度的 ECE 与 AUROC；合规率跨模型差异
- 统计：图像聚类 bootstrap 1000 次（pilot 用 10bin 点估计）；Holm 校正用于跨格式两两比较

## 4. 合规率处理规则（冻结）

- 合规 = 输出含 "Answer:" 标记（B/C）
- 不合规输出：主分析计入（acc 按解析结果判定）；附合规子集敏感性分析
- 合规率本身作为主要发现 R3 报告

## 5. 预期结果方向（假设，非承诺）

- H1：ECE 跨格式差异 ≥10pp（pilot 已示 0.27/0.41/0.37）
- H2：非对角线迁移代价 > 对角线重校准收益（pilot：A→B 失败、B↔C 成功）
- H3：合规率跨模型差异大（pilot：Qwen 100% vs HealthGPT B 45%）

## 6. 停止条件

- 迁移矩阵与 pilot 方向完全相反（迁移全部成功且代价 <1pp）→ 重新定位为"校准格式稳健性"短文
- 双模型结果方向矛盾且效应 <2pp → 止损

## 7. 外部验证（阶段 3）

- VQA-RAD（315 图，seed 图像不相交自划分）：仅复测主终点，不重新筛选条件
- Gemma-4-E4B：第三模型稳健性

## 8. 局限（论文中如实报告）

- 432 题为过滤上限；正确性判定含别名归一化噪声；单数据集主实验；MedGemma 类模型的潜在训练污染不适用于本地 Qwen/HealthGPT，但 SLAKE 为公开基准的泄漏风险如实声明
