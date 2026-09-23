# WINO-DLLM 方法结果总表

更新时间：2026-09-23。所有数值均为百分比。HumanEval 和 MBPP 使用 Pass@1；Sudoku 按现有 evaluator 使用 **cell accuracy**；其余数据集使用样本级 Accuracy。`—` 表示当前没有完整、可核验的结果文件。主表沿用原始 benchmark extractor 的官方口径；文末另列 GSM8K 答案提取审计。

## Accuracy / Pass@1

| Method | GSM8K | MATH-500 | HumanEval | MBPP | Countdown | Sudoku† | ARC-E | ARC-C | AVG (8)‡ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LLaDA（论文） | 73.24 | 32.00 | 37.80 | 36.40 | 24.21 | 14.23 | 59.13 | 51.87 | 41.11 |
| WINO（论文） | 75.82 | 34.20 | 42.07 | 36.40 | 33.20 | **15.20** | 81.19 | 73.89 | 49.00 |
| WINO Remask (local matched) | 77.56 | 34.40 | 42.68 | 36.20 | — | — | — | — | — |
| WINO-No-Same (post-hoc ablation) | 78.92 | — | — | — | — | — | — | — | — |
| WINO-No-Diff (post-hoc ablation) | 78.17 | — | — | — | — | — | — | — | — |
| Soft Revision V1 | 78.32 | 34.00 | 42.07 | 36.40 | 30.86 | 13.48 | **81.65** | 75.92 | 49.09 |
| Soft Revision V1 (same-run control) | **79.15** | — | — | — | — | — | — | — | — |
| Soft Revision V2 | 78.32 | — | — | — | — | — | — | — | — |
| Soft Revision V3 (renewable) | 77.33 | — | — | — | — | — | — | — | — |
| Soft Revision V4 | 78.09 | — | — | — | — | — | — | — | — |
| Soft Revision V5 | 77.79 | — | — | — | — | — | — | — | — |
| Soft Revision V6 | 78.85 | 34.60 | 43.90 | 36.00 | 35.94 | 13.45 | 81.61 | 74.58 | **49.87** |
| Factorized Soft Revision | 76.50 | — | — | — | — | — | — | — | — |
| Contextualized Sequence Slot | 77.03 | — | — | — | — | — | — | — | — |
| Factorized Logical Slot | 78.77 | — | — | — | — | — | — | — | — |
| Explicit Factorized Soft | 78.47 | — | — | — | — | — | — | — | — |
| V6-Pure-MASK | 78.09 | — | — | — | — | — | — | — | — |
| V6-Posterior-Only | 77.79 | — | — | — | — | — | — | — | — |
| V6-Old+MASK | 78.24 | — | — | — | — | — | — | — | — |
| Residual Soft Revision | 77.56 | — | — | — | — | — | — | — | — |
| Feature-wise Residual Soft | 77.18 | — | — | — | — | — | — | — | — |
| Temporal Revision Memory | 78.24 | — | — | — | — | — | — | — | — |
| Soft De-Anchor | 77.79 | — | — | — | — | — | — | — | — |
| Dual-State Soft Revision | 77.41 | — | — | — | — | — | — | — | — |
| Revision Register | 77.41 | — | — | — | — | — | — | — | — |
| Contrastive Soft Revision | 78.17 | — | — | — | — | — | — | — | — |
| Geometry-Aware Soft Revision | 78.24 | — | — | — | — | — | — | — | — |
| Revision-Aware Soft Adapter | 78.24 | — | — | — | — | — | — | — | — |
| Soft Attention Attenuation | 78.77 | 34.40 | **44.51** | 35.80 | — | — | — | — | — |
| LWSR-3 | 77.94 | — | 43.90 | — | — | — | — | — | — |
| Three-State Current | 74.75 | — | — | — | — | — | — | — | — |
| Progressive Commitment | 60.73 | — | — | — | — | — | — | — | — |
| Selective Counterfactual Revision V7 | 62.24 | — | — | — | — | — | — | — | — |
| CCSE-V6 | 78.39 | **35.20** | — | 34.80 | **38.28** | — | — | — | — |
| Soft Trial | 78.85 | 34.80 | 43.90 | 35.80 | 33.98 | 13.33 | — | **76.25** | — |
| Counterfactual Revision | 78.77 | 34.20 | 43.90 | 36.00 | 32.42 | **13.73** | — | — | — |
| Proximal Soft Revision (lambda=0.5) | 78.54 | 34.00 | 43.90 | 35.60 | — | 13.53 | — | — | — |
| Fixed Soft Interpolation (alpha=0.25) | 79.00 | 34.60 | 43.90 | 35.80 | 35.16 | 13.40 | 81.52 | 73.91 | 49.66 |
| Fixed Soft Interpolation (alpha=0.5) | 79.08 | 34.80 | 42.07 | **36.60** | 34.38 | 13.40 | 81.14 | 73.24 | 49.34 |
| Fixed Soft Interpolation (alpha=0.75) | 79.00 | 34.60 | 43.90 | 36.00 | 33.20 | 13.43 | 81.40 | 74.25 | 49.47 |
| Adaptive Soft Commitment | 78.92 | 34.60 | 43.90 | 35.80 | 33.59 | 13.53 | 81.10 | 73.91 | 49.42 |
| Hard Anchor Prior (lambda=0.5) | 77.26 | 33.00 | 43.29 | 35.40 | 33.20 | — | — | — | — |
| Hard Anchor Prior (lambda=0.7) | 78.09 | 33.80 | 43.90 | 36.20 | 32.03 | — | — | — | — |
| Hard Anchor Prior (lambda=0.9) | 77.41 | 34.00 | 43.29 | 36.00 | 36.33 | — | — | — | — |
| Soft Branching (k=2) | 78.85 | — | — | — | — | — | — | — | — |
| Soft Branching (k=4) | 78.92 | — | — | — | — | — | — | — | — |

† Sudoku 的 cell accuracy 与整题准确率不是同一指标。V6 和 Adaptive 均完整解对 7/500 个 puzzle，即 exact-puzzle accuracy 为 1.40%，但两者 cell accuracy 略有不同。

‡ `AVG (8)` 是八列百分数的等权宏平均，且仅在八个数据集全部完成时计算。因为其中混合了 Sudoku cell accuracy 与其他任务的样本级 Accuracy/Pass@1，该平均值适合做统一汇总，不应解释为单一概率指标。

CCSE-V6 只评估了 GSM8K、MATH-500、MBPP、Countdown，不能填 `AVG (8)`。这四列的等权宏平均为 **46.67%**，匹配 V6 为 **46.35%**（+0.32 个百分点）。Countdown 本地评估集为 **256** 题；CCSE 正确 98/256。

粗体是当前每列已经完成结果中的最高值；并列值均加粗。GSM8K 的 same-run V1 control 单独保留，因为它与原始 V1 运行结果不同（79.15% vs 78.32%）。

Layerwise Revision、Revision Risk Budget Stage 1、candidate-level Revision Memory、Momentum Soft Refinement、KV Soft Revision、Layer-to-Layer、各类 oracle/causal/probe 属于诊断或停止方向，没有产生新的完整 benchmark decoder，因此不作为 Accuracy 方法行。只有 checkpoint 的实验不使用未完成样本计算 Accuracy。`Temporal Revision Memory` 是 residual-memory 完整 decoder，与前述 candidate-memory diagnostic 不是同一实验。

## 最近实验的完成状态

| Direction | 已完成范围 | 主要结果 | 决定 |
|---|---|---|---|
| Fixed Soft Interpolation | 3 个 alpha × 8 个数据集 | alpha=0.25/0.5/0.75 的 8-task AVG 分别为 49.66/49.34/49.47 | 完成；没有单一 alpha 跨任务占优 |
| Context-Coupled Soft Evolution (CCSE-V6) | GSM8K/MATH-500/MBPP/Countdown 全集 | 78.39/35.20/34.80/38.28；4-task AVG 46.67，比 V6 +0.32 | 主实验完成；MASK-CCSE 对照未启动，实验进程已暂停 |
| Adaptive Soft Commitment | 8 个数据集 | AVG 49.42；平均 alpha 大多接近上界，整体接近 V6 | 完成；没有超过 V6 的跨任务结果 |
| Hard Anchor Prior | 3 个 lambda × GSM8K/MATH/HumanEval/MBPP/Countdown | lambda=0.9 在 Countdown 为 36.33，但其余任务没有一致收益 | 完成预定范围；不支持通用改进 |
| Soft Branching Revision | GSM8K，k=2/4 | k=2 为 78.85%，k=4 为 78.92% | 完成 Stage 1；收益很弱，暂不扩展 benchmark |
| Contrastive Soft Revision | GSM8K 全集 | 1031/1319，78.17%，Avg NFE 42.49；vs V6 为 42/51（net -9） | 完成；去除 native self-conditioning 后整体回归 |
| Geometry-Aware Soft Revision | GSM8K 全集 | 1032/1319，78.24%，Avg NFE 42.59；vs V6 为 30/38（net -8） | 完成；球面几何没有优于原始 V6 mixing |
| Revision-Aware Soft Adapter | GSM8K train 轻量训练 + test 全集 | 1032/1319，78.24%，Avg NFE 42.83；vs V6 为 69/77（net -8） | 完成；训练后改变大量轨迹，但未提高准确率 |
| 三个新变体的跨任务扩展 | MATH-500 部分 checkpoint | Contrastive/Geometry/Adapter 分别停在 88/73/73 条，当前无存活进程 | 未完成；不报告 partial Accuracy |
| V6 Soft 机制消融 | Pure-MASK / Posterior-Only / Old+MASK × GSM8K 全集 | 78.09/77.79/78.24，均低于 V6 | 完成；decommitment、MASK-heavy、posterior semantics 三者组合最好 |
| Factorized Soft / Sequence Slot / Logical Slot / Explicit Factorized | 四种机制均完成 GSM8K 全集 | 官方准确率 76.50/77.03/78.77/78.47；Logical/Explicit 答案审计后 79.00/78.54 | 分开 Soft 成分或额外 slot 均未显示稳健优于 V6；见下文同口径修正 |
| Residual / Feature-wise / Residual Memory | GSM8K 全集 | 77.56/77.18/78.24；对应 net -17/-22/-8 | 完成；局部或历史 residual 没有超过完整 V6 Soft |
| De-Anchor / Dual-State / Revision Register | GSM8K 全集 | 77.79/77.41/77.41；对应 net -14/-19/-19 | 完成；把 revision 信息移到 attention side channel 明显弱于直接 Soft 替换 |
| Soft Attention Attenuation | GSM8K/MATH-500/HumanEval/MBPP 全集 | 78.77/34.40/44.51/35.80；4-task macro 48.37，V6 为 48.34 | 完成；宏平均几乎相同，任务间不一致 |
| LWSR-3 | GSM8K、HumanEval 全集；MBPP 仅 196 条 | GSM8K 77.94（net -12），HumanEval 43.90（0/0） | 已暂停；扩大为三 token 修复区没有收益 |
| Three-State Current | GSM8K 全集 | 986/1319，74.75%，Avg NFE 39.20 | 完成；更省 NFE，但 accuracy 显著下降 |
| Progressive Commitment | GSM8K 全集 | 801/1319，60.73%，Avg NFE 61.33；vs V6 net -239 | 完成；无 shadow 的对称 H→S 过度撤销 Hard |
| Selective Counterfactual Revision V7 | GSM8K 全集 | 821/1319，62.24%，Avg NFE 72.74；vs V6 net -219 | 完成；lookahead score 未对齐最终正确性且代价高 |
| WINO SAME/DIFF post-hoc ablation | GSM8K 全集，No-Same/No-Diff | 78.92% / 78.17%；仅抑制严格对齐的 7.77% / 14.38% 目标事件 | 完成；属于部分事件干预，不能解释成删除全部 SAME/DIFF |
| Momentum Soft Refinement | 4-sample smoke | V6 中 SOFT lifetime 恒为 1，没有任何可执行的 S->S refinement | smoke 后停止 |
| Layerwise Revision | GSM8K/MATH/HumanEval/MBPP probe | 最好 pooled AUC 0.603，低于 D_temp 的 0.627 | probe 后停止，不实现 gate |
| Revision Risk Budget | 4-task offline Stage 1 | cumulative D_temp pooled AUC 0.621，且与修正次数 Spearman=0.928 | Stage 1 后停止，不实现 budget |
| Revision Memory | 24-sample smoke | 60 次机会中仅改变 1 次候选，最终输出改变 0/24 | smoke 后停止 |
| KV Soft Revision | architecture inspection | 当前 LLaDA MDM 不支持跨轮 KV cache，真实 K_b/K_soft 需要额外分支计算 | 按停止条件终止，未改 decoder |

## 实验目的与结论索引

下面按机制归纳已经实际运行的实验。这里的“完成”只表示完成了各自预定范围；probe、smoke 和事件级 replay 不能与完整 benchmark Accuracy 混为一谈。

### 主解码与 Soft 表示方法

| 实验 | 想验证什么 | 已观察到的结论 |
|---|---|---|
| WINO Remask | Hard token 被 verifier 怀疑后先回 MASK，再重新生成，是否能提高并行解码质量 | 相对 Vanilla 有明显收益，但 GSM8K 中 91.06% 的 `H→M→H` 回到同一 identity，且重复循环负担很高 |
| Soft Revision V1 | 把 WINO 的 destructive `H→MASK` 改成 one-round `H→SOFT` | 相对本地 WINO 提高 Accuracy，并减少 NFE；证明“暂态 Soft 替代完全擦除”有效 |
| V2 | V1 的 Soft 恢复后是否应直接使用当前 top-1 | 与 V1 同为 78.32%，没有额外收益 |
| V3 renewable | 是否允许 Soft 跨多轮持续 `S→S` | GSM8K 降至 77.33%；持续 Soft 比 one-round transient Soft 更不稳 |
| V4 | 显式区分 `S→H(old/new)/MASK` 的完整仲裁 | GSM8K 78.09%，改善部分轨迹但仍低于 V6 |
| V5 | 限制 identity correction、偏向恢复旧 identity | GSM8K 77.79%；只保守回旧 token 不能解释 V6 的收益 |
| V6 | one-round Soft 后允许恢复旧 token、接受新 identity 或回 MASK | 当前完整 8-task 最优，AVG 49.87；核心优点是减少同 identity 重复修订，同时保留少量真正 identity correction |
| Proximal Soft | `H(a)→S` 时把 old one-hot 与 V6 posterior 混合，减小表示移动 | 多任务没有稳定收益；保留旧假设也会保留错误惯性 |
| Fixed Soft Interpolation | 已接受 `H(b)` 后再保留一部分先前 Soft 表示 | 部分 GSM8K/MATH/MBPP 有小收益，但 HumanEval/Countdown/ARC 回归；没有通用 alpha |
| Adaptive Soft Commitment | 用归一化 entropy 自动决定上述插值强度 | `H/log(V)` 动态范围太小，alpha 几乎总在 0.86–0.88；8-task AVG 低于 V6 |
| Hard Anchor Prior | 在被拒绝 Hard 进入 Soft 时持续混入旧 token | 大多回归；旧 identity 已被 verifier 怀疑，继续锚定常传播 stale context |
| Momentum Soft Refinement | 连续 Soft 轮之间累积 posterior | V6 的 Soft lifetime 恒为 1，没有 `S→S` 可供 momentum 生效，smoke 后停止 |
| Soft Branching k=2/4 | 用 top-k revision posterior 保留多个候选，避免立即坍缩 | 输出改变很多，但 GSM8K 仅与 V6 持平或 +1 个样本；缺少判别方向 |
| Counterfactual Revision | 在 identity change 时比较 OLD/NEW/SOFT 三条短期分支的全局后果 | 多任务没有稳定超过 V6，额外分支计算未换来可靠选择优势 |
| Soft Trial | 新 identity 先保持一轮 trial Soft，再由 V6 仲裁 | 个别任务变化，但 8-task 不完整且已完成任务没有一致净收益 |
| V6-Pure-MASK | 保留 V6 lifecycle，只把 revision-round 输入换成 MASK | 78.09%，低于 V6；temporary decommitment 有用但单独不足 |
| V6-Posterior-Only | 去掉最外层 0.5 MASK mixing，仅保留 posterior Soft | 77.79%，三项机制消融中最差；V6 的 MASK-heavy mixing 有贡献 |
| V6-Old+MASK | 只保留 old token + MASK，不注入 alternative posterior semantics | 78.24%，仍低于 V6；alternative semantics 有小但真实的组合贡献 |
| Residual Soft | `E_old + ρ(E_v6-E_old)`，按 old-token shadow support 决定修订幅度 | 77.56%，vs V6 net -17；动态缩小整体 revision 会丢失有效重构 |
| Feature-wise Residual | 只更新 `abs(E_v6-E_old)` 较大的 embedding 维度 | 77.18%，vs V6 net -22；embedding 维度差异不是可解释的独立“错误 feature” |
| Temporal Revision Memory | 跨 diffusion step 累积 direction-consistent residual | 78.24%，NFE 42.40，但 vs V6 net -8；复用 residual 未提高正确性 |
| Contrastive Soft | 用 `log p_shadow-log p_native` 强调去除 self-conditioning 后的新语义 | 72.55% revision posterior top-1 被改写，GSM8K 78.17%；过度校正 |
| Geometry-Aware Soft | 保持 posterior，改用归一化方向聚合与 SLERP | 与 V6 Soft cosine 0.992，但 norm 被重标定；GSM8K 78.24%，无几何收益 |
| Revision-Aware Adapter | 冻结 backbone，训练小 adapter 让 revised position 匹配 shadow teacher | 改变大量轨迹但 GSM8K 78.24%；局部 distillation 目标未对齐最终任务正确性 |

### 上下文传播、状态机与 side-channel

| 实验 | 想验证什么 | 已观察到的结论 |
|---|---|---|
| CCSE-V6 | 新 `H→S` 后先额外 forward，让 Soft 信息传播，再做本轮其他 commit | 决策确实大量改变；4-task AVG +0.32，但 GSM8K/MBPP 回归且物理 forward 明显增加，方向不稳定 |
| LWSR-3 | identity change 时同时 soften 相邻三 token，修复局部区域 | GSM8K 77.94（net -12），HumanEval 与 V6 完全相同；固定局部窗口没有收益 |
| Soft Attention Attenuation | Soft 可读全局，但其他 token 对 Soft key 的 attention 减半 | 四任务 macro 48.37 vs V6 48.34，几乎相同；只有 HumanEval +1 个样本，非一致改进 |
| Soft De-Anchor | 不替换 Hard identity，只降低其他 query 对被怀疑位置的 attention | GSM8K 77.79（net -14）；单纯降低 contextual authority 不如 V6 Soft |
| Dual-State | Hard identity 走主 residual，shadow belief 仅作为 K/V side-channel | GSM8K 77.41（net -19）；把 identity 与 belief 拆开没有保留 V6 优势 |
| Revision Register | Hard identity 不变，把 revision residual 放入少量额外 register slots | GSM8K 77.41（net -19）；独立 register 没有形成有效修订通路 |
| Three-State Current | 把 M/S/H 变成统一状态机，并让 MASK 未提交位置先进入 Soft | NFE 降到 39.20，但 GSM8K 仅 74.75%；更激进的 Soft 化损伤 Accuracy |
| Progressive Commitment | 完全去 shadow，用 temporal/self disagreement 驱动对称 M/S/H 转换 | GSM8K 60.73、NFE 61.33；正确 Hard 被大量撤销，是明确负结果 |
| Selective Counterfactual Revision V7 | disagreement 只提 pivot，再用多候选 downstream entropy lookahead 决定 `H→S` | GSM8K 62.24、NFE 72.74；score 与最终正确性错位，且 verification 代价高 |
| V6 Counterfactual Masked Verifier | 唯一替换 V6 的 shadow H→S verifier：逐 pivot 临时 MASK 检查 | 20-sample sanity 14/20，但平均 27.05 个额外 forward、93.8s/sample；未运行 full，说明朴素 leave-one-out 成本不可接受 |
| Layer-to-Layer Soft | 在同一 forward 的 25/50/75% 层读取 posterior 并注入衰减 Soft residual | 20-sample probe 中间 logits 几乎不支持 candidate，Soft 与 MASK cosine >0.9997；按停止条件未跑 full |
| KV Soft Revision | 在 KV memory 内插值 Soft/Hard | LLaDA 是双向全序列重算、没有跨轮 AR KV cache；若真实构造分支 KV 会违反无额外 forward 约束，未实施 |

### 诊断、oracle 与因果分析

| 实验 | 想回答什么 | 主要结论 |
|---|---|---|
| WINO same-token return | 91.06% 是一次性 remask，还是反复回到同一假设 | 48.96% same-return 位置至少循环两次；57.61% remask compute 重访已返回的同 identity 位置，错误样本负担更高但不能据此判 token 对错 |
| WINO vs V6 revision analysis | Soft 是否真正缩短 WINO 的同 identity 循环；identity change 是否必要 | 四任务上 V6 same-ID events/sample 大幅下降；首个 identity change 的 NEW-only/OLD-only 都约 0–2%，多数分支同对或同错 |
| SAME/DIFF post-hoc ablation | WINO 的收益主要来自 91% SAME 还是少量 DIFF | 严格可对齐子集里 No-Same/No-Diff 都未降 Accuracy，但覆盖率仅 7.77%/14.38%，不能外推为删除全部事件 |
| Same-ID causal control | 最终 identity 固定为同一个 `a` 时，Hard/Mask/Soft 中间态如何改变轨迹 | 24 个事件中 Mask 与 Soft 最终输出 24/24 相同，最符合 `Mask≈Soft>Hard` 的暂时重新条件化，而非 Soft 独有语义收益 |
| Candidate Oracle | V6 top-1 失败时，同一 shadow posterior 的 rank 2–4 是否存在可救分支 | eligible regressions 中 alternative rescue 14、OLD rescue 13；GSM8K oracle headroom +0.76pp，但 MATH/HE 很弱，说明候选选择只解释部分失败 |
| Layerwise Revision Probe | helpful/harmful candidate 是否以不同方式跨层出现 | 最佳 pooled AUC 0.603，低于 D_temp 0.627；不实现 layer gate |
| Temporal / Risk Budget Probe | 累计 D_temp 是否比单事件风险更可靠 | cumulative AUC 0.621，且与 correction count 相关 0.928；没有独立、稳定信号，不实现 budget |
| Candidate Revision Memory | 历史候选 bank 是否改进下一次 candidate selection | 重复位置很多，但 identity oscillation 仅 0.26–2.03%；smoke 60 次仅改 1 次候选且 0/24 输出变化，停止 |
| Soft Readability | pretrained LLaDA 如何读取人工 Soft state | Soft 可读但更接近 MASK；Soft→Hard 路径常非单调；shadow 在 identity correction 上仍提供不同的 leave-one-out 信息 |
| Context-Guided Latent Revision | 用 `P_read→P_shadow` 梯度是否能找到有效连续修订方向 | 梯度非退化但能量下降不对应 recovery，且每次 backward 约 1.21× forward；Stage 1 停止 |
| Denoising Consistency | NEW revision 是否让其他已知 token 更容易被 mask-denoise 重建 | pooled sample AUC 0.546，任务方向不一致，低于 D_temp；停止 |
| Stable Anchor Verification | NEW revision 是否破坏此前稳定 Hard anchors | pooled AUC 0.456–0.477，信号方向不一致；停止 |
| Revision Frontier Probe | OLD/NEW counterfactual flip 是否预测其他 Hard token 后续被 reopen | frontier 很稀疏（11 positions），但 later-reopen enrichment 37.59×；说明能发现内部不稳定性 |
| Transactional Frontier Intervention | 提前 reopen frontier 是否改善最终答案 | 9 个事件中 0 recovery/0 regression，随机对照同样无 correctness 变化；NO-GO，不扩 full |
| Soft Attention Matrix Analysis | attenuation 的 recovery/regression 是否有可复现 attention 模式 | attenuation 确实削弱 Soft→context 传播，但 recovered/regressed 变化相近；Soft/shadow attention 也无判别模式 |
| V6 Cross-Benchmark Error Analysis | V6 新错误主要来自哪里 | 主要瓶颈是局部 revision 缺少最终任务约束方向；MATH/code 还受基础能力限制，Countdown/ARC/Sudoku 有显式全局约束和输出协议错误 |

## GSM8K：V6 后续方法的统一比较

这张表只放完整 1319 题结果，并使用同一历史 V6 做 paired comparison。`Recovered/Regressed` 均是相对 V6；`—` 表示现有汇总没有保存可直接核验的 paired 数值。

| Method | Correct / 1319 | Accuracy | Avg NFE | Recovered | Regressed | Net |
|---|---:|---:|---:|---:|---:|---:|
| Soft Revision V6 | 1040 | 78.85 | 42.60 | — | — | — |
| Factorized Logical Slot | 1039 | 78.77 | 42.44 | 49 | 50 | -1 |
| Explicit Factorized Soft | 1035 | 78.47 | 42.56 | 53 | 58 | -5 |
| Contextualized Sequence Slot | 1016 | 77.03 | 43.40 | 47 | 71 | -24 |
| Factorized Soft Revision | 1009 | 76.50 | 41.16 | 41 | 72 | -31 |
| Fixed Soft Interpolation alpha=0.5 | **1043** | **79.08** | 42.57 | 8 | 5 | +3 |
| Soft Branching k=4 | 1041 | 78.92 | 42.57 | 6 | 5 | +1 |
| Soft Attention Attenuation | 1039 | 78.77 | 42.56 | 42 | 43 | -1 |
| V6-Old+MASK | 1032 | 78.24 | 42.73 | 44 | 52 | -8 |
| Temporal Revision Memory | 1032 | 78.24 | **42.40** | 31 | 39 | -8 |
| Geometry-Aware Soft | 1032 | 78.24 | 42.59 | 30 | 38 | -8 |
| Revision-Aware Adapter | 1032 | 78.24 | 42.83 | 69 | 77 | -8 |
| Contrastive Soft | 1031 | 78.17 | 42.49 | 42 | 51 | -9 |
| V6-Pure-MASK | 1030 | 78.09 | 42.49 | 37 | 47 | -10 |
| LWSR-3 | 1028 | 77.94 | 42.59 | 12 | 24 | -12 |
| V6-Posterior-Only | 1026 | 77.79 | 42.71 | 46 | 60 | -14 |
| Soft De-Anchor | 1026 | 77.79 | 42.93 | 45 | 59 | -14 |
| Residual Soft | 1023 | 77.56 | 42.87 | 45 | 62 | -17 |
| Dual-State Soft | 1021 | 77.41 | 43.10 | 45 | 64 | -19 |
| Revision Register | 1021 | 77.41 | 43.10 | 45 | 64 | -19 |
| Feature-wise Residual | 1018 | 77.18 | 42.93 | 45 | 67 | -22 |
| Three-State Current | 986 | 74.75 | 39.20 | — | — | — |
| Selective Counterfactual Revision V7 | 821 | 62.24 | 72.74 | 58 | 277 | -219 |
| Progressive Commitment | 801 | 60.73 | 61.33 | 51 | 290 | -239 |

这组统一比较给出三个较清楚的结论。第一，V6 的优势不是任何单一成分：Pure-MASK、Posterior-Only、Old+MASK 均下降，说明 temporary decommitment、MASK-heavy geometry 和 shadow posterior semantics 需要一起工作。第二，围绕 old Hard 做 residual、feature gate、memory 或 side-channel 都会削弱 V6；被 verifier 拒绝后，保留旧表示常等价于保留 stale context。第三，目前只有 fixed alpha=0.5 和 branch k=4 在 GSM8K 略高于 V6，但前者跨任务不稳定，后者只多 1 个正确样本，因此还没有新的稳健方法超过 V6。

### Factorized 实验与答案提取审计

| 方法 | 官方正确数 | 审计后正确数 | 审计后准确率 | 官方 paired vs V6 |
|---|---:|---:|---:|---:|
| WINO | 1023 | 1024 | 77.63% | — |
| V6 | 1040 | 1041 | 78.92% | — |
| Factorized Logical Slot | 1039 | 1042 | 79.00% | 49 recovered / 50 regressed |
| Explicit Factorized Soft | 1035 | 1036 | 78.54% | 53 recovered / 58 regressed |

官方 extractor 将样本 1001 最终答案 `2:00 pm` 误读为 `00`；这四个方法都被同样漏判。Logical Slot 另有样本 839、868 的推理过程 `\boxed{}` 被误当成最终答案。审计只更改明确的答案提取错误，保留原始输出和官方分数；因为样本 1001 在四个方法中同时由错改对，Explicit 与 V6 的 paired 净值仍为 **−5**，Logical Slot 与 V6 的审计后净值仍为 **+1**。后者只差 1 题，不足以证明稳定优于 V6。详见 [逐样本审计](explicit_factorized_soft/answer_extraction_audit.json)；可发布的 per-sample 输出和摘要位于各实验目录。

## CCSE-V6 四任务结果

| Dataset | CCSE correct / total | V6 Acc | CCSE Acc | Δ Acc | V6 Avg NFE | CCSE physical Avg NFE | Recovered / regressed vs V6 |
|---|---:|---:|---:|---:|---:|---:|---:|
| GSM8K | 1034/1319 | 78.85 | 78.39 | −0.45 | 42.60 | 59.30 | 73 / 79（net −6） |
| MATH-500 | 176/500 | 34.60 | 35.20 | +0.60 | 70.63 | 85.11 | 22 / 19（net +3） |
| MBPP | 174/500 | 36.00 | 34.80 | −1.20 | 94.89 | 106.73 | 2 / 8（net −6） |
| Countdown | 98/256 | 35.94 | 38.28 | +2.34 | 88.54 | 128.73 | 47 / 41（net +6） |

新增 SOFT 状态经过 coupling forward 后确实会改变后续决定：四任务共记录 63,987 次下游 draft decision change。但 paired net 仅在 MATH-500 和 Countdown 为正，GSM8K 与 MBPP 反而回归；额外 coupling forward 平均为 15.57/13.31/10.83/34.52 次/样本。因此 +0.32 点的四任务宏平均不足以说明同步策略跨任务稳定改善 accuracy–NFE tradeoff。[完整结果与机制统计](/root/lx/WINO-DLLM/LLaDA/results/context_coupled_soft_evolution/report_context_coupled_soft_evolution.md) 已单独保存。按照原实验条件，MASK-CCSE 对照符合启动条件，但目前按暂停要求没有启动。

## 表示层实验分析

### Fixed Soft Interpolation

固定插值是这组方向里唯一在多个 benchmark 上产生正向 paired net recovery 的方法。alpha=0.5 相对 V6 在 GSM8K 为 8 recovered / 5 regressed（+3），MATH-500 为 1/0（+1），MBPP 为 7/4（+3）。但它在 HumanEval 为 4/7（-3）、Countdown 为 9/13（-4）、ARC-E 为 19/30（-11）、ARC-C 为 0/4（-4）。这说明保留 Soft 表示确实能改变并偶尔改善轨迹，但效果具有明显任务依赖性，不能解释为“更软的提交普遍更安全”。

三组 alpha 的 8-task AVG 都低于 V6 的 49.87：alpha=0.25 最接近，为 49.66；alpha=0.5 为 49.34；alpha=0.75 为 49.47。alpha=0.5 在早期四个 reasoning/code benchmark 上显得最好，是因为后续 Countdown 和 ARC 结果尚未纳入时形成的局部结论。

### Adaptive Soft Commitment

Entropy-aware alpha 没有复现固定 alpha=0.5 的优势。GSM8K 中 adaptive alpha 的均值为 0.876，范围约 0.769–0.900；词表大小归一化把大多数 posterior entropy 压成了很小的数，因此公式几乎总选择接近 Hard 的提交。最终结果也更接近 V6：GSM8K 仅 +1 net recovery，MATH/HumanEval 与 V6 相同，MBPP -1，Countdown -6，ARC-E -12，ARC-C -2。

这否定的是当前的 `H(P)/log(V)` 映射，而不是所有自适应强度方案。该归一化缺少有效动态范围，不能根据事件不确定性充分拉开 alpha。

### Hard Anchor Prior

Hard Anchor 对轨迹的干预比 commit-only interpolation 更早：旧 token 先验被混入 SOFT 输入，因而会同时改变后续候选形成与仲裁轨迹。结果整体不稳定。lambda=0.5 在 GSM8K 相对 V6 为 46 recovered / 67 regressed，lambda=0.7 为 48/58；MATH 及多数 MBPP 设置也没有可靠净收益。lambda=0.9 在 Countdown 得到 36.33%，比 V6 高 0.39 个百分点，但增加约 0.59 NFE，且该收益没有跨任务复现。

因此旧 Hard 信息在某些长算术轨迹里可能有用，但持续把它注入 SOFT 状态容易形成错误惯性。现有结果更支持在“提交后的短暂表示”上保留不确定性，而不是在候选形成阶段持续锚定旧 token。

### Momentum Soft Refinement

这个方向在当前 V6 生命周期下没有实验自由度。所有 SOFT 位置都在下一次正常 forward 立即转成 H(a)、H(b) 或 MASK，平均 lifetime 为 1；smoke 中连续 SOFT 状态对为 0。因此 lambda=0.5/0.7/0.9 均严格复现 V6。要让 momentum 生效必须先加入 S->S 或延迟仲裁，那会成为另一种 lifecycle 实验。

### Soft Branching Revision

GSM8K 上，k=2 得到 1040/1319（78.85%），相对 V6 为 6 recovered / 6 regressed；k=4 得到 1041/1319（78.92%），为 6/5，净提升 1。两者 Avg NFE 分别为 42.51 和 42.57，与 V6 的 42.60 基本相同。虽然 k=2/k=4 分别改变了 174/160 个样本的生成输出，但正确性变化几乎抵消。

因此“避免立即坍缩为单个候选”会显著扰动轨迹，却没有显示出对应的判别收益。k=4 的 +0.08 个百分点只有 1 个样本，且低于固定 alpha=0.5 的 79.08%。这个结果没有严格触发“只增加 NFE”的停止条件，但效应量太小，目前没有足够依据直接扩展到其他 benchmark。

### Contrastive / Geometry / Adapter 三个新变体

三个实验都在完整 GSM8K 1319 题上完成，且 20-sample smoke 中历史 V6 输出、第一轮 revision trigger、one-round Soft lifecycle 均通过校验。三者只改变 H→Soft 表示，V6 的 shadow verifier、trigger、threshold、max-accept 与下一轮 arbitration 保持不变。

| Method | Correct / Total | Accuracy | Δ vs V6 | Avg NFE | Δ NFE | Recovered | Regressed | Net |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Soft Revision V6 | 1040/1319 | 78.85 | — | 42.60 | — | — | — | — |
| Contrastive Soft Revision | 1031/1319 | 78.17 | -0.68 | 42.49 | -0.11 | 42 | 51 | -9 |
| Geometry-Aware Soft Revision | 1032/1319 | 78.24 | -0.61 | 42.59 | -0.01 | 30 | 38 | -8 |
| Revision-Aware Soft Adapter | 1032/1319 | 78.24 | -0.61 | 42.83 | +0.23 | 69 | 77 | -8 |

Contrastive posterior 的影响非常强：old token 的平均概率从 shadow posterior 的 0.652 降到 contrastive posterior 的 0.135，中位数从 0.750 降到 0.018；39,653 次 revision 中有 28,768 次（72.55%）改变 posterior top-1。它并未提升最终准确率，说明完整 shadow posterior 中被扣除的 native/self-conditioned 成分并非主要是有害信息；当前 eta=1 的对比重加权更像过度校正。

Geometry 变体与原始 V6 Soft 的平均 cosine 达 0.9923，但把表示 norm 从平均 5.736 提高到固定的约 7.883。最终有 30 个 recovery、38 个 regression。这表明在方向几乎不变时，单独把 Euclidean mixing 替换为归一化方向聚合和 SLERP 没有带来收益，反而可能因为 norm 重标定扰动轨迹。

Adapter 在 GSM8K train 的 8 个样本、113 个 revision event 上收集数据，执行 64 次更新，backbone 全部冻结。它在 test 上制造了最多的 paired 变化（69 recovery、77 regression），但净值仍为 -8；adapter residual norm 中位数为 18.55，明显不是轻微修正。现有训练目标能够改变 Soft 表示，却没有把 shadow evidence distillation 对齐到最终任务正确性。

三次运行与其他 GPU 作业并发，记录的绝对 latency 受资源竞争影响，不适合与历史 V6 latency 作严格比较；Accuracy 与逻辑 NFE 不受这一并发口径影响。完整结构化结果分别保存在 [Contrastive](/root/lx/WINO-DLLM/LLaDA/results/soft_revision_contrastive/soft_revision_contrastive_summary.json)、[Geometry](/root/lx/WINO-DLLM/LLaDA/results/soft_revision_geometry/soft_revision_geometry_summary.json) 和 [Adapter](/root/lx/WINO-DLLM/LLaDA/results/soft_revision_adapter/soft_revision_adapter_summary.json)。

## 当前判断

现有证据不支持“V6 的主要问题只是 Hard commit 太激进”。如果这是主因，Fixed Interpolation 和 Soft Branching 应该在任务间一致减少 regression；实际收益集中于 GSM8K/MATH/MBPP，而 Countdown、ARC 和部分 HumanEval 下降。更符合数据的解释是：revision representation 会改变共享推理轨迹，但这种变化没有稳定方向；局部保留不确定性能够制造 recoveries，也会制造数量相近的 regressions。

按 8-task AVG，当前仍是 V6 最好（49.87），其次是 Fixed alpha=0.25（49.66）、Fixed alpha=0.75（49.47）、Adaptive（49.42）和 Fixed alpha=0.5（49.34）。这些宏平均混合了 Sudoku cell accuracy，仅适合作为统一比较；paired 结果同样表明没有新方法跨任务稳定支配 V6。

## WINO revision 消融（GSM8K）

完整 GSM8K 的 No-Same/No-Diff 使用已有 Full WINO 轨迹提供事后 SAME/DIFF 标签。干预后只抑制仍能按 block、round、position 和 old-token identity 严格对齐的历史事件；轨迹分叉后出现的新增或未匹配 revision 保持正常执行。因此这里测量的是**严格对齐事件子集的因果干预**，不是删除全部 SAME-ID 或 DIFF-ID revision。

| Method | Correct / Total | Accuracy | Δ vs Full WINO | Avg NFE | Recovered | Regressed | Net |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full WINO | 1023/1319 | 77.56 | — | 46.25 | — | — | — |
| WINO-No-Same | 1041/1319 | 78.92 | +1.36 | 45.58 | 78 | 60 | +18 |
| WINO-No-Diff | 1031/1319 | 78.17 | +0.61 | 46.10 | 74 | 66 | +8 |

| Ablation | Historical target events | Strictly matched and suppressed | Coverage | Outputs changed |
|---|---:|---:|---:|---:|
| No-Same | 57,237 | 4,449 | 7.77% | 1,138/1,319 |
| No-Diff | 5,631 | 810 | 14.38% | 1,076/1,319 |

这组结果表明，在仍可严格对齐的子集上，阻止 SAME-ID 或 DIFF-ID remask 都没有降低最终 Accuracy；No-Same 的净变化更正向。但覆盖率较低，不能据此声称“91% 的 SAME-ID revision 整体有害”或“全部 DIFF-ID revision 不重要”。Full WINO 相对 Only-Draft 的收益仍证明 verification/remasking 模块整体有价值，而当前消融说明其收益不能简单等同于某一类最终 token identity transition。

另有一个仅覆盖 GSM8K 前 100 条的本地 `WINO Only-Draft / No-Revision` 对照：72/100、Accuracy 72.00%、Avg NFE 39.38。同一批 fresh runs 中 Vanilla 为 77/100、Full WINO 为 82/100、Full V6 为 79/100。由于样本范围不同，这组 100-sample 数据不加入顶部完整 benchmark 主表。

[完整 SAME/DIFF 汇总](/root/lx/WINO-DLLM/LLaDA/results/wino_v6_revision_ablation/wino_no_same_no_diff_full_gsm8k_summary.json)

## WINO 论文中的效率结果

论文 Table 1 使用 LLaDA-8B-Instruct、generation length 256、block length 128。除 Sudoku 使用 4-shot 外，其余 benchmark 均为 zero-shot。WINO 的 verification threshold 为 $\tau_2=0.9$，draft threshold $\tau_1$ 从 $\{0.5,0.6,0.7\}$ 中选择。

| Dataset | LLaDA Acc | WINO Acc | Acc Δ | LLaDA Steps | WINO Steps | Step reduction | WINO TPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| GSM8K | 73.24 | 75.82 | +2.58 | 256 | 41.93 | 6.10× | 100.53 |
| MATH-500 | 32.00 | 34.20 | +2.20 | 256 | 74.44 | 3.44× | 55.86 |
| HumanEval | 37.80 | 42.07 | +4.27 | 256 | 93.32 | 2.74× | 37.19 |
| MBPP | 36.40 | 36.40 | +0.00 | 256 | 96.57 | 2.65× | 45.39 |
| Countdown | 24.21 | 33.20 | +8.99 | 256 | 105.88 | 2.41× | 38.97 |
| Sudoku | 14.23 | 15.20 | +0.97 | 256 | 131.96 | 1.94× | 21.11 |
| ARC-E | 59.13 | 81.19 | +22.06 | 256 | 40.19 | 6.37× | 101.61 |
| ARC-C | 51.87 | 73.89 | +22.02 | 256 | 47.41 | 5.40× | 85.42 |

## V6 相对论文 WINO

这张差值表只描述最终数值差异。论文与本地实验可能存在代码版本、draft threshold 选择、数据预处理或 evaluator 细节差异，因此不能把差值全部归因于 Soft Revision。

| Dataset | WINO paper Acc | V6 Acc | Δ Acc | WINO paper Steps | V6 Avg NFE | Δ Steps/NFE |
|---|---:|---:|---:|---:|---:|---:|
| GSM8K | 75.82 | 78.85 | +3.03 | 41.93 | 42.60 | +0.67 |
| MATH-500 | 34.20 | 34.60 | +0.40 | 74.44 | 70.63 | -3.81 |
| HumanEval | 42.07 | 43.90 | +1.83 | 93.32 | 86.10 | -7.22 |
| MBPP | 36.40 | 36.00 | -0.40 | 96.57 | 94.89 | -1.68 |
| Countdown | 33.20 | 35.94 | +2.74 | 105.88 | 88.54 | -17.34 |
| Sudoku | 15.20 | 13.45 | -1.75 | 131.96 | 124.82 | -7.14 |
| ARC-E | 81.19 | 81.61 | +0.42 | 40.19 | 35.31 | -4.88 |
| ARC-C | 73.89 | 74.58 | +0.69 | 47.41 | 42.62 | -4.79 |

从这组对照看，V6 在 8 个任务中的 6 个高于论文 WINO，同时在其中 5 个任务减少平均 Steps/NFE。落后的两个任务是 MBPP（-0.40）和 Sudoku（-1.75）。GSM8K Accuracy 更高，但平均 NFE 比论文 WINO 多 0.67。

## 论文来源

Hong et al., *Wide-In, Narrow-Out: Revokable Decoding for Efficient and Effective DLLMs*, 2025，Table 1 与 Section 5.1。论文文件位于 `/root/lx/Hong 等 - 2025 - Wide-In, Narrow-Out Revokable Decoding for Efficient and Effective DLLMs.pdf`。
