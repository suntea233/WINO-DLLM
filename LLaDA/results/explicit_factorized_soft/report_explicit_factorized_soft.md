# Explicit Factorized Soft Revision — GSM8K

| Method | Accuracy | Correct | Avg NFE | Avg latency |
|---|---:|---:|---:|---:|
| Historical V6 | 78.8476% | 1040/1319 | 42.60 | 4.766s |
| Explicit Factorized Soft Revision | 78.4685% | 1035/1319 | 42.56 | 13.016s |

Paired against historical V6: recovered 53, regressed 58, net -5.

## Mechanism diagnostics

- H→Soft events: 39021
- Same-ID events/sample: 9.5049
- S→H(old): 12537; S→H(new): 4081; S→MASK: 22403
- Factorized rounds: 18436
- Added sequence tokens: 39021 (one per Soft position)
- Mean Soft positions/factorized round: 2.117
- Maximum sequence-length increment: 16
- Mean C relative attention within C/M pair: 0.5114
- Endpoint posterior fusion/overwrite events: 0 (must be zero)
- Estimated linear token-work overhead: 0.13%
- Estimated quadratic attention-work overhead: 0.26%
- Peak allocated GPU memory: 17.13 GiB

Pivot resolution uses the untouched same-forward V6/WINO shadow posterior. C/M endpoint logits are never fused and never overwrite the verifier row.

## Answer-extraction audit (original scores preserved)

The official GSM8K extractor reads the last number inside a box. For sample 1001, the final answer is `\boxed{2:00 \text{ pm}}` and the reference is `2`; it extracted `00` and marked this answer wrong. The same formatting and misgrading occur in the saved WINO and V6 outputs. Re-evaluating all 1319 saved responses with final-`<answer>` priority and this time-format correction gives:

| Method | Official correct | Audited correct | Audited accuracy |
|---|---:|---:|---:|
| WINO | 1023 | 1024 | 77.6346% |
| V6 | 1040 | 1041 | 78.9234% |
| Explicit Factorized Soft | 1035 | 1036 | 78.5444% |

No other Explicit prediction or correctness changed. Its audited paired comparison with V6 remains **53 recovered / 58 regressed / net −5**. Thus the extraction correction does not reverse the result. Applying the same rule to Factorized Logical Slot corrects its previously identified samples 839 and 868 plus time-formatted sample 1001, giving **1042/1319 (78.9992%)** versus audited V6's **1041/1319**. The saved benchmark artifacts and official scores above remain untouched; the reproducible per-sample correction is in [answer_extraction_audit.json](answer_extraction_audit.json).
