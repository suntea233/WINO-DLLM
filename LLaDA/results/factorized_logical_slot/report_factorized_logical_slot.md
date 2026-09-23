# Factorized Logical Slot

| Method | Accuracy | Correct | Avg NFE | Avg latency |
|---|---:|---:|---:|---:|
| V6 historical | 78.8476% | 1040/1319 | 42.598 | 4.766s |
| V6 custom-attention control | 78.8476% | 1040/1319 | 42.598 | 4.766s |
| Factorized Logical Slot | 78.7718% | 1039/1319 | 42.441 | 5.798s |

Paired Factorized vs V6: recovered 49, regressed 50, net -1.

Sequence length changes: 0. First-trigger matches: 1319/1319.

Revision outcomes: old 12070, new 4286, MASK 21962.

Mean C gate: 0.534620; Layer-0 C gate: 0.826912; authority ratio: 1.162594.

The pivot resolution path uses the untouched V6/WINO shadow posterior; no C/M endpoint posterior is constructed or fused.

## Answer-extraction audit

The saved final answers for GSM8K samples 839 and 868 were correct inside the final `<answer>` section, but the official extractor selected an earlier `\boxed{}` from the reasoning. Sample 1001 ended with `\boxed{2:00 \text{ pm}}`; the extractor read `00` although the reference answer is `2`. Correcting only these three extraction errors gives **1042/1319 (78.9992%)**. Applying the same time-format correction to V6 gives **1041/1319 (78.9234%)**. The audited paired result is **49 recovered / 48 regressed, net +1**. Original results above remain unchanged; see the [audit JSON](../explicit_factorized_soft/answer_extraction_audit.json).
