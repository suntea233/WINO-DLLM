# Recent factorized Soft experiments (GSM8K)

The four directories below contain the saved report, a small `published_summary.json`, and one row per evaluated sample in `published_samples.jsonl` (index, final response, prediction, correctness, NFE, latency). The full local run artifacts contain large per-round attention diagnostics and are intentionally omitted from this GitHub release.

| Experiment | Official correct / 1319 | Avg NFE | Paired vs historical V6 |
|---|---:|---:|---:|
| [Factorized Soft Revision](factorized_soft_revision/report_factorized_soft_revision.md) | 1009 | 41.16 | 41 recovered / 72 regressed |
| [Contextualized Sequence Slot](soft_revision_sequence_slot/report_soft_revision_sequence_slot.md) | 1016 | 43.40 | 47 recovered / 71 regressed |
| [Factorized Logical Slot](factorized_logical_slot/report_factorized_logical_slot.md) | 1039 | 42.44 | 49 recovered / 50 regressed |
| [Explicit Factorized Soft](explicit_factorized_soft/report_explicit_factorized_soft.md) | 1035 | 42.56 | 53 recovered / 58 regressed |

The official GSM8K extractor misread `2:00 pm` as `00` for sample 1001 in WINO, V6, Logical Slot, and Explicit Factorized Soft. Logical Slot also had two final-box extraction errors (839 and 868). [The audit record](explicit_factorized_soft/answer_extraction_audit.json) preserves both official and corrected counts; the corrected counts are 1024 WINO, 1041 V6, 1042 Logical Slot, and 1036 Explicit. The comparison with V6 remains +1 for Logical Slot and −5 for Explicit after the common time-format correction.

The original official evaluations and complete raw local artifacts were not overwritten.
