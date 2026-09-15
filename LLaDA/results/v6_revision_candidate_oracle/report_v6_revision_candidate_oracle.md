# V6 Revision Candidate Oracle

Each branch changes exactly one S->H decision and then resumes frozen V6.
Only final official benchmark correctness is used. Oracle-Alt is an upper bound, not a decoder.
GSM8K pairing uses the same-run V1 control (1044/1319), whose output differs from the older archived V1 run (1033/1319) despite matching saved configuration. This avoids mixing run vintages; the paired groups should not be conflated.

| Dataset | Eligible recovered | Eligible regressed | Any Alt Rescue | OLD Rescue |
|---|---:|---:|---:|---:|
| gsm8k | 17 | 21 | 10 | 10 |
| math500 | 14 | 16 | 1 | 1 |
| mbpp | 0 | 5 | 3 | 2 |
| humaneval | 4 | 2 | 0 | 0 |

| Dataset | Rank2 | Rank3 | Rank4 | Any Alt |
|---|---:|---:|---:|---:|
| gsm8k | 9 | 9 | 7 | 10 |
| math500 | 1 | 0 | 0 | 1 |
| mbpp | 2 | 3 | 1 | 3 |
| humaneval | 0 | 0 | 0 | 0 |

| Dataset | V6 accuracy | Oracle-Alt accuracy | Oracle gain (pp) |
|---|---:|---:|---:|
| gsm8k | 78.85% | 79.61% | 0.76 |
| math500 | 34.60% | 34.80% | 0.20 |
| mbpp | 36.00% | 36.60% | 0.60 |
| humaneval | 43.90% | 43.90% | 0.00 |

| Dataset | Eligible recovered | Top1-only | B1 + alternative correct | Multiple alternatives correct | OLD correct |
|---|---:|---:|---:|---:|---:|
| gsm8k | 17 | 3 | 14 | 14 | 14 |
| math500 | 14 | 1 | 13 | 12 | 12 |
| mbpp | 0 | 0 | 0 | 0 | 0 |
| humaneval | 4 | 0 | 4 | 4 | 4 |

Across eligible regressions, alternative ranks rescue 14 samples and OLD rescues 13.

MBPP and other samples without an identity-changing V6 event are excluded from replay.
Outcome groups are sample-level; one earliest identity-changing event is replayed per eligible sample.
