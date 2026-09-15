# WINO vs V6 Revision Analysis

## TABLE 1 — Same-Identity Behavior

| Dataset | WINO Same-ID Events/Sample | V6 Same-ID Events/Sample | WINO Repeat Rate | V6 Repeat Rate | WINO NFE | V6 NFE |
|---|---:|---:|---:|---:|---:|---:|
| gsm8k | 43.29 | 9.30 | 48.5% | 11.8% | 46.25 | 42.60 |
| math500 | 31.83 | 6.94 | 48.9% | 14.9% | 73.49 | 70.63 |
| mbpp | 22.49 | 5.07 | 48.8% | 24.3% | 97.49 | 94.89 |
| humaneval | 44.72 | 8.09 | 51.1% | 19.2% | 92.30 | 86.10 |

## TABLE 2 — Identity Change Necessity

| Dataset | Identity-Change Samples | NEW-only | OLD-only | Both Correct | Both Wrong |
|---|---:|---:|---:|---:|---:|
| gsm8k | 1110 | 13 (1.2%) | 14 (1.3%) | 859 (77.4%) | 224 (20.2%) |
| math500 | 304 | 4 (1.3%) | 3 (1.0%) | 94 (30.9%) | 203 (66.8%) |
| mbpp | 97 | 0 (0.0%) | 2 (2.1%) | 30 (30.9%) | 65 (67.0%) |
| humaneval | 86 | 1 (1.2%) | 0 (0.0%) | 33 (38.4%) | 52 (60.5%) |

## Analysis 1 details

| Dataset | Method | Same-ID Return Rate | Repeated Loop Rate | Same-ID Events/Sample | Mean/median returns per affected position | Mean/median repeated challenges | Next-round challenge | Mean rounds to stability | NFE |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gsm8k | WINO | 91.1% | 48.5% | 43.29 | 2.59/1.0 | 1.63/1.0 | 49.5% | 6.87 | 46.25 |
| gsm8k | V6 | 31.9% | 11.8% | 9.30 | 1.18/1.0 | 0.27/0.0 | 12.4% | 3.44 | 42.60 |
| math500 | WINO | 91.1% | 48.9% | 31.83 | 2.29/1.0 | 1.33/1.0 | 36.4% | 8.68 | 73.49 |
| math500 | V6 | 29.8% | 14.9% | 6.94 | 1.24/1.0 | 0.34/0.0 | 11.1% | 4.49 | 70.63 |
| mbpp | WINO | 93.8% | 48.8% | 22.49 | 2.07/1.0 | 1.10/1.0 | 23.9% | 11.11 | 97.49 |
| mbpp | V6 | 28.1% | 24.3% | 5.07 | 1.38/1.0 | 0.46/0.0 | 11.8% | 7.19 | 94.89 |
| humaneval | WINO | 93.7% | 51.1% | 44.72 | 2.41/2.0 | 1.44/1.0 | 41.8% | 9.41 | 92.30 |
| humaneval | V6 | 27.0% | 19.2% | 8.09 | 1.32/1.0 | 0.41/0.0 | 14.9% | 5.35 | 86.10 |

### Matched outcome groups

| Dataset | Group | N | WINO Same-ID Events | V6 Same-ID Events | WINO repeated challenges | V6 repeated challenges | WINO NFE | V6 NFE |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| gsm8k | both_correct | 959 | 36.02 | 8.21 | 21.91 | 1.80 | 43.10 | 40.31 |
| gsm8k | WINO_wrong_to_V6_correct | 81 | 66.49 | 10.88 | 44.48 | 2.42 | 52.81 | 44.32 |
| gsm8k | WINO_correct_to_V6_wrong | 64 | 49.94 | 11.70 | 31.69 | 2.64 | 49.33 | 48.03 |
| gsm8k | both_wrong | 215 | 64.96 | 12.87 | 43.78 | 3.46 | 56.90 | 50.53 |
| math500 | both_correct | 156 | 24.17 | 5.46 | 14.06 | 1.38 | 57.97 | 55.87 |
| math500 | WINO_wrong_to_V6_correct | 17 | 49.18 | 8.12 | 30.12 | 2.24 | 79.53 | 69.94 |
| math500 | WINO_correct_to_V6_wrong | 13 | 34.77 | 8.92 | 20.31 | 2.23 | 69.69 | 63.69 |
| math500 | both_wrong | 314 | 34.57 | 7.54 | 20.09 | 2.12 | 81.04 | 78.30 |
| mbpp | both_correct | 176 | 19.30 | 4.38 | 10.26 | 1.56 | 90.62 | 88.63 |
| mbpp | WINO_wrong_to_V6_correct | 4 | 36.50 | 5.00 | 19.50 | 1.00 | 103.25 | 111.25 |
| mbpp | WINO_correct_to_V6_wrong | 4 | 43.50 | 6.00 | 28.00 | 1.25 | 114.00 | 94.25 |
| mbpp | both_wrong | 316 | 23.82 | 5.44 | 12.52 | 1.78 | 101.04 | 98.18 |
| humaneval | both_correct | 69 | 42.58 | 7.43 | 25.48 | 2.30 | 87.70 | 83.67 |
| humaneval | WINO_wrong_to_V6_correct | 3 | 23.67 | 7.67 | 8.67 | 1.33 | 65.33 | 75.00 |
| humaneval | WINO_correct_to_V6_wrong | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| humaneval | both_wrong | 92 | 47.01 | 8.59 | 28.11 | 2.67 | 96.64 | 88.29 |

## Analysis 2 aggregate

Across 1597 samples: NEW-only 18 (1.1%), OLD-only 19 (1.2%), both correct 1016 (63.6%), both wrong 544 (34.1%).

| Historical outcome | N | NEW-only | OLD-only | Both correct | Both wrong |
|---|---:|---:|---:|---:|---:|
| V6_correct | 1034 | 18 | 0 | 1016 | 0 |
| V6_wrong | 563 | 0 | 19 | 0 | 544 |

## Representative cases

### gsm8k
- NEW-only, sample 241: `-` → `\[`. NEW final correctness=True; OLD final correctness=False. The single forced identity choice changes the continuation and only the stated branch passes the final evaluator.
- OLD-only, sample 244: `Ġfries` → `Ġ`. NEW final correctness=False; OLD final correctness=True. The single forced identity choice changes the continuation and only the stated branch passes the final evaluator.

### math500
- NEW-only, sample 44: `Ġnorth` → `Ġsouth`. NEW final correctness=True; OLD final correctness=False. The single forced identity choice changes the continuation and only the stated branch passes the final evaluator.
- OLD-only, sample 157: `Ġrearrange` → `Ġmove`. NEW final correctness=False; OLD final correctness=True. The single forced identity choice changes the continuation and only the stated branch passes the final evaluator.

### mbpp
- NEW-only: none observed.
- OLD-only, sample 98: `Ġlength` → `Ġn`. NEW final correctness=False; OLD final correctness=True. The single forced identity choice changes the continuation and only the stated branch passes the final evaluator.

### humaneval
- NEW-only, sample 116: `Ġeach` → `Ġa`. NEW final correctness=True; OLD final correctness=False. The single forced identity choice changes the continuation and only the stated branch passes the final evaluator.
- OLD-only: none observed.

## Conclusions

1. Does Soft improve WINO repeated same-identity revision behavior? **YES**. This refers to observed loop frequency/duration, not token correctness.
2. Across first identity-changing events, the most common observed category is **redundant** (1016/1597).
