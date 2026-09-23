# Contextualized Sequence-Slot Soft Revision — GSM8K

| Method | Accuracy | Correct | Avg NFE | Avg latency |
|---|---:|---:|---:|---:|
| WINO | 77.5588% | 1023/1319 | 46.25 | 3.380s |
| Current-run Soft Revision V6 | 78.8476% | 1040/1319 | 42.60 | 12.053s |
| Sequence Slot | 77.0281% | 1016/1319 | 43.40 | 12.604s |

## Paired against V6

- Recovered: 47
- Regressed: 71
- Net: -24
- Exact outputs changed: 1125

## Mechanism

- Revision events: 41285
- Sequence-slot forwards: 18903
- Total slot tokens: 41285
- Average slots per affected forward: 2.184
- Maximum slots in one forward: 19
- Unresolved-query/slot visible pairs: 1571412
- Protected-query/slot visible pairs: 0
- Resolution old/new/MASK: 10720 / 4790 / 25775

The original Hard identity remains in the canonical sequence. Each challenged
position contributes one true, contextualized V6-Soft token for one normal
forward. Only unresolved generation queries may attend to that token; prompt,
committed Hard, pivot, and WINO shadow queries remain isolated. No additional
backbone call is made.

Historical saved V6 exact replay mismatches: 0/1319.
The paired comparison therefore uses V6 regenerated in the same process.
