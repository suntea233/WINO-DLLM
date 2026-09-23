# Factorized Soft Revision — GSM8K

## Final result

| Method | Accuracy | Correct | Avg NFE | H→Soft events |
|---|---:|---:|---:|---:|
| Soft Revision V6 | 78.85% | 1040/1319 | 42.60 | 38,433 |
| Factorized Soft Revision | 76.50% | 1009/1319 | 41.16 | 37,538 |

Paired against V6: **41 recovered, 72 regressed, net −31**. The paired sign imbalance is unlikely under equal recovery/regression probability (exact two-sided binomial p≈0.0046). Exact final responses changed for 1,071/1,319 samples (81.2%).

Factorized reduces logical NFE by 1.44/sample (3.4%), but this comes with a 2.35-point accuracy loss. Its measured 13.63 s/sample latency is not directly comparable with the historical V6 latency because GPU contention and read-only per-layer attention instrumentation differed.

## Revision dynamics

| Resolution | V6 | Factorized |
|---|---:|---:|
| S→H(old) | 12,272 (31.9%) | 20,032 (53.4%) |
| S→H(new) | 4,322 (11.2%) | 3,535 (9.4%) |
| S→MASK | 21,839 (56.8%) | 13,971 (37.2%) |

On 1,128 samples with a contemporary V6 replay, the first H→Soft trigger matched for all 1,128 samples. Nevertheless, the first resolution changed in 295 cases (26.2%). The largest transition was **188 V6 S→MASK events becoming Factorized S→H(old)**. In those events, Factorized raised old-token probability by 0.151 on average.

Across matched first events, old-token probability increased by 0.050 on average; it increased in 71.5% of events. Entropy decreased in 70.7% of events. Factorization therefore makes the revision posterior more confident and more likely to restore the old identity.

## Attention behavior

Factorized added exactly one M token per Soft position: 37,538 tokens over 17,606 factorized forwards, averaging 2.13 active Soft positions per such round (maximum 17).

Although the C/M log priors are both `log(0.5)`, actual C-relative attention is highly layer dependent:

- layer 0: **0.9005 C**
- layer 1: 0.5224 C
- layer 2: 0.4188 C
- later-layer aggregate: generally about 0.47–0.52 C
- all-layer mean: 0.5069 C

The overall mean near 0.5 hides a strong first-layer semantic bias. C and M endpoint top-1 predictions disagree in 9.95% of events. Late fusion differs from the same-forward raw shadow top-1 in 8.20%.

## Interpretation

The failure is not caused by the H→Soft trigger: the first trigger is identical to V6. The main issue is that C sees its own semantic state and is strongly preferred in the first layer. The implementation then computes `p_rev = 0.5 p_C + 0.5 p_M` and uses it in place of the raw leave-one-out shadow posterior for V6 arbitration. This reintroduces self-conditioning into the verifier, raises support for the old identity, and converts many uncertain `S→MASK` outcomes into premature `S→H(old)` outcomes.

Pre-mixing in V6 is therefore not a crude version of late fusion. It constrains semantic and MASK information into one input vector before every Transformer layer, while preserving WINO shadow as the contextual arbiter. Separate C/M tokens create two nonlinear hidden trajectories; averaging their output probabilities cannot recover the trajectory of the original Mixed Soft input.

A clean follow-up would preserve raw WINO shadow for `S→H(old/new)/MASK`, prevent `p_rev` from overwriting the verifier posterior, and use C/M only to test how unresolved drafting positions read the temporary revision. A float-mask V6 control is also required to exclude SDPA mask-type effects. No alpha tuning is justified by this result.

## Compute and memory

- Estimated added linear token work: 0.13%
- Estimated added quadratic attention work: 0.26%
- Peak allocated GPU memory: 17.14 GiB

These FLOP estimates exclude the diagnostic-only per-layer C/M attention-statistics computation.
