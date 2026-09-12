"""Create the requested GSM8K Hard-to-Soft Revision comparison artifacts."""

import csv
import json
import math
import os


ROOT = os.path.join(os.path.dirname(__file__), "results", "soft_revision")


def fmt(value):
    return f"{value:.4f}"


def main():
    with open(os.path.join(ROOT, "gsm8k_wino_remask.json"), encoding="utf-8") as handle:
        baseline = json.load(handle)
    with open(os.path.join(ROOT, "gsm8k_soft_revision.json"), encoding="utf-8") as handle:
        soft = json.load(handle)

    fields = [
        "Method", "Accuracy", "Correct", "Total", "Avg NFE", "Avg Latency",
        "Suspicious H", "H->MASK", "H->SOFT", "H->S->H", "H->S->M",
        "Soft recovery rate",
    ]
    rows = []
    for label, result in (("WINO Remask", baseline), ("Soft Revision", soft)):
        m = result["metrics"]
        rows.append({
            "Method": label,
            "Accuracy": m["accuracy"],
            "Correct": m["correct"],
            "Total": m["total"],
            "Avg NFE": m["average_nfe"],
            "Avg Latency": m["average_latency"],
            "Suspicious H": m["total_suspicious_hard_tokens"],
            "H->MASK": m["total_h_to_mask_revisions"],
            "H->SOFT": m["total_h_to_soft_revisions"],
            "H->S->H": m["total_h_to_s_to_h"],
            "H->S->M": m["total_h_to_s_to_m"],
            "Soft recovery rate": m["soft_recovery_rate"],
        })
    with open(os.path.join(ROOT, "summary.csv"), "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    base_by_index = {row["index"]: row for row in baseline["samples"]}
    soft_by_index = {row["index"]: row for row in soft["samples"]}
    if set(base_by_index) != set(soft_by_index):
        raise RuntimeError("WINO and Soft Revision sample IDs do not match")
    recovered = sum(
        not base_by_index[i]["is_correct"] and soft_by_index[i]["is_correct"]
        for i in base_by_index)
    regressed = sum(
        base_by_index[i]["is_correct"] and not soft_by_index[i]["is_correct"]
        for i in base_by_index)
    net = recovered - regressed
    discordant = recovered + regressed
    smaller = min(recovered, regressed)
    paired_exact_p = min(
        1.0,
        2.0 * sum(math.comb(discordant, k) for k in range(smaller + 1))
        / (2 ** discordant),
    ) if discordant else 1.0

    bm, sm = baseline["metrics"], soft["metrics"]
    delta = sm["accuracy"] - bm["accuracy"]
    nfe_delta = sm["average_nfe"] - bm["average_nfe"]
    latency_delta = sm["average_latency"] / bm["average_latency"] - 1.0
    recovery_rate = sm["soft_recovery_rate"]
    fallback_rate = 1.0 - recovery_rate if sm["total_h_to_soft_revisions"] else 0.0

    accuracy_answer = (
        "Yes" if delta > 0 else "No" if delta < 0 else "No accuracy difference was observed")
    strength_answer = (
        "No.  The direction is encouraging, but the gain is small and the paired exact test does not provide strong evidence on this single benchmark."
        if paired_exact_p >= 0.05 else
        "Yes as a follow-up experiment, although replication on other benchmarks is still needed."
    )
    propagation = (
        "Matched outcomes favor useful uncertainty preservation"
        if net > 0 else "Matched outcomes do not show a net benefit from retaining uncertainty"
    )
    report = f"""Hard-to-Soft Revision: GSM8K report
=====================================

Fixed official WINO setting: LLaDA-8B-Instruct, GSM8K test (n={sm['total']}),
gen_length=256, block_length=128, temperature=0, threshold=0.6,
threshold_back=0.9.  No parameter was tuned on GSM8K correctness.

Method             Accuracy        Correct/Total    Avg NFE    Avg Latency
WINO Remask        {fmt(bm['accuracy'])}          {bm['correct']}/{bm['total']}          {bm['average_nfe']:.2f}      {bm['average_latency']:.3f}s
Soft Revision      {fmt(sm['accuracy'])}          {sm['correct']}/{sm['total']}          {sm['average_nfe']:.2f}      {sm['average_latency']:.3f}s

Accuracy delta (Soft - WINO): {delta:+.4f}
Average NFE delta: {nfe_delta:+.2f}
Relative latency delta: {latency_delta:+.2%}
WINO wrong -> Soft correct: {recovered}
WINO correct -> Soft wrong: {regressed}
Net recovery: {net:+d}
Two-sided paired exact p-value: {paired_exact_p:.4f}
H->S count: {sm['total_h_to_soft_revisions']}
H->S->H count: {sm['total_h_to_s_to_h']}
H->S->M count: {sm['total_h_to_s_to_m']}
Soft recovery rate: {recovery_rate:.2%}
Soft-to-MASK fallback rate: {fallback_rate:.2%}

1. Does HARD -> SOFT outperform WINO's HARD -> MASK under the identical detector?
{accuracy_answer}: the accuracy difference is {delta:+.4f}, with paired net
recovery {net:+d} ({recovered} recoveries and {regressed} regressions).

2. Does Soft Revision preserve WINO's NFE advantage?
It adds zero auxiliary backbone calls: both decoders use one forward per normal
decoding round.  Its observed average NFE difference is {nfe_delta:+.2f}, caused
by changed stopping trajectories rather than lookahead forwards.

3. How often does a softened token recover to HARD on the next verification?
{sm['total_h_to_s_to_h']} of {sm['total_h_to_soft_revisions']} revisions,
or {recovery_rate:.2%}.

4. How often does Soft Revision still fall back to MASK?
{sm['total_h_to_s_to_m']} of {sm['total_h_to_soft_revisions']} revisions,
or {fallback_rate:.2%}.

5. Does it preserve useful uncertainty or cause additional error propagation?
{propagation}: the paired net recovery is {net:+d}, while {regressed}
regressions remain.  This is a descriptive comparison; outcome counts alone
cannot establish the causal mechanism.

6. Is the result strong enough for a second-stage H/S/M controller?
{strength_answer}
"""
    with open(os.path.join(ROOT, "report_soft_revision.txt"), "w", encoding="utf-8") as handle:
        handle.write(report)


if __name__ == "__main__":
    main()
