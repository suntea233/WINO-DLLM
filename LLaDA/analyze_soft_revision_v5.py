#!/usr/bin/env python3
"""Offline analysis of the completed GSM8K Soft Revision V5 ablation."""

import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
OUT_DIR = RESULTS / "soft_revision_v5"


def load(relative_path):
    return json.loads((RESULTS / relative_path).read_text())


def mean(values):
    return statistics.fmean(values) if values else 0.0


def exact_mcnemar_p(recovered, regressed):
    total = recovered + regressed
    smaller = min(recovered, regressed)
    return min(
        1.0,
        2.0 * sum(math.comb(total, index) for index in range(smaller + 1))
        / 2**total,
    )


def main():
    wino = load("soft_revision/gsm8k_wino_remask.json")
    v1 = load("soft_revision/gsm8k_soft_revision.json")
    v4 = load("soft_revision_v4/gsm8k_soft_revision_v4.json")
    v5 = load("soft_revision_v5/gsm8k_soft_revision_v5.json")
    rows = v5["samples"]
    events = [event for row in rows for event in row["diagnostics"]["revision_events"]]
    identity_changes = [event for event in events if event.get("identity_changed")]
    if identity_changes:
        raise RuntimeError(f"V5 contains {len(identity_changes)} identity changes")

    branches = Counter()
    for event in events:
        if event["commit_reason"] == "p_old>=threshold_back":
            branches["high_confidence_old_hard"] += 1
        elif event["commit_reason"] == "same_identity_early_recommit":
            branches["same_identity_early_recommit"] += 1
        elif (event["current_top1_token_id"] != event["original_hard_token_id"]
              and event["p_top1"] >= 0.6):
            branches["different_identity_candidate_to_mask"] += 1
        else:
            branches["low_top1_to_mask"] += 1

    def probability_summary(predicate):
        selected = [event for event in events if predicate(event)]
        return {
            "events": len(selected),
            "mean_p_old": mean([event["p_old"] for event in selected]),
            "mean_p_top1": mean([event["p_top1"] for event in selected]),
        }

    probabilities = {
        "high_confidence_old_hard": probability_summary(
            lambda event: event["commit_reason"] == "p_old>=threshold_back"),
        "same_identity_early_recommit": probability_summary(
            lambda event: event["commit_reason"] == "same_identity_early_recommit"),
        "different_identity_candidate_to_mask": probability_summary(
            lambda event: event["current_top1_token_id"]
            != event["original_hard_token_id"] and event["p_top1"] >= 0.6),
        "low_top1_to_mask": probability_summary(
            lambda event: event["final_action"] == "S->M"
            and not (event["current_top1_token_id"]
                     != event["original_hard_token_id"]
                     and event["p_top1"] >= 0.6)),
    }

    v5_by_id = {row["index"]: row for row in rows}
    group_statistics = {}
    for reference_name, reference in [("v1", v1), ("v4", v4)]:
        reference_by_id = {row["index"]: row for row in reference["samples"]}
        groups = defaultdict(list)
        for index, row in v5_by_id.items():
            old_correct = reference_by_id[index]["is_correct"]
            new_correct = row["is_correct"]
            group = (
                "both_correct" if old_correct and new_correct else
                "recovered" if not old_correct and new_correct else
                "regressed" if old_correct and not new_correct else "both_wrong")
            groups[group].append(row)
        group_statistics[reference_name] = {}
        for group, group_rows in groups.items():
            diagnostics = [row["diagnostics"] for row in group_rows]
            group_statistics[reference_name][group] = {
                "samples": len(group_rows),
                "mean_nfe": mean([row["nfe"] for row in group_rows]),
                "mean_h_to_s": mean([
                    diagnostic["num_h_to_soft_revisions"] for diagnostic in diagnostics]),
                "mean_same_identity_early_recommits": mean([
                    sum(event["commit_reason"] == "same_identity_early_recommit"
                        for event in diagnostic["revision_events"])
                    for diagnostic in diagnostics]),
                "mean_different_identity_candidates_sent_to_mask": mean([
                    sum(event["final_action"] == "S->M"
                        and event["current_top1_token_id"]
                        != event["original_hard_token_id"]
                        and event["p_top1"] >= 0.6
                        for event in diagnostic["revision_events"])
                    for diagnostic in diagnostics]),
                "mean_s_to_mask": mean([
                    diagnostic["num_h_to_s_to_m"] for diagnostic in diagnostics]),
            }

    metrics = v5["metrics"]
    comparisons = {}
    for name, reference in [("wino", wino), ("v1", v1), ("v4", v4)]:
        reference_metrics = reference["metrics"]
        pair = v5["paired"][name]
        comparisons[name] = {
            "accuracy": reference_metrics["accuracy"],
            "accuracy_delta": metrics["accuracy"] - reference_metrics["accuracy"],
            "average_nfe": reference_metrics["average_nfe"],
            "nfe_delta": metrics["average_nfe"] - reference_metrics["average_nfe"],
            "relative_nfe": metrics["average_nfe"] / reference_metrics["average_nfe"],
            "average_latency": reference_metrics["average_latency"],
            "latency_delta": metrics["average_latency"] - reference_metrics["average_latency"],
            "recovered": pair["recovered"],
            "regressed": pair["regressed"],
            "net_recovery": pair["net_recovery"],
            "exact_mcnemar_p": exact_mcnemar_p(
                pair["recovered"], pair["regressed"]),
        }

    nfe_reduction_v1_v4 = (
        v1["metrics"]["average_nfe"] - v4["metrics"]["average_nfe"])
    nfe_reduction_v1_v5 = (
        v1["metrics"]["average_nfe"] - metrics["average_nfe"])
    analysis = {
        "metrics": metrics,
        "comparisons": comparisons,
        "nfe_reduction_retained_from_v4": (
            nfe_reduction_v1_v5 / nfe_reduction_v1_v4),
        "branch_counts": dict(branches),
        "branch_percentages": {
            name: count / len(events) for name, count in branches.items()
        },
        "branch_probability_statistics": probabilities,
        "outcome_group_statistics": group_statistics,
        "interpretation": {
            "effect_a_same_identity_early_recommit_vs_v1": {
                "net_accuracy_samples": comparisons["v1"]["net_recovery"],
                "accuracy_delta": comparisons["v1"]["accuracy_delta"],
                "nfe_delta": comparisons["v1"]["nfe_delta"],
            },
            "effect_b_identity_correction_v4_vs_v5": {
                "net_accuracy_samples_for_v4": -comparisons["v4"]["net_recovery"],
                "accuracy_delta_v4_minus_v5": (
                    v4["metrics"]["accuracy"] - metrics["accuracy"]),
                "nfe_delta_v4_minus_v5": (
                    v4["metrics"]["average_nfe"] - metrics["average_nfe"]),
            },
        },
        "representative_cases_where_disabling_identity_change_helps": [
            39, 80, 150, 278, 393, 945, 1102],
        "representative_cases_where_disabling_identity_change_hurts": [
            198, 241, 292, 554, 1199],
        "limitations": [
            "The V4-versus-V5 contrast sends a different-identity candidate to HARD in V4 and MASK in V5, so subsequent trajectories and NFE also differ.",
            "Small paired differences do not establish a reliable accuracy effect.",
            "Latency came from separate runs under different GPU load and is not directly comparable.",
        ],
    }
    (OUT_DIR / "analysis_v5.json").write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2) + "\n")

    c1, c4 = comparisons["v1"], comparisons["v4"]
    v1_recovered = group_statistics["v1"]["recovered"]
    v1_regressed = group_statistics["v1"]["regressed"]
    lines = [
        "Soft Revision V5: same-identity early-recommit analysis",
        "========================================================",
        "",
        "Main result",
        "-----------",
        f"V5: {metrics['accuracy']:.4%} ({metrics['correct']}/{metrics['total']}), average NFE {metrics['average_nfe']:.2f}, average latency {metrics['average_latency']:.3f}s.",
        f"Versus V1: accuracy {c1['accuracy_delta']:+.4%} ({c1['net_recovery']:+d} samples), NFE {c1['nfe_delta']:+.2f}, paired exact p={c1['exact_mcnemar_p']:.3f}.",
        f"Versus V4: accuracy {c4['accuracy_delta']:+.4%} ({c4['net_recovery']:+d} samples), NFE {c4['nfe_delta']:+.2f}, paired exact p={c4['exact_mcnemar_p']:.3f}.",
        f"V5 retains {analysis['nfe_reduction_retained_from_v4']:.2%} of V4's NFE reduction relative to V1.",
        "",
        "Rule verification and branch counts",
        "-----------------------------------",
        f"Total H->S: {len(events)}",
        f"p_old>=0.9 -> H(a): {branches['high_confidence_old_hard']}",
        f"same-identity early recommit -> H(a): {branches['same_identity_early_recommit']}",
        f"different top-1 with p_top1>=0.6 -> MASK: {branches['different_identity_candidate_to_mask']}",
        f"low top-1 -> MASK: {branches['low_top1_to_mask']}",
        f"Identity-changing H(a)->S->H(b): {len(identity_changes)}",
        "",
        f"Different-identity candidates sent to MASK have mean p_old={probabilities['different_identity_candidate_to_mask']['mean_p_old']:.3f} and mean p_top1={probabilities['different_identity_candidate_to_mask']['mean_p_top1']:.3f}.",
        "",
        "Separating V4's two effects",
        "---------------------------",
        f"Effect A, same-identity early recommit (V5 versus V1): 68 recoveries, 75 regressions, net -7; average NFE changes by {c1['nfe_delta']:+.2f}.",
        "Effect B, identity correction (V4 versus V5): identity correction recovers 24 V5 failures but loses 20 V5 successes, net +4 for V4; average NFE changes by -0.71.",
        "Thus the observed V4 accuracy loss versus V1 is not caused mainly by identity-changing commits. Removing them makes accuracy four samples lower.",
        "",
        "Instability versus V1",
        "---------------------",
        f"Recovered samples: H->S {v1_recovered['mean_h_to_s']:.2f}, early recommits {v1_recovered['mean_same_identity_early_recommits']:.2f}, S->MASK {v1_recovered['mean_s_to_mask']:.2f}, NFE {v1_recovered['mean_nfe']:.2f}.",
        f"Regressed samples: H->S {v1_regressed['mean_h_to_s']:.2f}, early recommits {v1_regressed['mean_same_identity_early_recommits']:.2f}, S->MASK {v1_regressed['mean_s_to_mask']:.2f}, NFE {v1_regressed['mean_nfe']:.2f}.",
        "Regression samples again have more revisions and early recommits, but these statistics are descriptive rather than causal.",
        "",
        "Representative V4/V5 flips",
        "---------------------------",
        "Disabling identity changes helps samples 39 (18 instead of 24), 80 (10 instead of 0), 278 (40 instead of 30), 393 (20 instead of 25), and 1102 (113 instead of 93).",
        "Disabling identity changes hurts samples 198 (200 instead of 320), 241 (5 instead of 6), 292 (105 instead of 75), 554 (4 instead of 54), and 1199 (1 instead of 2).",
        "",
        "Answer to the main question",
        "---------------------------",
        "V5 retains most of V4's NFE reduction, but it does not remove the accuracy loss. It reaches 77.79%, below both V4 at 78.09% and V1 at 78.32%.",
        "The results are more consistent with same-identity early recommit causing the small aggregate accuracy cost, while identity correction provides a small net recovery and an additional NFE reduction. The paired accuracy differences are small and not statistically persuasive in this single GSM8K run.",
        "Wall-clock latency should not be compared directly because these runs used different GPU load conditions.",
    ]
    (OUT_DIR / "report_soft_revision_v5_analysis.txt").write_text(
        "\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
