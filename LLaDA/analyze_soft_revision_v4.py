#!/usr/bin/env python3
"""Offline descriptive analysis of the completed GSM8K Soft Revision V4 run."""

import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / "results/soft_revision_v4"
V4_PATH = RESULT_DIR / "gsm8k_soft_revision_v4.json"
V1_PATH = ROOT / "results/soft_revision/gsm8k_soft_revision.json"
WINO_PATH = ROOT / "results/soft_revision/gsm8k_wino_remask.json"


def mean(values):
    return statistics.fmean(values) if values else 0.0


def exact_mcnemar_p(recovered, regressed):
    n = recovered + regressed
    k = min(recovered, regressed)
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)


def token_category(token):
    value = token.replace("Ġ", "").replace("Ċ", "").strip()
    if "<|" in value:
        return "special"
    if any(character.isdigit() for character in value):
        return "numeric"
    if value in {"+", "-", "*", "/", "=", "×", "÷"} or value.lower() in {
            "plus", "minus", "times", "half", "double", "twice"}:
        return "operator"
    if value and all(not character.isalnum() for character in value):
        return "punctuation"
    return "word"


def main():
    v4 = json.loads(V4_PATH.read_text())
    v1 = json.loads(V1_PATH.read_text())
    wino = json.loads(WINO_PATH.read_text())
    rows = v4["samples"]
    rows_by_id = {row["index"]: row for row in rows}
    v1_by_id = {row["index"]: row for row in v1["samples"]}

    events = [event for row in rows for event in row["diagnostics"]["revision_events"]]
    changed = [event for event in events if event.get("identity_changed")]
    action_counts = Counter(event["final_action"] for event in events)
    top1_same = sum(
        event["final_action"] == "S->H(b)" and not event["identity_changed"]
        for event in events)

    outcome_groups = defaultdict(list)
    for row in rows:
        old_correct = v1_by_id[row["index"]]["is_correct"]
        new_correct = row["is_correct"]
        group = (
            "both_correct" if old_correct and new_correct else
            "recovered" if not old_correct and new_correct else
            "regressed" if old_correct and not new_correct else "both_wrong")
        outcome_groups[group].append(row)

    group_statistics = {}
    for group, group_rows in outcome_groups.items():
        diagnostics = [row["diagnostics"] for row in group_rows]
        changed_by_sample = [
            sum(event.get("identity_changed", False)
                for event in diagnostic["revision_events"])
            for diagnostic in diagnostics]
        group_statistics[group] = {
            "samples": len(group_rows),
            "mean_nfe": mean([row["nfe"] for row in group_rows]),
            "mean_h_to_s": mean([
                diagnostic["num_h_to_soft_revisions"] for diagnostic in diagnostics]),
            "mean_s_to_m": mean([
                diagnostic["num_h_to_s_to_m"] for diagnostic in diagnostics]),
            "mean_changed_identity_events": mean(changed_by_sample),
            "fraction_samples_with_identity_change": (
                sum(value > 0 for value in changed_by_sample) / len(group_rows)),
        }

    stage_all = Counter()
    stage_changed = Counter()
    for row in rows:
        for event in row["diagnostics"]["revision_events"]:
            stage = min(3, int(4 * event["created_round"] / max(1, row["nfe"])))
            stage_all[stage] += 1
            if event.get("identity_changed"):
                stage_changed[stage] += 1
    stage_labels = ["0-25%", "25-50%", "50-75%", "75-100%"]
    stage_statistics = {
        stage_labels[index]: {
            "all_soft_events": stage_all[index],
            "changed_identity_events": stage_changed[index],
            "changed_rate": stage_changed[index] / stage_all[index],
            "share_of_all_changed_events": stage_changed[index] / len(changed),
        }
        for index in range(4)
    }

    category_counts = Counter(
        token_category(event["original_hard_token_string"]) for event in changed)
    pair_counts = Counter(
        (event["original_hard_token_string"], event["new_hard_token_string"])
        for event in changed)

    def probability_summary(predicate):
        selected = [event for event in events if predicate(event)]
        return {
            "events": len(selected),
            "mean_p_old": mean([event["p_old"] for event in selected]),
            "mean_p_top1": mean([event["p_top1"] for event in selected]),
        }

    metrics = v4["metrics"]
    comparisons = {}
    for name, reference in [("v1", v1), ("wino", wino)]:
        reference_metrics = reference["metrics"]
        paired = v4["paired"][name]
        comparisons[name] = {
            "reference_accuracy": reference_metrics["accuracy"],
            "accuracy_delta": metrics["accuracy"] - reference_metrics["accuracy"],
            "reference_average_nfe": reference_metrics["average_nfe"],
            "nfe_delta": metrics["average_nfe"] - reference_metrics["average_nfe"],
            "relative_nfe": metrics["average_nfe"] / reference_metrics["average_nfe"],
            "reference_average_latency": reference_metrics["average_latency"],
            "latency_delta": metrics["average_latency"] - reference_metrics["average_latency"],
            "recovered": paired["recovered"],
            "regressed": paired["regressed"],
            "net_recovery": paired["net_recovery"],
            "exact_mcnemar_p": exact_mcnemar_p(
                paired["recovered"], paired["regressed"]),
        }

    analysis = {
        "metrics": metrics,
        "comparisons": comparisons,
        "arbitration": {
            "action_counts": dict(action_counts),
            "middle_branch_same_identity": top1_same,
            "middle_branch_changed_identity": len(changed),
            "changed_fraction_of_middle_branch": (
                len(changed) / action_counts["S->H(b)"]),
            "probabilities": {
                "old_hard_confirmed": probability_summary(
                    lambda event: event["final_action"] == "S->H(a)"),
                "middle_branch_same_identity": probability_summary(
                    lambda event: event["final_action"] == "S->H(b)"
                    and not event["identity_changed"]),
                "middle_branch_changed_identity": probability_summary(
                    lambda event: event.get("identity_changed", False)),
                "mask_fallback": probability_summary(
                    lambda event: event["final_action"] == "S->M"),
            },
        },
        "outcome_group_statistics_vs_v1": group_statistics,
        "changed_token_categories": {
            category: {
                "count": count,
                "percentage": count / len(changed),
            }
            for category, count in category_counts.most_common()
        },
        "changed_event_stage_statistics": stage_statistics,
        "most_common_changed_pairs": [
            {"old": old, "new": new, "count": count}
            for (old, new), count in pair_counts.most_common(20)
        ],
        "representative_recoveries_vs_v1": [98, 112, 316, 620, 866, 1199, 1255],
        "representative_regressions_vs_v1": [39, 165, 174, 201, 261, 707, 855],
        "limitations": [
            "The middle branch changes both remasking behavior and token identity; V4 is not a pure identity-only ablation.",
            "Event frequency and outcome are observational and do not establish causality.",
            "Latency was measured in separate runs under potentially different machine load.",
        ],
    }
    (RESULT_DIR / "analysis_v4.json").write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2) + "\n")

    v1_cmp = comparisons["v1"]
    wino_cmp = comparisons["wino"]
    recovered = group_statistics["recovered"]
    regressed = group_statistics["regressed"]
    lines = [
        "Soft Revision V4: result analysis",
        "=================================",
        "",
        "Main result",
        "-----------",
        f"V4: {metrics['accuracy']:.4%} ({metrics['correct']}/{metrics['total']}), average NFE {metrics['average_nfe']:.2f}, average latency {metrics['average_latency']:.3f}s.",
        f"Versus V1: {v1_cmp['accuracy_delta']:+.4%} ({v1_cmp['net_recovery']:+d} net samples), NFE {v1_cmp['nfe_delta']:+.2f} ({v1_cmp['relative_nfe']:.3f}x), paired exact p={v1_cmp['exact_mcnemar_p']:.3f}.",
        f"Versus WINO: {wino_cmp['accuracy_delta']:+.4%} ({wino_cmp['net_recovery']:+d} net samples), NFE {wino_cmp['nfe_delta']:+.2f} ({wino_cmp['relative_nfe']:.3f}x), paired exact p={wino_cmp['exact_mcnemar_p']:.3f}.",
        "",
        "Arbitration behavior",
        "--------------------",
        f"Total H->S: {len(events)}",
        f"S->H(a) via p_old>=0.9: {action_counts['S->H(a)']}",
        f"S->H(b) middle branch: {action_counts['S->H(b)']}",
        f"  b==a: {top1_same}",
        f"  b!=a: {len(changed)} ({len(changed) / action_counts['S->H(b)']:.2%} of middle-branch commits)",
        f"S->M: {action_counts['S->M']}",
        "",
        f"Changed events have mean p_old={analysis['arbitration']['probabilities']['middle_branch_changed_identity']['mean_p_old']:.3f} and mean p_top1={analysis['arbitration']['probabilities']['middle_branch_changed_identity']['mean_p_top1']:.3f}.",
        "Most middle-branch actions recommit the same token earlier than V1 would; this is the main source of fewer MASK states and lower NFE.",
        "",
        "Paired outcomes versus V1",
        "--------------------------",
        f"V1 wrong -> V4 correct: {v1_cmp['recovered']}",
        f"V1 correct -> V4 wrong: {v1_cmp['regressed']}",
        f"Net recovery: {v1_cmp['net_recovery']:+d}",
        f"Recovered: changed events {recovered['mean_changed_identity_events']:.2f}/sample; {recovered['fraction_samples_with_identity_change']:.2%} of samples contain one.",
        f"Regressed: changed events {regressed['mean_changed_identity_events']:.2f}/sample; {regressed['fraction_samples_with_identity_change']:.2%} of samples contain one.",
        f"Recovered: H->S {recovered['mean_h_to_s']:.2f}/sample, S->M {recovered['mean_s_to_m']:.2f}/sample, NFE {recovered['mean_nfe']:.2f}.",
        f"Regressed: H->S {regressed['mean_h_to_s']:.2f}/sample, S->M {regressed['mean_s_to_m']:.2f}/sample, NFE {regressed['mean_nfe']:.2f}.",
        "",
        "Changed-token types",
        "-------------------",
    ]
    for category, values in analysis["changed_token_categories"].items():
        lines.append(f"{category}: {values['count']} ({values['percentage']:.2%})")
    lines += ["", "Changed-token timing", "--------------------"]
    for label, values in stage_statistics.items():
        lines.append(
            f"{label}: {values['changed_identity_events']} changed events; "
            f"{values['changed_rate']:.2%} of all Soft episodes in this stage")
    lines += [
        "",
        "Representative paired cases",
        "---------------------------",
        "Recoveries: sample 620 restores the missing leading 1 in 1509; sample 866 repairs 3F+11 to 3F+1 and reaches 28; sample 1255 changes repeated 4 tokens to 2 and reaches 12.",
        "Regressions: sample 39 changes 4/8 tokens and reaches 24 instead of 18; sample 201 corrupts several digits in the annual total; sample 707 derives 8 percent but changed late tokens lead to a boxed 20 percent.",
        "Some recoveries have no identity-changing event (for example sample 98). Their change comes from the middle branch recommitting the same identity instead of V1's MASK fallback.",
        "",
        "Conclusion",
        "----------",
        "V4 substantially changes individual trajectories but does not improve aggregate GSM8K accuracy over V1: 78 recoveries are offset by 81 regressions. The three-sample difference is small relative to 159 discordant pairs.",
        "Identity changes are slightly more frequent in regressed than recovered samples, and the highest changed-event rate occurs in the final decoding quarter. This is consistent with late numeric/format changes being risky, but it is not causal evidence.",
        "V4 does improve the compute result: average NFE falls by 2.93 versus V1 and by 6.25 versus WINO. Most of that saving comes from same-identity early recommitment, rather than the 4315 true identity changes.",
        "The latency comparison does not follow NFE in this run; V4 was measured slower than the older saved runs, so cross-run system load and the larger number of Soft embedding operations should be considered before interpreting wall-clock time.",
    ]
    (RESULT_DIR / "report_soft_revision_v4_analysis.txt").write_text(
        "\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
