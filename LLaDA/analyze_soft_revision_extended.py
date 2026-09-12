"""Offline paired analysis and extended report for frozen Soft Revision runs."""

import csv
import json
import math
import os
import statistics


ROOT = os.path.join(os.path.dirname(__file__), "results", "soft_revision")


def mean(values):
    return statistics.fmean(values) if values else None


def median(values):
    return statistics.median(values) if values else None


def pearson(xs, ys):
    if len(xs) < 2:
        return None
    mx, my = mean(xs), mean(ys)
    numerator = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denominator = math.sqrt(
        sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    return numerator / denominator if denominator else None


def summarize_group(indices, soft_by_id, events_by_id):
    rows = []
    for index in indices:
        events = events_by_id.get(index, [])
        hsh = [event for event in events if event["outcome"] == "H->S->H"]
        hsm = [event for event in events if event["outcome"] == "H->S->M"]
        rows.append({
            "h_to_s": len(events),
            "h_to_s_to_h": len(hsh),
            "h_to_s_to_m": len(hsm),
            "first_revision_round": min(
                (event["created_round"] for event in events), default=None),
            "unique_revised_positions": len({event["position"] for event in events}),
        })
    result = {"samples": len(indices)}
    for field in (
        "h_to_s", "h_to_s_to_h", "h_to_s_to_m", "unique_revised_positions"):
        values = [row[field] for row in rows]
        result[field] = {
            "total": sum(values), "mean": mean(values), "median": median(values)}
    first_rounds = [row["first_revision_round"] for row in rows
                    if row["first_revision_round"] is not None]
    result["first_revision_round"] = {
        "available_samples": len(first_rounds),
        "mean": mean(first_rounds),
        "median": median(first_rounds),
        "min": min(first_rounds) if first_rounds else None,
        "max": max(first_rounds) if first_rounds else None,
    }
    return result


def analyze_gsm8k():
    with open(os.path.join(ROOT, "gsm8k_wino_remask.json"), encoding="utf-8") as handle:
        baseline = json.load(handle)
    with open(os.path.join(ROOT, "gsm8k_soft_revision.json"), encoding="utf-8") as handle:
        soft = json.load(handle)
    events_by_id = {}
    with open(os.path.join(ROOT, "revision_trace.jsonl"), encoding="utf-8") as handle:
        for line in handle:
            event = json.loads(line)
            events_by_id.setdefault(event["sample_index"], []).append(event)
    baseline_by_id = {row["index"]: row for row in baseline["samples"]}
    soft_by_id = {row["index"]: row for row in soft["samples"]}
    if set(baseline_by_id) != set(soft_by_id):
        raise RuntimeError("GSM8K sample IDs do not match")
    recovered = [index for index in baseline_by_id
                 if not baseline_by_id[index]["is_correct"] and soft_by_id[index]["is_correct"]]
    regressed = [index for index in baseline_by_id
                 if baseline_by_id[index]["is_correct"] and not soft_by_id[index]["is_correct"]]

    hsh_events = [event for events in events_by_id.values() for event in events
                  if event["outcome"] == "H->S->H"]
    hsm_events = [event for events in events_by_id.values() for event in events
                  if event["outcome"] == "H->S->M"]
    # In the frozen implementation, a passing SOFT position clears its logical
    # mask and keeps x_block's original token_id; no new argmax is assigned.
    old_equals_new = len(hsh_events)
    old_differs_new = 0

    def remaining_rounds(event):
        total_steps = soft_by_id[event["sample_index"]]["steps"]
        return max(0, total_steps - event["resolved_round"] - 1)

    sample_nfe_rows = []
    for index in baseline_by_id:
        baseline_row, soft_row = baseline_by_id[index], soft_by_id[index]
        events = events_by_id.get(index, [])
        sample_nfe_rows.append({
            "index": index,
            "outcome_group": (
                "recovered" if index in recovered else
                "regressed" if index in regressed else
                "other"),
            "wino_nfe": baseline_row["nfe"],
            "soft_nfe": soft_row["nfe"],
            "delta_nfe": soft_row["nfe"] - baseline_row["nfe"],
            "wino_h_to_m": baseline_row["diagnostics"]["num_h_to_mask_revisions"],
            "soft_h_to_s_to_h": sum(event["outcome"] == "H->S->H" for event in events),
            "soft_h_to_s_to_m": sum(event["outcome"] == "H->S->M" for event in events),
        })
    with open(os.path.join(ROOT, "gsm8k_revision_nfe_samples.csv"), "w",
              newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sample_nfe_rows[0]))
        writer.writeheader()
        writer.writerows(sample_nfe_rows)

    def nfe_group(indices):
        rows = [row for row in sample_nfe_rows if row["index"] in set(indices)]
        return {
            "samples": len(rows),
            "mean_wino_nfe": mean([row["wino_nfe"] for row in rows]),
            "mean_soft_nfe": mean([row["soft_nfe"] for row in rows]),
            "mean_delta_nfe": mean([row["delta_nfe"] for row in rows]),
        }

    analysis = {
        "paired_groups": {
            "recovered": recovered,
            "regressed": regressed,
            "net_recovery": len(recovered) - len(regressed),
        },
        "group_statistics": {
            "recovered": summarize_group(recovered, soft_by_id, events_by_id),
            "regressed": summarize_group(regressed, soft_by_id, events_by_id),
        },
        "hard_recovery_identity": {
            "old_hard_equals_new_hard": old_equals_new,
            "old_hard_differs_new_hard": old_differs_new,
            "interpretation": (
                "Exact by frozen decoder semantics: H->S->H retains the stored old token ID; "
                "the verification pass does not assign a new argmax token."),
        },
        "nfe_analysis": {
            "all_samples": nfe_group(list(baseline_by_id)),
            "recovered": nfe_group(recovered),
            "regressed": nfe_group(regressed),
            "mean_remaining_rounds_after_soft_h_to_s_to_h": mean(
                [remaining_rounds(event) for event in hsh_events]),
            "mean_remaining_rounds_after_soft_h_to_s_to_m": mean(
                [remaining_rounds(event) for event in hsm_events]),
            "correlation_delta_nfe_vs_soft_h_to_s_to_m_count": pearson(
                [row["delta_nfe"] for row in sample_nfe_rows],
                [row["soft_h_to_s_to_m"] for row in sample_nfe_rows]),
            "correlation_delta_nfe_vs_revision_count_difference": pearson(
                [row["delta_nfe"] for row in sample_nfe_rows],
                [row["soft_h_to_s_to_m"] - row["wino_h_to_m"]
                 for row in sample_nfe_rows]),
            "unavailable_exact_event_fields": {
                "remaining_mask_count": (
                    "Not stored in revision_trace.jsonl or final per-sample diagnostics."),
                "new_hard_commits_in_following_round": (
                    "Only the revised position's H->S->H resolution is stored; other draft "
                    "commits in that round were not retained."),
                "wino_h_to_m_remaining_rounds": (
                    "The WINO baseline stores aggregate H->M counts but no event rounds."),
            },
        },
    }
    with open(os.path.join(ROOT, "gsm8k_revision_analysis.json"), "w", encoding="utf-8") as handle:
        json.dump(analysis, handle, ensure_ascii=False, indent=2)
    return analysis


if __name__ == "__main__":
    value = analyze_gsm8k()
    print(json.dumps(value, ensure_ascii=False, indent=2))
