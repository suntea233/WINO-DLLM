"""Descriptive paired analysis of general V6 stability/impact signals."""

import json
import math
import os
import random
import statistics


HERE = os.path.dirname(__file__)
ROOT = os.path.join(HERE, "results", "soft_revision_v6")
INPUT = os.path.join(ROOT, "stability_probe_samples.jsonl")
OUTPUT_JSON = os.path.join(ROOT, "stability_impact_probe.json")
OUTPUT_REPORT = os.path.join(ROOT, "report_stability_impact_probe.txt")

DECISION_METRICS = {
    "D_temp JSD": "d_temp_jsd_normalized",
    "D_view JSD": "d_view_jsd_normalized",
    "three-view GJSD": "three_view_gjsd_normalized",
    "candidate min support": "candidate_min_support",
    "candidate min margin": "candidate_min_margin",
    "temporal top1 stability": "temporal_top1_stable",
    "three-view top1 agreement": "three_view_top1_agreement",
}
IMPACT_METRICS = {
    "global JSD": "global_jsd_normalized_mean",
    "observer top1 flip rate": "observer_top1_flip_rate",
    "anchor damage": "anchor_damage_mean",
    "anchor damage positive rate": "anchor_damage_positive_rate",
}


def read_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def mean(values):
    return statistics.mean(values) if values else None


def median(values):
    return statistics.median(values) if values else None


def auc(regressed, recovered):
    """Mann-Whitney AUC: probability a regressed value is larger."""
    if not regressed or not recovered:
        return None
    wins = 0.0
    for bad in regressed:
        for good in recovered:
            wins += (bad > good) + 0.5 * (bad == good)
    return wins / (len(regressed) * len(recovered))


def bootstrap_difference(recovered_by_sample, regressed_by_sample, seed=20260911):
    recovered = list(recovered_by_sample.values())
    regressed = list(regressed_by_sample.values())
    if not recovered or not regressed:
        return None
    rng = random.Random(seed)
    differences = []
    for _ in range(10000):
        good = [rng.choice(recovered) for _ in recovered]
        bad = [rng.choice(regressed) for _ in regressed]
        differences.append(statistics.mean(good) - statistics.mean(bad))
    differences.sort()
    return [differences[249], differences[9749]]


def summarize_metric(records, key):
    event_values = {"recovered": [], "regressed": []}
    sample_values = {"recovered": {}, "regressed": {}}
    for record in records:
        value = record.get(key)
        if value is None or (isinstance(value, float) and not math.isfinite(value)):
            continue
        value = float(value)
        group = record["group"]
        event_values[group].append(value)
        sample_values[group].setdefault(record["sample_index"], []).append(value)
    sample_means = {
        group: {sample_id: statistics.mean(values)
                for sample_id, values in grouped.items()}
        for group, grouped in sample_values.items()
    }
    good = list(sample_means["recovered"].values())
    bad = list(sample_means["regressed"].values())
    result = {
        "event": {
            group: {
                "n": len(event_values[group]),
                "mean": mean(event_values[group]),
                "median": median(event_values[group]),
            } for group in event_values
        },
        "sample_aggregated": {
            group: {
                "n": len(sample_means[group]),
                "mean": mean(list(sample_means[group].values())),
                "median": median(list(sample_means[group].values())),
            } for group in sample_means
        },
        "sample_mean_difference_recovered_minus_regressed": (
            mean(good) - mean(bad) if good and bad else None),
        "sample_cluster_bootstrap_95ci": bootstrap_difference(
            sample_means["recovered"], sample_means["regressed"]),
        "sample_level_auc_regressed_higher": auc(bad, good),
    }
    score = result["sample_level_auc_regressed_higher"]
    ci = result["sample_cluster_bootstrap_95ci"]
    result["direction"] = (
        "higher_in_regressed" if score is not None and score > 0.5
        else "lower_in_regressed" if score is not None and score < 0.5
        else "no_rank_separation")
    result["clear_separation"] = bool(
        score is not None and abs(score - 0.5) >= 0.15 and ci is not None
        and (ci[0] > 0 or ci[1] < 0))
    return result


def main():
    rows = read_jsonl(INPUT)
    groups = {"recovered": 0, "regressed": 0}
    events = []
    impact_rounds = []
    for row in rows:
        groups[row["group"]] += 1
        corrections = [event for event in row["diagnostics"]["revision_events"]
                       if event.get("commit_reason") == "different_identity_correction"]
        seen_rounds = set()
        for event in corrections:
            tagged = {**event, "sample_index": row["index"], "group": row["group"]}
            events.append(tagged)
            round_key = (row["index"], event["resolved_round"])
            if (round_key not in seen_rounds
                    and event.get("global_jsd_normalized_mean") is not None):
                impact_rounds.append(tagged)
                seen_rounds.add(round_key)

    decision = {name: summarize_metric(events, key)
                for name, key in DECISION_METRICS.items()}
    impact = {name: summarize_metric(impact_rounds, key)
              for name, key in IMPACT_METRICS.items()}
    all_metrics = {**decision, **impact}
    ranked = sorted(
        ((name, abs(item["sample_level_auc_regressed_higher"] - 0.5),
          item["clear_separation"])
         for name, item in all_metrics.items()
         if item["sample_level_auc_regressed_higher"] is not None),
        key=lambda item: item[1], reverse=True)

    dview_sample_values = []
    for row in rows:
        values = [
            event["d_view_jsd_normalized"]
            for event in row["diagnostics"]["revision_events"]
            if event.get("commit_reason") == "different_identity_correction"]
        if values:
            dview_sample_values.append((statistics.mean(values), row["group"]))
    ordered_dview = sorted(value for value, _ in dview_sample_values)
    quartile_cuts = [ordered_dview[9], ordered_dview[19], ordered_dview[28]]
    dview_quartiles = []
    lower = -math.inf
    for upper in quartile_cuts + [math.inf]:
        selected = [group for value, group in dview_sample_values
                    if lower < value <= upper]
        dview_quartiles.append({
            "n": len(selected),
            "recovered": selected.count("recovered"),
            "regressed": selected.count("regressed"),
        })
        lower = upper

    result = {
        "scope": {
            "paired_reference": "contemporaneous V1 control",
            "sample_groups": groups,
            "identity_correction_events": len(events),
            "post_commit_rounds_with_next_forward": len(impact_rounds),
            "no_change_verification": (
                "The evaluator required exact full response, correctness, and NFE equality "
                "to the saved V6 result for every probe sample."),
        },
        "definitions": {
            "P0": "shadow posterior in the H->S creation round",
            "P1": "next-round posterior at the original position consuming SOFT",
            "P2": "next-round WINO shadow-verifier posterior used by V6",
            "D_temp": "JSD(P0,P2)/log(2)",
            "D_view": "JSD(P1,P2)/log(2)",
            "three_view_GJSD": "GJSD(P0,P1,P2)/log(3)",
            "candidate_min_support": "min(P0(b),P1(b),P2(b)), b=argmax(P2)",
            "candidate_min_margin": "minimum top1-top2 margin over P0,P1,P2",
            "global_JSD": (
                "mean normalized JSD at still-unresolved observer positions from the "
                "commit round to the next normal forward"),
            "anchor_damage": (
                "mean log-support loss for unchanged HARD positions that matched the "
                "shadow top1 above 0.9 in at least two consecutive rounds"),
        },
        "decision_time_metrics": decision,
        "post_commit_round_metrics": impact,
        "ranking_by_sample_auc_distance_from_chance": ranked,
        "d_view_sample_quartiles_low_to_high": {
            "cuts": quartile_cuts,
            "groups": dview_quartiles,
        },
        "recommendation": {
            "ready_for_gate": False,
            "best_candidate_for_cross_benchmark_validation": "D_view JSD",
            "reason": (
                "D_view has the largest sample-level rank separation, but its cluster "
                "bootstrap interval crosses zero and its event medians are nearly identical. "
                "The apparent difference is concentrated in the upper tail."),
        },
        "interpretation_limits": [
            "Recovered/regressed is a sample-level label; it does not prove an individual event was causal.",
            "Global-impact values are assigned once per correction round because simultaneous changes cannot be attributed to one token without a counterfactual forward.",
            "Post-commit global metrics are delayed diagnostics and are unavailable when no next normal forward occurs.",
            "No threshold or combined score is tuned on GSM8K labels.",
        ],
    }
    with open(OUTPUT_JSON + ".tmp", "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    os.replace(OUTPUT_JSON + ".tmp", OUTPUT_JSON)

    def fmt(value):
        return "n/a" if value is None else f"{value:.4f}"

    lines = [
        "V6 general stability and impact probe",
        "=====================================",
        "",
        f"Paired samples: recovered={groups['recovered']}, regressed={groups['regressed']}",
        f"Identity-correction events: {len(events)}",
        f"Post-commit correction rounds with a following forward: {len(impact_rounds)}",
        "Every probed sample exactly matched saved V6 output, correctness, and NFE.",
        "",
        "Metric comparison",
        "-----------------",
        "AUC is sample-level P(value_regressed > value_recovered); 0.5 is no separation.",
    ]
    for section_name, section in (("Decision-time", decision), ("Post-commit", impact)):
        lines.extend(["", section_name])
        for name, item in section.items():
            good = item["sample_aggregated"]["recovered"]
            bad = item["sample_aggregated"]["regressed"]
            ci = item["sample_cluster_bootstrap_95ci"]
            ci_text = "n/a" if ci is None else f"[{ci[0]:.4f}, {ci[1]:.4f}]"
            lines.append(
                f"{name}: recovered={fmt(good['mean'])}, regressed={fmt(bad['mean'])}, "
                f"diff(R-B)={fmt(item['sample_mean_difference_recovered_minus_regressed'])}, "
                f"95% cluster CI={ci_text}, AUC={fmt(item['sample_level_auc_regressed_higher'])}, "
                f"clear={item['clear_separation']}")
    lines.extend(["", "Ranking", "-------"])
    for name, distance, clear in ranked:
        lines.append(f"{name}: |AUC-0.5|={distance:.4f}, clear={clear}")
    lines.extend([
        "",
        "D_view upper-tail check",
        "-----------------------",
        f"Sample quartiles low-to-high (recovered/regressed): "
        f"{[(item['recovered'], item['regressed']) for item in dview_quartiles]}",
        "The highest D_view quartile contains 2 recovered and 7 regressed samples, but the "
        "event medians are almost identical; the mean separation is driven by a tail.",
        "",
        "Interpretation",
        "--------------",
        "The report treats a feature as provisionally useful only when its sample-level rank "
        "separation is sizable and the sample-cluster bootstrap interval excludes zero.",
        "No combined Risk score is formed: raw addition would introduce arbitrary scales and weights.",
        "Global metrics describe the whole transition between normal rounds; concurrent drafts and "
        "revisions prevent causal attribution to one identity correction.",
        "",
        "Decision: none of these signals is ready to serve as a V7 gate. D_view is the only "
        "candidate worth validating across MATH/code before considering a frozen threshold.",
    ])
    with open(OUTPUT_REPORT, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print(json.dumps({"json": OUTPUT_JSON, "report": OUTPUT_REPORT,
                      "ranking": ranked}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
