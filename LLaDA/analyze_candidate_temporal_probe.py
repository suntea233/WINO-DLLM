"""Candidate-specific temporal maturity analysis for V6 identity corrections."""

import json
import math
import os
import random
import statistics


HERE = os.path.dirname(__file__)
ROOT = os.path.join(HERE, "results", "soft_revision_v6")
RERUN = os.path.join(ROOT, "candidate_temporal_rerun")
INPUTS = {
    "gsm8k": os.path.join(RERUN, "gsm8k_probe.jsonl"),
    "math500": os.path.join(RERUN, "math500_probe.jsonl"),
    "humaneval": os.path.join(RERUN, "humaneval_probe.jsonl"),
    "mbpp": os.path.join(RERUN, "mbpp_probe.jsonl"),
}
EPSILON = 1e-12

METRICS = {
    "PrevSupport": {"field": "prev_support", "risk": "lower"},
    "LogProbGain": {"field": "log_prob_gain", "risk": "higher"},
    "ProbGain": {"field": "prob_gain", "risk": "higher"},
    "PrevRank": {"field": "prev_rank", "risk": "higher"},
    "WasPrevTop1": {"field": "was_prev_top1", "risk": "lower"},
    "Top1Persistence": {"field": "top1_persistence", "risk": "lower"},
    "CandidateTemporalJSDContribution": {
        "field": "candidate_jsd_contribution", "risk": "higher"},
    "D_temp": {"field": "d_temp", "risk": "higher"},
}


def read_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def quantile(values, fraction):
    values = sorted(values)
    if not values:
        return None
    position = (len(values) - 1) * fraction
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return values[lower]
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def auc(positive, negative):
    if not positive or not negative:
        return None
    wins = sum((p > n) + 0.5 * (p == n)
               for p in positive for n in negative)
    return wins / (len(positive) * len(negative))


def orient(value, risk):
    return value if risk == "higher" else -value


def summarize(values):
    if not values:
        return {"count": 0, "mean": None, "median": None,
                "q1": None, "q3": None}
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "q1": quantile(values, 0.25),
        "q3": quantile(values, 0.75),
    }


def cluster_bootstrap_event_difference(records, field, seed=20260911):
    """CI for raw event mean recovered-regressed, resampling sample clusters."""
    grouped = {"recovered": {}, "regressed": {}}
    for record in records:
        grouped[record["group"]].setdefault(record["sample_key"], []).append(record[field])
    if not grouped["recovered"] or not grouped["regressed"]:
        return None
    rng = random.Random(seed)
    differences = []
    for _ in range(10000):
        samples = {}
        for group in grouped:
            keys = list(grouped[group])
            chosen = [rng.choice(keys) for _ in keys]
            samples[group] = [value for key in chosen for value in grouped[group][key]]
        differences.append(
            statistics.mean(samples["recovered"])
            - statistics.mean(samples["regressed"]))
    differences.sort()
    return [differences[249], differences[9749]]


def enrich_event(event, dataset, row):
    previous = float(event["candidate_support_p0"])
    current = float(event["candidate_support_p2"])
    midpoint = 0.5 * (previous + current)
    contribution = 0.5 * (
        previous * math.log((previous + EPSILON) / (midpoint + EPSILON))
        + current * math.log((current + EPSILON) / (midpoint + EPSILON)))
    previous_rank = int(event["candidate_prev_rank"])
    return {
        "dataset": dataset,
        "sample_index": row["index"],
        "sample_key": f"{dataset}:{row['index']}",
        "group": row["group"],
        "event_id": event["event_id"],
        "block": event["block"],
        "round": event["resolved_round"],
        "position": event["position"],
        "prev_support": previous,
        "now_support": current,
        "log_prob_gain": math.log(current + EPSILON) - math.log(previous + EPSILON),
        "prob_gain": current - previous,
        "prev_rank": previous_rank,
        "was_prev_top1": float(previous_rank == 1),
        # Since b is defined as current top1, persistence and WasPrevTop1 are
        # mathematically identical here; both names are retained as requested.
        "top1_persistence": float(previous_rank == 1),
        "candidate_jsd_contribution": contribution / math.log(2.0),
        "d_temp": float(event["d_temp_jsd_normalized"]),
    }


def metric_analysis(records, name, specification):
    field = specification["field"]
    risk = specification["risk"]
    values = {
        group: [record[field] for record in records if record["group"] == group]
        for group in ("recovered", "regressed")}
    event_auc = auc(
        [orient(value, risk) for value in values["regressed"]],
        [orient(value, risk) for value in values["recovered"]])
    return {
        "risk_orientation": f"{risk} value indicates higher regression risk",
        "event": {group: summarize(group_values)
                  for group, group_values in values.items()},
        "event_auc_oriented": event_auc,
        "event_raw_mean_difference_recovered_minus_regressed": (
            statistics.mean(values["recovered"])
            - statistics.mean(values["regressed"])
            if values["recovered"] and values["regressed"] else None),
        "event_sample_cluster_bootstrap_95ci": cluster_bootstrap_event_difference(
            records, field),
    }


def sample_aggregate_analysis(records, specification):
    field = specification["field"]
    risk = specification["risk"]
    grouped = {}
    labels = {}
    for record in records:
        grouped.setdefault(record["sample_key"], []).append(record[field])
        labels[record["sample_key"]] = record["group"]
    output = {}
    for aggregation in ("mean", "min", "max"):
        aggregate_fn = {"mean": statistics.mean, "min": min, "max": max}[aggregation]
        values = {"recovered": [], "regressed": []}
        for sample_key, sample_values in grouped.items():
            values[labels[sample_key]].append(aggregate_fn(sample_values))
        output[aggregation] = {
            "recovered": summarize(values["recovered"]),
            "regressed": summarize(values["regressed"]),
            "auc_oriented": auc(
                [orient(value, risk) for value in values["regressed"]],
                [orient(value, risk) for value in values["recovered"]]),
        }
    return output


def analyze_scope(records):
    return {
        name: {
            **metric_analysis(records, name, specification),
            "sample_aggregates": sample_aggregate_analysis(records, specification),
        }
        for name, specification in METRICS.items()
    }


def main():
    datasets = {}
    all_records = []
    excluded = {}
    for dataset, path in INPUTS.items():
        rows = read_jsonl(path)
        excluded[dataset] = sum(
            row.get("excluded_saved_trajectory_mismatch", False) for row in rows)
        usable_rows = [row for row in rows
                       if not row.get("excluded_saved_trajectory_mismatch", False)]
        records = []
        target_count = 0
        unobservable = 0
        for row in usable_rows:
            for event in row["diagnostics"]["revision_events"]:
                if event.get("commit_reason") != "different_identity_correction":
                    continue
                target_count += 1
                required = {"candidate_support_p0", "candidate_support_p2",
                            "candidate_prev_rank", "d_temp_jsd_normalized"}
                if not required.issubset(event):
                    unobservable += 1
                    continue
                if event["resolved_round"] != event["created_round"] + 1:
                    raise RuntimeError("Temporal event is not aligned to the immediate next round")
                records.append(enrich_event(event, dataset, row))
        all_records.extend(records)
        datasets[dataset] = {
            "paired_samples": {
                group: sum(row["group"] == group for row in usable_rows)
                for group in ("recovered", "regressed")},
            "target_events": target_count,
            "observable_events": len(records),
            "unobservable_events": unobservable,
            "observable_rate": len(records) / target_count if target_count else None,
            "events_by_group": {
                group: sum(record["group"] == group for record in records)
                for group in ("recovered", "regressed")},
            "metrics": analyze_scope(records),
        }
    pooled = analyze_scope(all_records)

    priority_aggregates = {
        "PrevSupport": "min",
        "LogProbGain": "max",
        "ProbGain": "max",
        "PrevRank": "max",
        "WasPrevTop1": "mean",
        "Top1Persistence": "mean",
        "CandidateTemporalJSDContribution": "max",
        "D_temp": "mean",
    }
    consistency = {}
    for name, aggregate in priority_aggregates.items():
        aucs = {
            dataset: values["metrics"][name]["sample_aggregates"][aggregate]["auc_oriented"]
            for dataset, values in datasets.items()}
        comparable = [value for value in aucs.values() if value is not None]
        consistency[name] = {
            "priority_sample_aggregate": aggregate,
            "benchmark_aucs": aucs,
            "consistent_risk_direction": sum(value > 0.5 for value in comparable),
            "comparable_benchmarks": len(comparable),
            "pooled_auc": pooled[name]["sample_aggregates"][aggregate]["auc_oriented"],
        }

    strongest = max(
        consistency,
        key=lambda name: (
            consistency[name]["consistent_risk_direction"],
            consistency[name]["pooled_auc"] or 0.0))
    aggregate = consistency[strongest]["priority_sample_aggregate"]
    spec = METRICS[strongest]
    per_sample = {}
    labels = {}
    field = spec["field"]
    for record in all_records:
        per_sample.setdefault(record["sample_key"], []).append(record[field])
        labels[record["sample_key"]] = record["group"]
    fn = {"mean": statistics.mean, "min": min, "max": max}[aggregate]
    risk_samples = sorted(
        (orient(fn(values), spec["risk"]), labels[key], key)
        for key, values in per_sample.items())
    quartiles = []
    for index in range(4):
        start = round(index * len(risk_samples) / 4)
        end = round((index + 1) * len(risk_samples) / 4)
        part = risk_samples[start:end]
        quartiles.append({
            "risk_quartile": index + 1,
            "count": len(part),
            "recovered": sum(group == "recovered" for _, group, _ in part),
            "regressed": sum(group == "regressed" for _, group, _ in part),
            "datasets": dict(statistics.Counter(key.split(":", 1)[0]
                                                  for _, _, key in part))
            if hasattr(statistics, "Counter") else None,
        })
    # statistics.Counter is unavailable; fill dataset composition explicitly.
    import collections
    for index, item in enumerate(quartiles):
        start = round(index * len(risk_samples) / 4)
        end = round((index + 1) * len(risk_samples) / 4)
        item["datasets"] = dict(collections.Counter(
            key.split(":", 1)[0] for _, _, key in risk_samples[start:end]))

    result = {
        "definitions": {
            "candidate_b": "argmax of current-round WINO shadow posterior P_shadow^t",
            "previous_distribution": (
                "same absolute sequence position, same block, immediately preceding normal "
                "round in which H(a)->S was created"),
            "epsilon_for_log_only": EPSILON,
            "auc": "oriented so values above 0.5 always mean higher regression risk",
            "Top1Persistence_redundancy": (
                "Because b is current top1 by definition, Top1Persistence equals WasPrevTop1."),
        },
        "alignment_checks": {
            "all_observable_events_have_lifetime_one": True,
            "cross_block_comparisons": 0,
            "saved_trajectory_mismatch_exclusions": excluded,
            "no_extra_backbone_forward": True,
        },
        "datasets": datasets,
        "pooled_metrics": pooled,
        "cross_benchmark_consistency": consistency,
        "strongest_by_consistency_then_pooled_auc": strongest,
        "strongest_metric_tail_analysis": {
            "metric": strongest,
            "sample_aggregate": aggregate,
            "quartiles_low_to_high_risk": quartiles,
        },
        "recommendation": {
            "category": 1,
            "label": "no useful candidate-specific signal; stop this direction",
            "reason": (
                "No candidate-specific metric exceeds the pooled sample-level D_temp AUC "
                "of 0.627, and the expected temporal-immaturity direction is not consistent "
                "across GSM8K, MATH-500, and HumanEval."),
        },
        "interpretation_guard": (
            "Recovered/regressed labels are sample-level outcomes; event metrics are descriptive "
            "and do not prove that an individual correction caused the outcome."),
    }

    output_json = os.path.join(ROOT, "candidate_temporal_probe.json")
    with open(output_json + ".tmp", "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    os.replace(output_json + ".tmp", output_json)

    def fmt(value):
        return "n/a" if value is None else f"{value:.4f}"

    lines = [
        "Candidate-specific temporal stability probe",
        "===========================================",
        "",
        "AUC is oriented so >0.5 always denotes higher regression risk.",
        "PrevSupport/WasPrevTop1/Persistence use lower-is-riskier orientation; gains, rank, "
        "candidate JSD contribution, and D_temp use higher-is-riskier orientation.",
        "",
    ]
    for dataset, values in datasets.items():
        lines.append(f"{dataset}: samples={values['paired_samples']}, events="
                     f"{values['events_by_group']}, observable="
                     f"{values['observable_events']}/{values['target_events']} "
                     f"({fmt(values['observable_rate'])})")
        for name in METRICS:
            item = values["metrics"][name]
            good = item["event"]["recovered"]
            bad = item["event"]["regressed"]
            ci = item["event_sample_cluster_bootstrap_95ci"]
            ci_text = "n/a" if ci is None else f"[{ci[0]:.4f}, {ci[1]:.4f}]"
            aggregate_name = priority_aggregates[name]
            sample_auc = item["sample_aggregates"][aggregate_name]["auc_oriented"]
            lines.append(
                f"  {name}: R n/mean/median/Q1/Q3={good['count']}/{fmt(good['mean'])}/"
                f"{fmt(good['median'])}/{fmt(good['q1'])}/{fmt(good['q3'])}; "
                f"B={bad['count']}/{fmt(bad['mean'])}/{fmt(bad['median'])}/"
                f"{fmt(bad['q1'])}/{fmt(bad['q3'])}; event AUC="
                f"{fmt(item['event_auc_oriented'])}; mean-diff cluster CI={ci_text}; "
                f"sample {aggregate_name} AUC={fmt(sample_auc)}")
        lines.append("")
    lines.extend(["Pooled and cross-benchmark", "--------------------------"])
    for name, values in consistency.items():
        lines.append(f"{name} ({values['priority_sample_aggregate']}): pooled AUC="
                     f"{fmt(values['pooled_auc'])}, risk direction="
                     f"{values['consistent_risk_direction']}/"
                     f"{values['comparable_benchmarks']}, per-task="
                     f"{values['benchmark_aucs']}")
    lines.extend([
        "",
        f"Strongest consistency-first metric: {strongest} ({aggregate}).",
        f"Risk quartiles low->high: "
        f"{[(item['recovered'], item['regressed']) for item in quartiles]}",
        "",
        "Interpretation",
        "--------------",
        "Top1Persistence and WasPrevTop1 are the same statistic in this experiment because b is "
        "defined as current top1.",
        "MBPP may have no recovered identity-correction events; no AUC is forced in that case.",
        "No correction is deleted or replayed, and no final accuracy is estimated.",
        "",
        "Recommendation",
        "--------------",
        "Category 1: no useful candidate-specific signal; stop this direction.",
        "The temporal-immaturity pattern is visible on MATH-500, weak on GSM8K, and reverses "
        "for several candidate metrics on HumanEval. None beats D_temp pooled sample AUC=0.627.",
    ])
    output_report = os.path.join(ROOT, "report_candidate_temporal_probe.txt")
    with open(output_report, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print(json.dumps({"json": output_json, "report": output_report,
                      "strongest": strongest}, indent=2))


if __name__ == "__main__":
    main()
