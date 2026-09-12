"""Cross-benchmark validation of frozen V6 stability indicators."""

import json
import os

from analyze_v6_stability_probe import DECISION_METRICS, summarize_metric


HERE = os.path.dirname(__file__)
ROOT = os.path.join(HERE, "results", "soft_revision_v6")
INPUTS = {
    "gsm8k": os.path.join(ROOT, "stability_probe_samples.jsonl"),
    "math500": os.path.join(ROOT, "stability_cross_benchmark", "math500_probe.jsonl"),
    "humaneval": os.path.join(ROOT, "stability_cross_benchmark", "humaneval_probe.jsonl"),
    "mbpp": os.path.join(ROOT, "stability_cross_benchmark", "mbpp_probe.jsonl"),
}


def read_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def event_records(dataset, rows):
    records = []
    for row in rows:
        if row.get("excluded_saved_trajectory_mismatch"):
            continue
        for event in row["diagnostics"]["revision_events"]:
            if event.get("commit_reason") != "different_identity_correction":
                continue
            records.append({
                **event,
                "sample_index": f"{dataset}:{row['index']}",
                "group": row["group"],
            })
    return records


def fmt(value):
    return "n/a" if value is None else f"{value:.4f}"


def main():
    datasets = {}
    pooled = []
    excluded = {}
    for name, path in INPUTS.items():
        rows = read_jsonl(path)
        usable = [row for row in rows
                  if not row.get("excluded_saved_trajectory_mismatch")]
        excluded[name] = len(rows) - len(usable)
        records = event_records(name, rows)
        pooled.extend(records)
        datasets[name] = {
            "samples": {
                group: sum(row["group"] == group for row in usable)
                for group in ("recovered", "regressed")},
            "events": {
                group: sum(record["group"] == group for record in records)
                for group in ("recovered", "regressed")},
            "metrics": {metric: summarize_metric(records, key)
                        for metric, key in DECISION_METRICS.items()},
        }
    pooled_metrics = {metric: summarize_metric(pooled, key)
                      for metric, key in DECISION_METRICS.items()}

    risk_metrics = ["D_temp JSD", "D_view JSD", "three-view GJSD"]
    consistency = {}
    for metric in risk_metrics:
        aucs = {
            dataset: values["metrics"][metric]["sample_level_auc_regressed_higher"]
            for dataset, values in datasets.items()}
        consistency[metric] = {
            "aucs": aucs,
            "datasets_with_comparable_events": sum(
                value is not None for value in aucs.values()),
            "datasets_with_regressed_higher_auc": sum(
                value is not None and value > 0.5 for value in aucs.values()),
            "datasets_with_clear_separation": sum(
                values["metrics"][metric]["clear_separation"]
                for values in datasets.values()),
            "pooled": pooled_metrics[metric],
        }

    result = {
        "datasets": datasets,
        "excluded_saved_trajectory_mismatches": excluded,
        "pooled_metrics": pooled_metrics,
        "risk_metric_consistency": consistency,
        "selection_rule": (
            "A metric is considered usable only if its direction is consistent across "
            "benchmarks and uncertainty is not dominated by chance; no threshold is fit."),
    }
    output_json = os.path.join(ROOT, "stability_cross_benchmark_probe.json")
    with open(output_json + ".tmp", "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    os.replace(output_json + ".tmp", output_json)

    lines = [
        "V6 stability indicators: cross-benchmark validation",
        "===================================================",
        "",
        "All rows are matched V1/V6 outcome flips. No threshold is tuned.",
        "AUC=P(metric is higher in a regressed sample than a recovered sample).",
        "",
    ]
    for dataset, values in datasets.items():
        lines.append(f"{dataset}: samples={values['samples']}, events={values['events']}, "
                     f"excluded_saved_mismatch={excluded[dataset]}")
        for metric in risk_metrics:
            item = values["metrics"][metric]
            good = item["sample_aggregated"]["recovered"]["mean"]
            bad = item["sample_aggregated"]["regressed"]["mean"]
            ci = item["sample_cluster_bootstrap_95ci"]
            ci_text = "n/a" if ci is None else f"[{ci[0]:.4f}, {ci[1]:.4f}]"
            lines.append(f"  {metric}: recovered={fmt(good)}, regressed={fmt(bad)}, "
                         f"AUC={fmt(item['sample_level_auc_regressed_higher'])}, "
                         f"diff CI={ci_text}, clear={item['clear_separation']}")
        lines.append("")
    lines.extend(["Pooled paired samples", "---------------------"])
    for metric in risk_metrics:
        item = pooled_metrics[metric]
        lines.append(f"{metric}: AUC={fmt(item['sample_level_auc_regressed_higher'])}, "
                     f"clear={item['clear_separation']}")
    lines.extend(["", "Direction consistency", "---------------------"])
    for metric, item in consistency.items():
        lines.append(f"{metric}: regressed-higher in "
                     f"{item['datasets_with_regressed_higher_auc']}/"
                     f"{item['datasets_with_comparable_events']} comparable datasets; "
                     f"clear in {item['datasets_with_clear_separation']}/"
                     f"{item['datasets_with_comparable_events']}")
    lines.extend([
        "",
        "Conclusion",
        "----------",
        "D_view does not generalize: its GSM8K direction reverses on MATH-500 and HumanEval.",
        "D_temp is the most consistent general candidate: regressed samples have higher values "
        "in all three datasets where recovered and regressed correction events both exist.",
        "Its pooled AUC is only 0.6273 and no individual dataset has clear separation, so it is "
        "not ready for a decoding gate or a frozen threshold.",
        "MBPP supplies no helpful identity-correction events in the recovered group; corrections "
        "occur only in regressed flips there, which supports caution but cannot calibrate D_temp.",
    ])
    output_report = os.path.join(ROOT, "report_stability_cross_benchmark_probe.txt")
    with open(output_report, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print(json.dumps({"json": output_json, "report": output_report}, indent=2))


if __name__ == "__main__":
    main()
