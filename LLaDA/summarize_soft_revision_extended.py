"""Summarize frozen Soft Revision results across four benchmarks."""

import csv
import json
import os


ROOT = os.path.join(os.path.dirname(__file__), "results", "soft_revision")
DATASETS = ("gsm8k", "math500", "humaneval", "mbpp")


def load(dataset, method):
    with open(os.path.join(ROOT, f"{dataset}_{method}.json"), encoding="utf-8") as handle:
        return json.load(handle)


def main():
    analysis_path = os.path.join(ROOT, "gsm8k_revision_analysis.json")
    with open(analysis_path, encoding="utf-8") as handle:
        gsm_analysis = json.load(handle)
    summary_rows = []
    paired_rows = []
    for dataset in DATASETS:
        baseline, soft = load(dataset, "wino_remask"), load(dataset, "soft_revision")
        baseline_by_id = {row["index"]: row for row in baseline["samples"]}
        soft_by_id = {row["index"]: row for row in soft["samples"]}
        if set(baseline_by_id) != set(soft_by_id):
            raise RuntimeError(f"Mismatched sample IDs for {dataset}")
        recovered = sum(
            not baseline_by_id[i]["is_correct"] and soft_by_id[i]["is_correct"]
            for i in baseline_by_id)
        regressed = sum(
            baseline_by_id[i]["is_correct"] and not soft_by_id[i]["is_correct"]
            for i in baseline_by_id)
        bm, sm = baseline["metrics"], soft["metrics"]
        row = {
            "Dataset": dataset,
            "WINO Accuracy": bm["accuracy"],
            "Soft Revision Accuracy": sm["accuracy"],
            "Delta Accuracy": sm["accuracy"] - bm["accuracy"],
            "WINO Avg NFE": bm["average_nfe"],
            "Soft Revision Avg NFE": sm["average_nfe"],
            "Delta NFE": sm["average_nfe"] - bm["average_nfe"],
            "WINO Avg Latency": bm["average_latency"],
            "Soft Revision Avg Latency": sm["average_latency"],
            "Recovered": recovered,
            "Regressed": regressed,
            "Net Recovery": recovered - regressed,
        }
        summary_rows.append(row)
        paired_rows.append({
            "dataset": dataset,
            "recovered": recovered,
            "regressed": regressed,
            "net_recovery": recovered - regressed,
        })
    with open(os.path.join(ROOT, "summary_extended.csv"), "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    with open(os.path.join(ROOT, "paired_extended.csv"), "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired_rows[0]))
        writer.writeheader()
        writer.writerows(paired_rows)

    recovered_stats = gsm_analysis["group_statistics"]["recovered"]
    regressed_stats = gsm_analysis["group_statistics"]["regressed"]
    identity = gsm_analysis["hard_recovery_identity"]
    nfe = gsm_analysis["nfe_analysis"]
    lines = [
        "Frozen Soft Revision extended report",
        "====================================",
        "",
        "All runs use the official WINO task configuration with an unchanged decoder,",
        "thresholds, Soft representation, one-forward lifecycle, prompts, and evaluators.",
        "",
        "Benchmark results",
        "-----------------",
        "Dataset     WINO Acc  Soft Acc  Delta    WINO NFE  Soft NFE  Delta NFE  Recovered  Regressed  Net",
    ]
    for row in summary_rows:
        lines.append(
            f"{row['Dataset']:<11} {row['WINO Accuracy']:.4f}    "
            f"{row['Soft Revision Accuracy']:.4f}    {row['Delta Accuracy']:+.4f}   "
            f"{row['WINO Avg NFE']:.2f}     {row['Soft Revision Avg NFE']:.2f}     "
            f"{row['Delta NFE']:+.2f}      {row['Recovered']:>4}       "
            f"{row['Regressed']:>4}      {row['Net Recovery']:+d}")
        lines.append(
            f"  Avg latency: WINO {row['WINO Avg Latency']:.3f}s; "
            f"Soft {row['Soft Revision Avg Latency']:.3f}s")

    lines.extend([
        "",
        "GSM8K recovered versus regressed revision behavior",
        "--------------------------------------------------",
        f"Recovered samples: {recovered_stats['samples']}",
        f"  H->S total/mean: {recovered_stats['h_to_s']['total']} / {recovered_stats['h_to_s']['mean']:.2f}",
        f"  H->S->H total/mean: {recovered_stats['h_to_s_to_h']['total']} / {recovered_stats['h_to_s_to_h']['mean']:.2f}",
        f"  H->S->M total/mean: {recovered_stats['h_to_s_to_m']['total']} / {recovered_stats['h_to_s_to_m']['mean']:.2f}",
        f"  First revision round mean/median: {recovered_stats['first_revision_round']['mean']:.2f} / {recovered_stats['first_revision_round']['median']:.1f}",
        f"  Unique revised positions mean: {recovered_stats['unique_revised_positions']['mean']:.2f}",
        f"Regressed samples: {regressed_stats['samples']}",
        f"  H->S total/mean: {regressed_stats['h_to_s']['total']} / {regressed_stats['h_to_s']['mean']:.2f}",
        f"  H->S->H total/mean: {regressed_stats['h_to_s_to_h']['total']} / {regressed_stats['h_to_s_to_h']['mean']:.2f}",
        f"  H->S->M total/mean: {regressed_stats['h_to_s_to_m']['total']} / {regressed_stats['h_to_s_to_m']['mean']:.2f}",
        f"  First revision round mean/median: {regressed_stats['first_revision_round']['mean']:.2f} / {regressed_stats['first_revision_round']['median']:.1f}",
        f"  Unique revised positions mean: {regressed_stats['unique_revised_positions']['mean']:.2f}",
        "",
        f"Across all H->S->H events, old HARD == resulting HARD: {identity['old_hard_equals_new_hard']}; old HARD != resulting HARD: {identity['old_hard_differs_new_hard']}.",
        "The frozen pass path retains the stored HARD ID; it does not assign a new argmax.",
        "Regressions have more H->S->M events per sample than recoveries, while H->S->H",
        "counts are similar. This is descriptive and does not establish causation.",
        "",
        "Why average NFE decreases on GSM8K",
        "-----------------------------------",
        f"All samples: WINO {nfe['all_samples']['mean_wino_nfe']:.2f}, Soft {nfe['all_samples']['mean_soft_nfe']:.2f}, delta {nfe['all_samples']['mean_delta_nfe']:+.2f}.",
        f"Recovered: mean delta {nfe['recovered']['mean_delta_nfe']:+.2f}; regressed: {nfe['regressed']['mean_delta_nfe']:+.2f}.",
        f"After H->S->H resolution, mean subsequent rounds: {nfe['mean_remaining_rounds_after_soft_h_to_s_to_h']:.2f}.",
        f"After H->S->M resolution, mean subsequent rounds: {nfe['mean_remaining_rounds_after_soft_h_to_s_to_m']:.2f}.",
        f"Correlation between per-sample NFE delta and H->S->M count: {nfe['correlation_delta_nfe_vs_soft_h_to_s_to_m_count']:.3f}.",
        f"Correlation between NFE delta and (Soft H->S->M - WINO H->M): {nfe['correlation_delta_nfe_vs_revision_count_difference']:.3f}.",
        "The observed NFE reduction is consistent with Soft Revision producing fewer repeated",
        "revision cycles and shorter trajectories. Correlations alone do not establish cause.",
        "",
        "Trace limitation",
        "----------------",
        "The saved GSM8K trace does not contain per-round remaining MASK counts, other new",
        "HARD commits, or event timestamps for baseline WINO H->M. Exact event-aligned",
        "comparisons for those three fields cannot be reconstructed offline. They are reported",
        "as unavailable rather than inferred, and GSM8K generation was not rerun.",
    ])
    with open(os.path.join(ROOT, "report_soft_revision_extended.txt"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
