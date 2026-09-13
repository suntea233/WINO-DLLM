#!/usr/bin/env python3
"""Combine all alpha/benchmark outputs into the requested result and report."""

import argparse
import json
from pathlib import Path


DATASETS = ("gsm8k", "math500", "humaneval", "mbpp")
ALPHAS = (0.25, 0.5, 0.75)


def tag(alpha):
    return str(alpha).replace(".", "p")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-dir", type=Path, required=True)
    args = ap.parse_args()
    results = {}
    for dataset in DATASETS:
        results[dataset] = {}
        for alpha in ALPHAS:
            path = args.result_dir / f"{dataset}_alpha_{tag(alpha)}.json"
            results[dataset][str(alpha)] = json.loads(path.read_text())

    rows = []
    for dataset in DATASETS:
        for alpha in ALPHAS:
            result = results[dataset][str(alpha)]
            m, c = result["metrics"], result["comparison"]
            p = c["paired_vs_v6"]
            rows.append({
                "dataset": dataset, "alpha": alpha,
                "accuracy": m["accuracy"], "correct": m["correct"],
                "total": m["total"], "average_nfe": m["average_nfe"],
                "average_latency": m["average_latency"],
                "v6_accuracy": c["v6_accuracy"],
                "v6_average_nfe": c["v6_average_nfe"],
                "delta_accuracy_vs_v6": c["delta_accuracy_vs_v6"],
                "delta_average_nfe_vs_v6": c["delta_average_nfe_vs_v6"],
                "recovered_vs_v6": p["recovered"],
                "regressed_vs_v6": p["regressed"],
                "net_recovery": p["net_recovery"],
                "interpolated_identity_commits": m[
                    "interpolated_identity_commits"],
            })
    # The explicit stop condition concerns paired outcomes: an alpha merely
    # loses V6-correct samples while recovering none on every task.
    only_reduces_recovery = all(
        row["recovered_vs_v6"] == 0 and row["regressed_vs_v6"] > 0
        for row in rows)
    output = {
        "experiment": "soft_interpolation_commitment",
        "configuration": {
            "baseline": "Soft Revision V6", "alphas": list(ALPHAS),
            "interpolation_lifetime_normal_forwards": 1,
            "soft_distribution": "current shadow posterior full expectation",
            "extra_backbone_forwards": 0,
        },
        "summary_rows": rows,
        "stop_condition": {
            "all_alphas_only_reduce_recovery_without_reducing_regression":
                only_reduces_recovery,
        },
    }
    (args.result_dir / "soft_interpolation_results.json").write_text(
        json.dumps(output, indent=2))
    lines = [
        "Soft Interpolation Commitment", "=============================", "",
        "Interpolation is applied for one normal forward after V6 accepts a",
        "different-identity correction. No extra backbone forward is used.", "",
        "Dataset       Alpha  Accuracy       Avg NFE  Delta Acc  Delta NFE  Rec/Reg/Net",
        "----------------------------------------------------------------------------",
    ]
    for row in rows:
        lines.append(
            f"{row['dataset']:<13} {row['alpha']:<5.2f}  "
            f"{row['accuracy']*100:6.2f}% ({row['correct']:>4}/{row['total']:<4}) "
            f"{row['average_nfe']:8.2f}  {row['delta_accuracy_vs_v6']*100:+8.2f}  "
            f"{row['delta_average_nfe_vs_v6']:+9.2f}  "
            f"{row['recovered_vs_v6']}/{row['regressed_vs_v6']}/{row['net_recovery']:+d}")
    lines += ["", f"Stop condition met: {only_reduces_recovery}",
              "Interpret paired differences descriptively; trajectory changes mean",
              "an interpolation event cannot be treated as an isolated causal intervention."]
    (args.result_dir / "report_soft_interpolation.txt").write_text(
        "\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
