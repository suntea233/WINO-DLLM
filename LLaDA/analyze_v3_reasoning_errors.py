#!/usr/bin/env python3
"""Offline paired reasoning-error analysis for Soft Revision V1 versus V3."""

import json
import statistics
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
V1_PATH = ROOT / "results/soft_revision/gsm8k_soft_revision.json"
V3_PATH = ROOT / "results/soft_revision_v3/gsm8k_soft_revision_v3.json"
OUT_DIR = ROOT / "results/soft_revision_v3"


# Manual review of every V1-correct -> V3-wrong sample.  Categories identify the
# dominant visible failure locus; several responses contain more than one error.
DIAGNOSES = {
    1: ("truncated_or_incomplete", "Output ends before the remaining white-fiber calculation and final answer."),
    14: ("arithmetic_execution", "Uses 20-4-4=8 instead of 12."),
    64: ("arithmetic_execution", "Uses 12/0.04=30 instead of 300."),
    97: ("semantic_or_bookkeeping", "Confuses the 16-ounce can quantity with the three-tomato relation."),
    125: ("arithmetic_execution", "Uses 12-2=8 instead of 10."),
    132: ("arithmetic_execution", "Uses 10*6=120 instead of 60."),
    150: ("arithmetic_execution", "Uses 40-36=40 instead of 4."),
    153: ("semantic_or_bookkeeping", "Reports the 18 present before the raccoon and drops earlier eaten/stolen apples."),
    161: ("semantic_or_bookkeeping", "Drops two Locsin animal categories from the final total."),
    172: ("arithmetic_execution", "Adds 8+8+16+19 as 41 instead of 51."),
    174: ("final_answer_or_unit", "Derives 5700 seconds = 95 minutes, then boxes 5700 although the requested unit is minutes."),
    175: ("semantic_or_bookkeeping", "Confuses profit, revenue, per-gallon cost, and the gallon count."),
    210: ("semantic_or_bookkeeping", "Subtracts the saved and babysitting amounts inconsistently when computing the remaining cost."),
    249: ("arithmetic_execution", "Uses 9200-3600=1600 instead of 5600."),
    278: ("semantic_or_bookkeeping", "Adds the extra 10 only once rather than once for each of two people."),
    284: ("semantic_or_bookkeeping", "Duplicates and omits weekly terms while aggregating the schedule."),
    310: ("semantic_or_bookkeeping", "Fails to subtract the trip expenses after accounting for savings."),
    354: ("arithmetic_execution", "Uses 400-80-275=145 instead of 45."),
    357: ("semantic_or_bookkeeping", "Mishandles the shared party cost and the number of people."),
    361: ("arithmetic_execution", "Uses 18000/900=2 instead of 20."),
    369: ("arithmetic_execution", "Contains multiple frame-cost and total-addition errors."),
    386: ("arithmetic_execution", "Breaks the cumulative sum at 65+94, leading to 78 instead of 80."),
    425: ("truncated_or_incomplete", "Stops after computing subtotals 90 and 60, before adding them and answering."),
    518: ("semantic_or_bookkeeping", "Calls 4*10+6*20 equal to 260, corrupting the money total and downstream count."),
    554: ("arithmetic_execution", "Uses the wrong per-mile time in the warm segment and also gives an inconsistent product."),
    600: ("arithmetic_execution", "Adds 2 where the story requires subtracting 2."),
    649: ("semantic_or_bookkeeping", "Drops the initial population term required by the reference aggregation."),
    683: ("semantic_or_bookkeeping", "Converts one hour to 120 minutes instead of 60."),
    684: ("arithmetic_execution", "After identifying 30 total and 19 elapsed, outputs 1 instead of 11."),
    707: ("final_answer_or_unit", "The reasoning reaches 8 percent but the boxed answer is 12."),
    717: ("truncated_or_incomplete", "Stops before computing the change and gives no completed answer."),
    855: ("arithmetic_execution", "Combines constants as 6B-19 rather than 6B-9."),
    878: ("arithmetic_execution", "Computes 9 for each boy, then uses 14-4 rather than the derived value."),
    932: ("semantic_or_bookkeeping", "Double-counts the initial balloons inside the mother's allocation."),
    996: ("semantic_or_bookkeeping", "Introduces Helene but omits her from the final age equation."),
    1026: ("final_answer_or_unit", "Correctly derives 43,200, then boxes 432,000."),
    1037: ("semantic_or_bookkeeping", "Computes the 40-dollar cost but omits the final 50-40 operation and boxes 0."),
    1044: ("semantic_or_bookkeeping", "Applies Cole's growth but omits Xavier's increase by 3."),
    1071: ("semantic_or_bookkeeping", "Uses 41-20=11, losing ten additional guests."),
    1102: ("arithmetic_execution", "Uses 70+16+27=93 instead of 113."),
    1118: ("semantic_or_bookkeeping", "Computes guppy and goldfish components but drops the goldfish terms in the final comparison."),
    1119: ("semantic_or_bookkeeping", "Introduces an unsupported reinterpretation that changes Dior's stated quantity."),
    1123: ("semantic_or_bookkeeping", "Double-counts Kory's amount while summing the dogs."),
    1128: ("arithmetic_execution", "Uses 7+2=10 instead of 9."),
    1142: ("final_answer_or_unit", "The reasoning computes 60-51=9, then boxes 19."),
    1155: ("semantic_or_bookkeeping", "Treats a 60-percent increase as the total duration rather than adding it to the base."),
    1187: ("arithmetic_execution", "Uses the wrong melt count and also evaluates 60/16 as 30."),
    1248: ("semantic_or_bookkeeping", "Drops the two-customers-per-car conversion."),
    1299: ("arithmetic_execution", "Uses 40-27=3 instead of 13."),
}


def first_word_difference(a: str, b: str):
    aw, bw = a.split(), b.split()
    i = 0
    while i < min(len(aw), len(bw)) and aw[i] == bw[i]:
        i += 1
    return i, i / max(1, min(len(aw), len(bw)))


def mean(xs):
    return statistics.mean(xs) if xs else None


def main():
    v1_raw = json.loads(V1_PATH.read_text())
    v3_raw = json.loads(V3_PATH.read_text())
    v1 = {int(s["index"]): s for s in v1_raw["samples"]}
    v3 = {int(s["index"]): s for s in v3_raw["samples"]}
    ids = sorted(set(v1) & set(v3))

    groups = {
        "both_correct": [],
        "v1_correct_v3_wrong": [],
        "v1_wrong_v3_correct": [],
        "both_wrong": [],
    }
    for i in ids:
        a, b = bool(v1[i]["is_correct"]), bool(v3[i]["is_correct"])
        key = ("both_correct" if a and b else "v1_correct_v3_wrong" if a else
               "v1_wrong_v3_correct" if b else "both_wrong")
        groups[key].append(i)

    regressed = groups["v1_correct_v3_wrong"]
    if set(regressed) != set(DIAGNOSES):
        raise RuntimeError(f"Manual review IDs do not match regressions: {set(regressed) ^ set(DIAGNOSES)}")

    def group_stats(group_ids):
        ds = [v3[i]["diagnostics"] for i in group_ids]
        prefix = [first_word_difference(v1[i]["full_response"], v3[i]["full_response"]) for i in group_ids]
        return {
            "samples": len(group_ids),
            "mean_h_to_s": mean([d["num_h_to_soft_revisions"] for d in ds]),
            "mean_s_to_s": mean([d.get("num_s_to_s", 0) for d in ds]),
            "mean_s_to_h": mean([d.get("num_s_to_h", 0) for d in ds]),
            "mean_s_to_m": mean([d.get("num_s_to_m", 0) for d in ds]),
            "mean_fixed_cycle_fallbacks": mean([d.get("num_stalled_soft_to_mask_fallbacks", 0) for d in ds]),
            "mean_nfe": mean([s["nfe"] for s in (v3[i] for i in group_ids)]),
            "mean_first_revision_round": mean([
                min((e["created_round"] for e in d.get("revision_events", [])), default=0) for d in ds
            ]),
            "median_common_prefix_words": statistics.median([p[0] for p in prefix]) if prefix else None,
            "median_common_prefix_fraction": statistics.median([p[1] for p in prefix]) if prefix else None,
        }

    category_counts = Counter(DIAGNOSES[i][0] for i in regressed)
    reviewed = []
    for i in regressed:
        category, note = DIAGNOSES[i]
        d = v3[i]["diagnostics"]
        prefix_words, prefix_fraction = first_word_difference(v1[i]["full_response"], v3[i]["full_response"])
        reviewed.append({
            "sample_index": i,
            "category": category,
            "diagnosis": note,
            "ground_truth": v1[i].get("ground_truth"),
            "v1_prediction": v1[i].get("prediction"),
            "v3_prediction": v3[i].get("prediction"),
            "v3_nfe": v3[i]["nfe"],
            "h_to_s": d["num_h_to_soft_revisions"],
            "s_to_s": d.get("num_s_to_s", 0),
            "s_to_h": d.get("num_s_to_h", 0),
            "s_to_m": d.get("num_s_to_m", 0),
            "common_prefix_words": prefix_words,
            "common_prefix_fraction": prefix_fraction,
        })

    result = {
        "scope": "GSM8K paired Soft Revision V1 versus renewable V3",
        "paired_outcomes": {k: len(v) for k, v in groups.items()},
        "net_recovery_v3_minus_v1": len(groups["v1_wrong_v3_correct"]) - len(regressed),
        "v1_accuracy": v1_raw["metrics"].get("accuracy"),
        "v3_accuracy": v3_raw["metrics"].get("accuracy"),
        "group_statistics": {k: group_stats(v) for k, v in groups.items()},
        "regression_error_categories": {
            k: {"count": n, "percentage": 100 * n / len(regressed)}
            for k, n in sorted(category_counts.items())
        },
        "reviewed_regressions": reviewed,
        "limitations": [
            "Categories are manual dominant-error labels and can overlap in reality.",
            "Generated text exposes the visible failure locus, not the exact internal causal token intervention.",
            "Revision frequency is observational and should not be interpreted as causal.",
        ],
    }
    out_json = OUT_DIR / "reasoning_error_analysis_v3.json"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")

    gs = result["group_statistics"]
    lines = [
        "Soft Revision V3 GSM8K: paired reasoning-error analysis",
        "========================================================",
        "",
        f"V1 accuracy: {100 * result['v1_accuracy']:.2f}%",
        f"V3 accuracy: {100 * result['v3_accuracy']:.2f}%",
        f"V1 correct -> V3 wrong: {len(regressed)}",
        f"V1 wrong -> V3 correct: {len(groups['v1_wrong_v3_correct'])}",
        f"Net recovery: {result['net_recovery_v3_minus_v1']}",
        "",
        "Dominant visible failure locus in the 49 regressions",
        "-----------------------------------------------------",
    ]
    for key in ["arithmetic_execution", "semantic_or_bookkeeping", "final_answer_or_unit", "truncated_or_incomplete"]:
        x = result["regression_error_categories"][key]
        lines.append(f"{key}: {x['count']} ({x['percentage']:.2f}%)")
    lines += [
        "",
        "Intervention diagnostics (means per sample)",
        "-------------------------------------------",
    ]
    for key in ["v1_correct_v3_wrong", "v1_wrong_v3_correct", "both_correct", "both_wrong"]:
        x = gs[key]
        lines.append(
            f"{key}: n={x['samples']}, H->S={x['mean_h_to_s']:.2f}, "
            f"S->S={x['mean_s_to_s']:.2f}, S->H={x['mean_s_to_h']:.2f}, "
            f"S->M={x['mean_s_to_m']:.2f}, NFE={x['mean_nfe']:.2f}"
        )
    lines += [
        "",
        "Interpretation",
        "--------------",
        "The largest visible error classes are semantic/bookkeeping failures and local arithmetic failures.",
        "Four regressions reach the correct numeric result or conversion in the reasoning but corrupt the boxed answer/unit; three terminate before completing the calculation.",
        "Recovered samples have at least as many H->S and S->S operations as regressed samples. Therefore revision count alone does not separate helpful from harmful trajectory changes.",
        "Both changed-outcome groups are revised much more often than stable-correct samples, which indicates that the mechanism is most active on unstable/harder trajectories.",
        "The output-level evidence is consistent with renewable SOFT states perturbing intermediate operands, operators, and bookkeeping, after which later text remains locally coherent around the wrong state.",
        "This is descriptive evidence: the saved traces do not identify a unique revision event as the cause of a particular reasoning error.",
        "",
        "Representative regressions",
        "--------------------------",
        "14: 20-4-4 is evaluated as 8.",
        "64: 12/0.04 is evaluated as 30.",
        "161: two computed animal categories disappear from the final total.",
        "174: 5700 seconds is correctly converted to 95 minutes, but 5700 is boxed.",
        "707: the reasoning obtains 8 percent, but 12 is boxed.",
        "1026: the reasoning obtains 43,200, but 432,000 is boxed.",
        "1118: all subcounts are present, but the goldfish terms are omitted from the final comparison.",
        "1142: the reasoning obtains 9, but 19 is boxed.",
        "",
        "Limitations",
        "-----------",
        "Manual categories label the dominant visible error and may overlap.",
        "Text comparison localizes the visible reasoning failure, not the exact internal causal intervention.",
    ]
    (OUT_DIR / "report_reasoning_error_v3.txt").write_text("\n".join(lines) + "\n")
    print(json.dumps({
        "paired": result["paired_outcomes"],
        "categories": result["regression_error_categories"],
        "group_statistics": result["group_statistics"],
        "outputs": [str(out_json), str(OUT_DIR / "report_reasoning_error_v3.txt")],
    }, indent=2))


if __name__ == "__main__":
    main()
