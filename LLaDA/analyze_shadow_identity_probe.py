"""Offline analysis of current-position versus WINO-shadow posteriors for V6."""

import argparse
import json
import os
import re
import statistics


ROOT = os.path.join(os.path.dirname(__file__), "results")
DEFAULT_PROBE = os.path.join(
    ROOT, "soft_revision_v6", "shadow_probe_rerun", "gsm8k_soft_revision_v6.json")
DEFAULT_ORIGINAL = os.path.join(
    ROOT, "soft_revision_v6", "gsm8k_soft_revision_v6.json")
DEFAULT_CONTROL = os.path.join(
    ROOT, "soft_revision_v6_control_v1", "gsm8k_soft_revision.json")
DEFAULT_HISTORICAL = os.path.join(
    ROOT, "soft_revision", "gsm8k_soft_revision.json")
DEFAULT_OUTPUT = os.path.join(ROOT, "soft_revision_v6")


def load(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def atomic_json(path, value):
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
    os.replace(temporary, path)


def numeric_event(event):
    old = event.get("original_hard_token_string", "")
    new = event.get("new_hard_token_string", "")
    return bool(re.search(r"[0-9]", old + new))


def summarize(events):
    if not events:
        return {
            "count": 0,
            "agreement_count": 0,
            "agreement_rate": None,
            "mean_p_shadow_b_soft": None,
            "median_p_shadow_b_soft": None,
            "fraction_p_shadow_b_soft_ge_0_6": None,
            "fraction_p_shadow_b_soft_ge_0_9": None,
        }
    values = [event["p_shadow_b_soft"] for event in events]
    agreements = [event["soft_shadow_top1_agreement"] for event in events]
    return {
        "count": len(events),
        "agreement_count": sum(agreements),
        "agreement_rate": sum(agreements) / len(events),
        "mean_p_shadow_b_soft": statistics.mean(values),
        "median_p_shadow_b_soft": statistics.median(values),
        "fraction_p_shadow_b_soft_ge_0_6": (
            sum(value >= 0.6 for value in values) / len(values)),
        "fraction_p_shadow_b_soft_ge_0_9": (
            sum(value >= 0.9 for value in values) / len(values)),
    }


def group(reference, probe):
    before = bool(reference["is_correct"])
    after = bool(probe["is_correct"])
    if before and after:
        return "both_correct"
    if not before and after:
        return "recovered"
    if before and not after:
        return "regressed"
    return "both_wrong"


def gate_result(events, event_groups, predicate):
    accepted = [event for event in events if predicate(event)]
    accepted_ids = {id(event) for event in accepted}
    result = {
        "accepted": len(accepted),
        "rejected": len(events) - len(accepted),
    }
    for name in ("recovered", "regressed"):
        selected = [
            event for event in events
            if event_groups[id(event)] == name]
        accepted_selected = [
            event for event in selected if id(event) in accepted_ids]
        result[f"{name}_events"] = len(selected)
        result[f"accepted_{name}_events"] = len(accepted_selected)
        result[f"rejected_{name}_events"] = (
            len(selected) - len(accepted_selected))
    return result


def analyze_reference(name, reference_rows, probe_rows):
    event_groups = {}
    grouped = {key: [] for key in (
        "both_correct", "recovered", "regressed", "both_wrong")}
    for index, row in probe_rows.items():
        label = group(reference_rows[index], row)
        target_events = [
            event for event in row["diagnostics"]["revision_events"]
            if event.get("identity_changed")]
        grouped[label].extend(target_events)
        for event in target_events:
            event_groups[id(event)] = label

    target_events = [event for values in grouped.values() for event in values]
    result = {
        "reference": name,
        "sample_outcomes": {
            label: sum(
                group(reference_rows[index], row) == label
                for index, row in probe_rows.items())
            for label in grouped
        },
        "recovered": {
            "all": summarize(grouped["recovered"]),
            "numeric": summarize([
                event for event in grouped["recovered"] if numeric_event(event)]),
            "non_numeric": summarize([
                event for event in grouped["recovered"] if not numeric_event(event)]),
        },
        "regressed": {
            "all": summarize(grouped["regressed"]),
            "numeric": summarize([
                event for event in grouped["regressed"] if numeric_event(event)]),
            "non_numeric": summarize([
                event for event in grouped["regressed"] if not numeric_event(event)]),
        },
    }
    result["gates"] = {
        "A_agreement": gate_result(
            target_events, event_groups,
            lambda event: event["soft_shadow_top1_agreement"]),
        "B_agreement_and_shadow_ge_0_6": gate_result(
            target_events, event_groups,
            lambda event: (
                event["soft_shadow_top1_agreement"]
                and event["p_shadow_b_soft"] >= 0.6)),
        "C_agreement_and_shadow_ge_0_9": gate_result(
            target_events, event_groups,
            lambda event: (
                event["soft_shadow_top1_agreement"]
                and event["p_shadow_b_soft"] >= 0.9)),
    }
    return result


def percent(value):
    return "n/a" if value is None else f"{100 * value:.2f}%"


def probability(value):
    return "n/a" if value is None else f"{value:.4f}"


def render_group(label, result):
    lines = [label]
    for subset in ("all", "numeric", "non_numeric"):
        item = result[subset]
        lines.append(
            f"  {subset}: n={item['count']}, agreement={percent(item['agreement_rate'])}, "
            f"p_shadow(b_soft) mean/median={probability(item['mean_p_shadow_b_soft'])}/"
            f"{probability(item['median_p_shadow_b_soft'])}, >=0.6="
            f"{percent(item['fraction_p_shadow_b_soft_ge_0_6'])}, >=0.9="
            f"{percent(item['fraction_p_shadow_b_soft_ge_0_9'])}")
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", default=DEFAULT_PROBE)
    parser.add_argument("--original", default=DEFAULT_ORIGINAL)
    parser.add_argument("--control", default=DEFAULT_CONTROL)
    parser.add_argument("--historical-v1", default=DEFAULT_HISTORICAL)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    probe = load(args.probe)
    original = load(args.original)
    control = load(args.control)
    historical = load(args.historical_v1)
    probe_rows = {row["index"]: row for row in probe["samples"]}
    original_rows = {row["index"]: row for row in original["samples"]}
    control_rows = {row["index"]: row for row in control["samples"]}
    historical_rows = {row["index"]: row for row in historical["samples"]}

    if set(probe_rows) != set(original_rows):
        raise RuntimeError("Probe and original V6 sample IDs differ")
    output_matches = sum(
        probe_rows[index]["full_response"] == original_rows[index]["full_response"]
        for index in probe_rows)
    nfe_matches = sum(
        probe_rows[index]["nfe"] == original_rows[index]["nfe"]
        for index in probe_rows)
    correctness_matches = sum(
        probe_rows[index]["is_correct"] == original_rows[index]["is_correct"]
        for index in probe_rows)
    total = len(probe_rows)
    verification = {
        "samples": total,
        "exact_output_matches": output_matches,
        "nfe_matches": nfe_matches,
        "correctness_matches": correctness_matches,
        "probe_accuracy": probe["metrics"]["accuracy"],
        "original_accuracy": original["metrics"]["accuracy"],
        "probe_average_nfe": probe["metrics"]["average_nfe"],
        "original_average_nfe": original["metrics"]["average_nfe"],
        "no_extra_forward": probe["metrics"]["average_nfe"] == original["metrics"]["average_nfe"],
    }
    if output_matches != total or nfe_matches != total or correctness_matches != total:
        raise RuntimeError(
            "Logging rerun does not exactly reproduce original V6; refusing paired probe")

    all_events = [
        event for row in probe_rows.values()
        for event in row["diagnostics"]["revision_events"]
        if event.get("identity_changed")]
    required = {
        "soft_top1_token_id", "shadow_top1_token_id", "p_soft_b_soft",
        "p_shadow_b_soft", "p_shadow_b_shadow",
        "soft_shadow_top1_agreement",
    }
    if any(not required.issubset(event) for event in all_events):
        raise RuntimeError("A target event is missing shadow-probe fields")

    analysis = {
        "definition": {
            "b_soft": "argmax posterior at the original position consuming the SOFT embedding",
            "b_shadow": "argmax posterior at WINO's appended shadow-verifier position",
            "actual_v6_commit_source": "b_shadow",
            "numeric_event": "the original or newly committed token string contains an ASCII digit",
            "event_outcome_caveat": (
                "Recovered/regressed labels are assigned at sample level; individual events "
                "are not proven causal."),
        },
        "verification": verification,
        "all_identity_corrections": {
            "all": summarize(all_events),
            "numeric": summarize([event for event in all_events if numeric_event(event)]),
            "non_numeric": summarize([event for event in all_events if not numeric_event(event)]),
        },
        "primary_contemporaneous_v1": analyze_reference(
            "contemporaneous_v1_control", control_rows, probe_rows),
        "historical_v1_for_reference": analyze_reference(
            "historical_v1", historical_rows, probe_rows),
    }

    primary = analysis["primary_contemporaneous_v1"]
    recovered_rate = primary["recovered"]["all"]["agreement_rate"]
    regressed_rate = primary["regressed"]["all"]["agreement_rate"]
    separation = recovered_rate - regressed_rate
    analysis["conclusion"] = {
        "agreement_rate_difference_recovered_minus_regressed": separation,
        "recommend_v7": False,
        "reason": (
            "A gate is recommended only after inspecting whether the observed separation "
            "is clear and robust; this probe does not simulate downstream trajectories."),
    }

    os.makedirs(args.output_dir, exist_ok=True)
    output_json = os.path.join(args.output_dir, "shadow_identity_probe.json")
    atomic_json(output_json, analysis)

    lines = [
        "Shadow verifier probe for V6 identity corrections",
        "=================================================",
        "",
        "Definitions",
        "-----------",
        "b_soft is the top-1 posterior at the original position that consumes the SOFT embedding.",
        "b_shadow is the top-1 posterior at WINO's appended shadow-verifier position.",
        "Code inspection shows that current V6 already commits b_shadow; the probe does not change this rule.",
        "",
        "No-change verification",
        "----------------------",
        f"Exact outputs: {output_matches}/{total}",
        f"Identical NFE: {nfe_matches}/{total}",
        f"Identical correctness: {correctness_matches}/{total}",
        f"Accuracy old/probe: {verification['original_accuracy']:.6%} / {verification['probe_accuracy']:.6%}",
        f"Average NFE old/probe: {verification['original_average_nfe']:.6f} / {verification['probe_average_nfe']:.6f}",
        "",
        "Primary paired groups: contemporaneous V1 control",
        "-------------------------------------------------",
        f"Sample outcomes: {primary['sample_outcomes']}",
    ]
    lines.extend(render_group("Recovered events", primary["recovered"]))
    lines.extend(render_group("Regressed events", primary["regressed"]))
    lines.extend(["", "Offline gates", "-------------"])
    for gate, values in primary["gates"].items():
        lines.append(f"{gate}: {values}")
    lines.extend([
        "",
        "Interpretation guard",
        "--------------------",
        "Gate A and Gate B may be identical because an agreeing b_soft is also V6's shadow top-1,",
        "and every target identity correction already requires shadow top-1 probability >= 0.6.",
        "Rejected events are not replayed; these counts are not estimated accuracy.",
        "Recovered/regressed is a sample-level label and does not prove that each event helped or harmed.",
        "",
        f"Agreement separation (recovered - regressed): {percent(separation)}",
    ])
    report_path = os.path.join(args.output_dir, "report_shadow_identity_probe.txt")
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print(json.dumps({
        "json": output_json,
        "report": report_path,
        "verification": verification,
        "primary": primary,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
