"""Offline diagnostics for completed V6 Sudoku and ARC runs."""

import collections
import json
import os
import re
import statistics

import pandas as pd


HERE = os.path.dirname(__file__)
ROOT = os.path.join(HERE, "results", "soft_revision_v6_reasoning")


def load(name):
    with open(os.path.join(ROOT, f"{name}_soft_revision_v6.json"),
              encoding="utf-8") as handle:
        return json.load(handle)


def mean(rows, key):
    return statistics.mean(row[key] for row in rows) if rows else None


def arc_analysis(name):
    data = load(name)
    rows = data["samples"]
    standard = [row for row in rows if row["target"] in "ABCD"]
    nonstandard = [row for row in rows if row["target"] not in "ABCD"]
    none_rows = [row for row in rows if row["prediction"] is None]
    recovered_boxed = []
    for row in none_rows:
        matches = re.findall(
            r"\\boxed\{\s*([A-E1-4])\s*\}", row["full_response"], re.I)
        if matches:
            recovered_boxed.append((row, matches[-1].upper()))
    numeric_to_letter = {"1": "A", "2": "B", "3": "C", "4": "D"}
    mapped_numeric_correct = sum(
        row["target"] in numeric_to_letter
        and numeric_to_letter[row["target"]] == row["prediction"]
        for row in rows)
    correct_rows = [row for row in rows if row["is_correct"]]
    wrong_rows = [row for row in rows if not row["is_correct"]]
    return {
        "official": data["metrics"],
        "standard_abcd": {
            "samples": len(standard),
            "correct": sum(row["is_correct"] for row in standard),
            "accuracy": sum(row["is_correct"] for row in standard) / len(standard),
        },
        "nonstandard_labels": {
            "samples": len(nonstandard),
            "target_counts": dict(collections.Counter(
                row["target"] for row in nonstandard)),
            "official_correct": sum(row["is_correct"] for row in nonstandard),
            "numeric_ordinal_to_letter_matches": mapped_numeric_correct,
            "diagnostic_accuracy_if_numeric_labels_are_mapped_by_ordinal": (
                (data["metrics"]["correct"] + mapped_numeric_correct) / len(rows)),
        },
        "format": {
            "prediction_none": len(none_rows),
            "prediction_none_rate": len(none_rows) / len(rows),
            "none_with_boxed_choice_elsewhere": len(recovered_boxed),
            "none_with_correct_boxed_choice_elsewhere": sum(
                choice == row["target"] for row, choice in recovered_boxed),
            "none_without_answer_tag": sum(
                "<answer>" not in row["full_response"] for row in none_rows),
        },
        "correct_wrong_diagnostics": {
            "correct": {
                "samples": len(correct_rows),
                "average_nfe": mean(correct_rows, "nfe"),
                "average_h_to_s": statistics.mean(
                    row["diagnostics"]["num_h_to_soft_revisions"]
                    for row in correct_rows),
                "average_identity_corrections": statistics.mean(
                    row["diagnostics"]["identity_corrections"]
                    for row in correct_rows),
            },
            "wrong": {
                "samples": len(wrong_rows),
                "average_nfe": mean(wrong_rows, "nfe"),
                "average_h_to_s": statistics.mean(
                    row["diagnostics"]["num_h_to_soft_revisions"]
                    for row in wrong_rows),
                "average_identity_corrections": statistics.mean(
                    row["diagnostics"]["identity_corrections"]
                    for row in wrong_rows),
            },
        },
    }


def legal_sudoku(grid):
    if len(grid) != 16 or not set(grid) <= set("1234"):
        return False
    expected = set("1234")
    units = [grid[index * 4:(index + 1) * 4] for index in range(4)]
    units += ["".join(grid[index + 4 * row] for row in range(4))
              for index in range(4)]
    units += ["".join(grid[(block_row + row) * 4 + block_column + column]
                        for row in range(2) for column in range(2))
              for block_row in (0, 2) for block_column in (0, 2)]
    return all(set(unit) == expected for unit in units)


def sudoku_analysis():
    data = load("sudoku")
    rows = data["samples"]
    frame = pd.read_csv(
        os.path.join(HERE, "data", "sudoku", "4x4_test_sudoku.csv"),
        dtype=str)
    format_valid = [row for row in rows
                    if len(row["prediction"]) == 16
                    and set(row["prediction"]) <= set("1234")]
    clue_consistent = []
    legal = []
    for row in rows:
        puzzle = frame.iloc[row["index"]]["Puzzle"]
        prediction = row["prediction"]
        if (len(prediction) == 16 and set(prediction) <= set("1234")
                and all(given == "0" or given == predicted
                        for given, predicted in zip(puzzle, prediction))):
            clue_consistent.append(row)
        if legal_sudoku(prediction):
            legal.append(row)
    example_solutions = list(frame.iloc[:4]["Solution"])
    example_copies = [row for row in rows if row["prediction"] in example_solutions]
    heldout = rows[4:]
    valid_heldout = [row for row in format_valid if row["index"] >= 4]
    exact = [row for row in rows if row["is_exact"]]
    invalid = [row for row in rows if row not in format_valid]
    return {
        "official": data["metrics"],
        "format": {
            "valid_16_digits_1_to_4": len(format_valid),
            "valid_format_rate": len(format_valid) / len(rows),
            "empty_extractions": sum(not row["prediction"] for row in rows),
            "legal_sudoku_grids": len(legal),
            "legal_grid_rate": len(legal) / len(rows),
            "clue_consistent_outputs": len(clue_consistent),
            "clue_consistent_rate": len(clue_consistent) / len(rows),
        },
        "conditional": {
            "cell_accuracy_given_valid_format": (
                sum(row["correct_cells"] for row in format_valid)
                / sum(row["total_cells"] for row in format_valid)),
            "valid_format_heldout_samples": len(valid_heldout),
            "cell_accuracy_given_valid_format_excluding_four_prompt_examples": (
                sum(row["correct_cells"] for row in valid_heldout)
                / sum(row["total_cells"] for row in valid_heldout)),
        },
        "prompt_example_overlap": {
            "first_four_test_rows_are_the_four_prompt_examples": True,
            "official_exact_among_first_four": sum(
                row["is_exact"] for row in rows[:4]),
            "heldout_samples_after_excluding_first_four": len(heldout),
            "heldout_correct_cells": sum(row["correct_cells"] for row in heldout),
            "heldout_total_cells": sum(row["total_cells"] for row in heldout),
            "heldout_cell_accuracy": (
                sum(row["correct_cells"] for row in heldout)
                / sum(row["total_cells"] for row in heldout)),
            "heldout_exact_puzzles": sum(row["is_exact"] for row in heldout),
            "heldout_exact_accuracy": sum(row["is_exact"] for row in heldout) / len(heldout),
            "outputs_equal_to_one_of_four_example_solutions": len(example_copies),
        },
        "exact_indices": [row["index"] for row in exact],
        "correct_cell_histogram": dict(collections.Counter(
            row["correct_cells"] for row in rows)),
        "valid_invalid_diagnostics": {
            "valid_format": {
                "samples": len(format_valid),
                "average_nfe": mean(format_valid, "nfe"),
                "average_h_to_s": statistics.mean(
                    row["diagnostics"]["num_h_to_soft_revisions"]
                    for row in format_valid),
                "average_identity_corrections": statistics.mean(
                    row["diagnostics"]["identity_corrections"]
                    for row in format_valid),
            },
            "invalid_format": {
                "samples": len(invalid),
                "average_nfe": mean(invalid, "nfe"),
                "average_h_to_s": statistics.mean(
                    row["diagnostics"]["num_h_to_soft_revisions"]
                    for row in invalid),
                "average_identity_corrections": statistics.mean(
                    row["diagnostics"]["identity_corrections"]
                    for row in invalid),
            },
        },
    }


def main():
    analysis = {
        "arc_easy": arc_analysis("arc_easy"),
        "arc_challenge": arc_analysis("arc_challenge"),
        "sudoku": sudoku_analysis(),
        "interpretation_guard": (
            "All subgroup and format analyses are descriptive. They do not establish that "
            "revision frequency or NFE causes correctness."),
    }
    output_json = os.path.join(ROOT, "reasoning_benchmark_analysis.json")
    with open(output_json + ".tmp", "w", encoding="utf-8") as handle:
        json.dump(analysis, handle, ensure_ascii=False, indent=2)
    os.replace(output_json + ".tmp", output_json)

    easy = analysis["arc_easy"]
    challenge = analysis["arc_challenge"]
    sudoku = analysis["sudoku"]
    lines = [
        "Soft Revision V6: Sudoku and ARC",
        "================================",
        "",
        "Official results",
        "----------------",
        f"ARC-Easy test: {easy['official']['correct']}/{easy['official']['total']} = "
        f"{easy['official']['accuracy']:.2%}; avg NFE={easy['official']['average_nfe']:.2f}; "
        f"latency={easy['official']['average_latency']:.3f}s",
        f"ARC-Challenge validation: {challenge['official']['correct']}/"
        f"{challenge['official']['total']} = {challenge['official']['accuracy']:.2%}; "
        f"avg NFE={challenge['official']['average_nfe']:.2f}; "
        f"latency={challenge['official']['average_latency']:.3f}s",
        f"Sudoku partial-credit: {sudoku['official']['correct_cells']}/"
        f"{sudoku['official']['total_empty_cells']} = {sudoku['official']['accuracy']:.2%}; "
        f"exact={sudoku['official']['exact_puzzles']}/{sudoku['official']['total']} = "
        f"{sudoku['official']['exact_puzzle_accuracy']:.2%}; "
        f"avg NFE={sudoku['official']['average_nfe']:.2f}; "
        f"latency={sudoku['official']['average_latency']:.3f}s",
        "",
        "ARC evaluation diagnostics",
        "--------------------------",
        f"ARC-E standard A-D subset: {easy['standard_abcd']['correct']}/"
        f"{easy['standard_abcd']['samples']} = {easy['standard_abcd']['accuracy']:.2%}.",
        f"ARC-E nonstandard labels: {easy['nonstandard_labels']['samples']}; official correct="
        f"{easy['nonstandard_labels']['official_correct']}. Mapping numeric labels by ordinal "
        f"would give {easy['nonstandard_labels']['numeric_ordinal_to_letter_matches']} extra matches "
        "(diagnostic only).",
        f"ARC-E parser returned None for {easy['format']['prediction_none']} outputs; "
        f"{easy['format']['none_with_correct_boxed_choice_elsewhere']} contain the correct boxed "
        "choice elsewhere but violate the official extraction structure.",
        f"ARC-C standard A-D subset: {challenge['standard_abcd']['correct']}/"
        f"{challenge['standard_abcd']['samples']} = {challenge['standard_abcd']['accuracy']:.2%}.",
        f"ARC-C nonstandard labels: {challenge['nonstandard_labels']['samples']}; official correct="
        f"{challenge['nonstandard_labels']['official_correct']}.",
        f"ARC-C parser returned None for {challenge['format']['prediction_none']} outputs; "
        f"{challenge['format']['none_with_correct_boxed_choice_elsewhere']} contain the correct "
        "boxed choice elsewhere.",
        "Official accuracies above are unchanged; mapped/lenient figures are diagnostics only.",
        "",
        "Sudoku diagnosis",
        "----------------",
        f"Valid 16-digit [1-4] outputs: {sudoku['format']['valid_16_digits_1_to_4']}/500 "
        f"({sudoku['format']['valid_format_rate']:.2%}).",
        f"Legal Sudoku grids: {sudoku['format']['legal_sudoku_grids']}/500 "
        f"({sudoku['format']['legal_grid_rate']:.2%}).",
        f"Outputs preserving all puzzle clues: {sudoku['format']['clue_consistent_outputs']}/500 "
        f"({sudoku['format']['clue_consistent_rate']:.2%}).",
        f"Cell accuracy conditional on valid digit format: "
        f"{sudoku['conditional']['cell_accuracy_given_valid_format']:.2%}.",
        "The first four test rows exactly repeat the four solved examples embedded in the prompt; "
        "all four are solved exactly.",
        f"Excluding those rows: cell accuracy={sudoku['prompt_example_overlap']['heldout_cell_accuracy']:.2%}, "
        f"exact={sudoku['prompt_example_overlap']['heldout_exact_puzzles']}/496 = "
        f"{sudoku['prompt_example_overlap']['heldout_exact_accuracy']:.2%}.",
        f"Outputs copying one of the four demonstration solutions: "
        f"{sudoku['prompt_example_overlap']['outputs_equal_to_one_of_four_example_solutions']}/500.",
        "",
        "Revision/NFE association",
        "------------------------",
        f"ARC-E correct/wrong avg NFE: "
        f"{easy['correct_wrong_diagnostics']['correct']['average_nfe']:.2f}/"
        f"{easy['correct_wrong_diagnostics']['wrong']['average_nfe']:.2f}; identity corrections: "
        f"{easy['correct_wrong_diagnostics']['correct']['average_identity_corrections']:.2f}/"
        f"{easy['correct_wrong_diagnostics']['wrong']['average_identity_corrections']:.2f}.",
        f"ARC-C correct/wrong avg NFE: "
        f"{challenge['correct_wrong_diagnostics']['correct']['average_nfe']:.2f}/"
        f"{challenge['correct_wrong_diagnostics']['wrong']['average_nfe']:.2f}; identity corrections: "
        f"{challenge['correct_wrong_diagnostics']['correct']['average_identity_corrections']:.2f}/"
        f"{challenge['correct_wrong_diagnostics']['wrong']['average_identity_corrections']:.2f}.",
        f"Sudoku valid/invalid-format avg H->S: "
        f"{sudoku['valid_invalid_diagnostics']['valid_format']['average_h_to_s']:.2f}/"
        f"{sudoku['valid_invalid_diagnostics']['invalid_format']['average_h_to_s']:.2f}; "
        f"identity corrections: "
        f"{sudoku['valid_invalid_diagnostics']['valid_format']['average_identity_corrections']:.2f}/"
        f"{sudoku['valid_invalid_diagnostics']['invalid_format']['average_identity_corrections']:.2f}.",
        "These are correlations and do not establish causal effects.",
    ]
    output_report = os.path.join(ROOT, "report_reasoning_benchmarks.txt")
    with open(output_report, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print(json.dumps({"json": output_json, "report": output_report}, indent=2))


if __name__ == "__main__":
    main()
