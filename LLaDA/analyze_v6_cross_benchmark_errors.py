"""Offline cross-benchmark error analysis for Soft Revision V6."""

import ast
import json
import math
import os
import re
from collections import Counter, defaultdict
from fractions import Fraction

import pyarrow.parquet as pq


ROOT = "/root/lx/WINO-DLLM/LLaDA/results"
OUT = os.path.join(ROOT, "v6_cross_benchmark_error_analysis")

V6 = {
    "gsm8k": f"{ROOT}/soft_revision_v6/gsm8k_soft_revision_v6.json",
    "math500": f"{ROOT}/soft_revision_v6/math500_soft_revision_v6.json",
    "humaneval": f"{ROOT}/soft_revision_v6/humaneval_soft_revision_v6.json",
    "mbpp": f"{ROOT}/soft_revision_v6/mbpp_soft_revision_v6.json",
    "countdown": f"{ROOT}/soft_revision_v6/countdown_soft_revision_v6.json",
    "sudoku": f"{ROOT}/soft_revision_v6_reasoning/sudoku_soft_revision_v6.json",
    "arc_easy": f"{ROOT}/soft_revision_v6_reasoning/arc_easy_soft_revision_v6.json",
    "arc_challenge": f"{ROOT}/soft_revision_v6_reasoning/arc_challenge_soft_revision_v6.json",
}
CONTROL = {
    "gsm8k": f"{ROOT}/soft_revision_v6_control_v1/gsm8k_soft_revision.json",
    "math500": f"{ROOT}/soft_revision/math500_soft_revision.json",
    "humaneval": f"{ROOT}/soft_revision/humaneval_soft_revision.json",
    "mbpp": f"{ROOT}/soft_revision/mbpp_soft_revision.json",
    "countdown": f"{ROOT}/soft_revision/countdown_soft_revision.json",
    "sudoku": f"{ROOT}/soft_revision/sudoku_soft_revision.json",
    "arc_easy": f"{ROOT}/soft_revision/arc_easy_soft_revision.json",
    "arc_challenge": f"{ROOT}/soft_revision/arc_challenge_soft_revision.json",
}


def load(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def correct(row, dataset):
    return bool(row.get("is_exact") if dataset == "sudoku" else row.get("is_correct"))


def diag(row, name):
    d = row.get("diagnostics", {})
    if name == "h_to_s":
        return d.get("num_h_to_soft_revisions", d.get("num_h_to_s", 0))
    if name == "identity":
        if "identity_corrections" in d:
            return d["identity_corrections"]
        return sum(e.get("commit_reason") == "different_identity_correction"
                   for e in d.get("revision_events", []))
    if name == "s_to_m":
        return d.get("num_h_to_s_to_m", d.get("num_s_to_m", 0))
    return row.get("nfe", row.get("steps", 0))


def mean(xs):
    return sum(xs) / len(xs) if xs else None


def group_stats(rows):
    identities = [diag(r, "identity") for r in rows]
    return {
        "samples": len(rows),
        "mean_nfe": mean([diag(r, "nfe") for r in rows]),
        "mean_h_to_s": mean([diag(r, "h_to_s") for r in rows]),
        "mean_identity_corrections": mean(identities),
        "samples_with_identity_correction": sum(x > 0 for x in identities),
        "total_identity_corrections": sum(identities),
        "mean_s_to_mask": mean([diag(r, "s_to_m") for r in rows]),
    }


def paired(dataset, ours, control):
    cb = {int(r["index"]): r for r in control}
    groups = defaultdict(list)
    for row in ours:
        c0, c1 = correct(cb[int(row["index"])], dataset), correct(row, dataset)
        group = ("recovered" if not c0 and c1 else "regressed" if c0 and not c1
                 else "both_correct" if c1 else "both_wrong")
        groups[group].append(row)
    return {g: group_stats(groups[g]) for g in
            ("both_correct", "recovered", "regressed", "both_wrong")}, groups


def gsm_taxonomy(rows, targets):
    counts = Counter()
    for row in rows:
        if row["is_correct"]:
            continue
        pred = row.get("prediction")
        response = row.get("full_response", "")
        if pred is None:
            counts["no_extractable_answer"] += 1
            continue
        target = targets[row["index"]]
        if not math.isfinite(float(pred)):
            counts["non_finite_answer"] += 1
        elif float(pred) == -target:
            counts["sign_error"] += 1
        elif abs(float(pred) - target) == 1:
            counts["off_by_one"] += 1
        elif target and abs(float(pred) - target) / abs(target) <= .1:
            counts["near_numeric_error_le_10pct"] += 1
        elif target and (abs(float(pred)) >= 10 * abs(target)
                         or abs(float(pred)) <= .1 * abs(target)):
            counts["order_of_magnitude_error"] += 1
        else:
            counts["other_wrong_numeric_answer"] += 1
        if "</answer>" not in response:
            counts["missing_answer_close_tag_overlap"] += 1
    return dict(counts)


def code_taxonomy(dataset, rows, docs):
    counts = Counter()
    for row in rows:
        if row["is_correct"]:
            continue
        completion = row.get("completion", "")
        if not completion.strip():
            counts["empty_completion"] += 1
            continue
        source = (docs[row["index"]]["prompt"] + completion
                  if dataset == "humaneval" else completion)
        try:
            tree = ast.parse(source)
        except (SyntaxError, ValueError, TypeError):
            counts["syntax_or_truncation_error"] += 1
            continue
        expected = (docs[row["index"]]["entry_point"] if dataset == "humaneval"
                    else docs[row["task_id"]]["entry_point"])
        names = {node.name for node in ast.walk(tree)
                 if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
        if expected not in names:
            counts["missing_required_function"] += 1
        else:
            counts["executable_but_semantically_wrong"] += 1
    return dict(counts)


ALLOWED_OPS = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b,
               ast.Mult: lambda a, b: a * b, ast.Div: lambda a, b: a / b}


def eval_expr(node, numbers):
    if isinstance(node, ast.Expression):
        return eval_expr(node.body, numbers)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        numbers.append(Fraction(str(node.value)))
        return Fraction(str(node.value))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -eval_expr(node.operand, numbers)
    if isinstance(node, ast.BinOp) and type(node.op) in ALLOWED_OPS:
        return ALLOWED_OPS[type(node.op)](eval_expr(node.left, numbers),
                                          eval_expr(node.right, numbers))
    raise ValueError("unsupported")


def countdown_taxonomy(rows, docs):
    counts = Counter()
    for row in rows:
        if row["is_correct"]:
            continue
        prediction = row.get("prediction")
        if not prediction:
            counts["no_extractable_expression"] += 1
            continue
        try:
            used = []
            value = eval_expr(ast.parse(prediction, mode="eval"), used)
        except Exception:
            counts["invalid_or_unsupported_expression"] += 1
            continue
        available = sorted(Fraction(x) for x in docs[row["index"]]["input"].split(","))
        if sorted(used) != available:
            counts["does_not_use_given_numbers_exactly_once"] += 1
        elif value != Fraction(docs[row["index"]]["output"]):
            counts["valid_numbers_but_wrong_value"] += 1
        else:
            counts["evaluator_or_extraction_mismatch"] += 1
    return dict(counts)


def arc_taxonomy(rows):
    wrong = [r for r in rows if not r["is_correct"]]
    nonstandard = [r for r in wrong if r["target"] not in "ABCD"]
    return {
        "wrong_total": len(wrong),
        "no_extracted_choice": sum(r.get("prediction") is None for r in wrong),
        "explicit_wrong_choice": sum(r.get("prediction") is not None for r in wrong),
        "nonstandard_target_label": len(nonstandard),
        "missing_answer_tag": sum("<answer>" not in r.get("full_response", "")
                                  for r in wrong),
    }


def main():
    os.makedirs(OUT, exist_ok=True)
    data = {name: load(path)["samples"] for name, path in V6.items()}
    controls = {name: load(path)["samples"] for name, path in CONTROL.items()}
    paired_stats, paired_groups = {}, {}
    for name in data:
        paired_stats[name], paired_groups[name] = paired(name, data[name], controls[name])

    gsm_docs = pq.read_table(
        "/root/lx/ReMix-DLLM/LLaDA/data/gsm8k/main/test-00000-of-00001.parquet"
    ).to_pylist()
    gsm_targets = [float(re.search(r"####\s*([-0-9.,]+)", d["answer"])
                         .group(1).replace(",", "")) for d in gsm_docs]
    math_docs = [json.loads(x) for x in open(
        "/root/lx/ReMix-DLLM/LLaDA/data/math500/test.jsonl", encoding="utf-8")]
    he_docs = pq.read_table(
        "/root/lx/ReMix-DLLM/LLaDA/data/humaneval/openai_humaneval/test-00000-of-00001.parquet"
    ).to_pylist()
    mbpp_all = [json.loads(x) for x in open(
        "/root/lx/WINO-DLLM/LLaDA/data/mbpp/mbpp.jsonl", encoding="utf-8")]
    mbpp_by_id = {int(x["task_id"]): {**x, "entry_point": re.search(
        r"(?:def\s+|assert\s+)([A-Za-z_]\w*)", x["code"] + "\n" + "\n".join(x["test_list"])).group(1)}
                  for x in mbpp_all}
    countdown_docs = [json.loads(x) for x in open(
        "/root/lx/WINO-DLLM/LLaDA/data/countdown/countdown_cd3_test.jsonl", encoding="utf-8")]

    math_by_subject = {}
    for subject in sorted(set(x["subject"] for x in math_docs)):
        ids = [i for i, x in enumerate(math_docs) if x["subject"] == subject]
        math_by_subject[subject] = {"total": len(ids), "correct": sum(
            data["math500"][i]["is_correct"] for i in ids)}
        math_by_subject[subject]["accuracy"] = (
            math_by_subject[subject]["correct"] / len(ids))
    math_by_level = {}
    for level in sorted(set(x["level"] for x in math_docs)):
        ids = [i for i, x in enumerate(math_docs) if x["level"] == level]
        math_by_level[str(level)] = {"total": len(ids), "correct": sum(
            data["math500"][i]["is_correct"] for i in ids)}
        math_by_level[str(level)]["accuracy"] = (
            math_by_level[str(level)]["correct"] / len(ids))

    taxonomy = {
        "gsm8k": gsm_taxonomy(data["gsm8k"], gsm_targets),
        "math500": {
            "wrong_total": sum(not r["is_correct"] for r in data["math500"]),
            "no_extractable_answer": sum(not r["is_correct"] and not r.get("prediction")
                                         for r in data["math500"]),
            "wrong_extracted_answer": sum(not r["is_correct"] and bool(r.get("prediction"))
                                          for r in data["math500"]),
            "by_subject": math_by_subject, "by_level": math_by_level,
        },
        "humaneval": code_taxonomy("humaneval", data["humaneval"], he_docs),
        "mbpp": code_taxonomy("mbpp", data["mbpp"], mbpp_by_id),
        "countdown": countdown_taxonomy(data["countdown"], countdown_docs),
        "arc_easy": arc_taxonomy(data["arc_easy"]),
        "arc_challenge": arc_taxonomy(data["arc_challenge"]),
        "sudoku": load(f"{ROOT}/soft_revision_v6_reasoning/reasoning_benchmark_analysis.json")["sudoku"],
    }

    sample_groups = {}
    for name, rows in data.items():
        sample_groups[name] = {
            "correct": group_stats([r for r in rows if correct(r, name)]),
            "wrong": group_stats([r for r in rows if not correct(r, name)]),
        }
    summary = {"method": "Soft Revision V6", "paired_control": {
        "gsm8k": "same-run V1 control", "others": "matched V1 outputs"},
        "paired": paired_stats, "correct_wrong_diagnostics": sample_groups,
        "error_taxonomy": taxonomy,
        "interpretation_guard": "Descriptive associations do not establish causal effects of revision actions."}
    with open(os.path.join(OUT, "v6_cross_benchmark_error_summary.json"), "w",
              encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    case_context = {
        "gsm8k": {i: {"problem": d["question"], "target": gsm_targets[i]}
                   for i, d in enumerate(gsm_docs)},
        "math500": {i: {"problem": d["problem"], "target": d["answer"],
                         "subject": d["subject"], "level": d["level"]}
                    for i, d in enumerate(math_docs)},
        "countdown": {i: {"problem": d["input"], "target": d["output"]}
                      for i, d in enumerate(countdown_docs)},
    }
    cases = []
    for name, groups in paired_groups.items():
        control_by_id = {int(r["index"]): r for r in controls[name]}
        for group in ("recovered", "regressed"):
            for row in groups[group][:5]:
                cases.append({"dataset": name, "group": group,
                              "index": row["index"],
                              **case_context.get(name, {}).get(int(row["index"]), {}),
                              "control_prediction": control_by_id[int(row["index"])].get("prediction"),
                              "v6_prediction": row.get("prediction"),
                              "target": case_context.get(name, {}).get(
                                  int(row["index"]), {}).get("target", row.get("target")),
                              "v6_identity_corrections": diag(row, "identity"),
                              "v6_response_excerpt": row.get("full_response", "")[:800]})
    with open(os.path.join(OUT, "paired_casebook.jsonl"), "w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case, ensure_ascii=False) + "\n")
    print(json.dumps({"paired": paired_stats, "taxonomy": taxonomy},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
