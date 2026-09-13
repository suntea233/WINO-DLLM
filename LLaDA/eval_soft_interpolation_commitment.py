"""Evaluate one-round interpolated V6 identity commitments."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import sys
import tempfile
import time

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

import decoding
from dataset_utils.gsm8k import (
    gsm8k_doc_to_text, gsm8k_extract_answer, gsm8k_is_correct,
)
from dataset_utils.humaneval import humaneval_doc_to_text, humaneval_extract_answer
from dataset_utils.math500 import (
    math500_doc_to_text, math500_extract_answer, math500_is_equiv)
from dataset_utils.mbpp import mbpp_doc_to_text, mbpp_extract_answer, mbpp_read_data
from dataset_utils.eval_correctness_mbpp.evaluation import (
    check_correctness as mbpp_check_correctness,
    process_humaneval_test as mbpp_process_test,
    read_dataset as mbpp_read_problems,
)
try:
    from human_eval.evaluation import evaluate_functional_correctness as evaluate_humaneval
except ModuleNotFoundError:
    sys.path.append("/root/lx/ReMix-DLLM/LLaDA/dev/lib/python3.11/site-packages")
    from human_eval.evaluation import evaluate_functional_correctness as evaluate_humaneval
from modeling_llada import LLaDAModelLM


HERE = os.path.dirname(__file__)
MODEL_PATH = "/root/lx/ReMix-DLLM/LLaDA/models/LLaDA-8B-Instruct"
RESULT_ROOT = "/root/lx/WINO-DLLM/LLaDA/results"
TASKS = {
    "gsm8k": {"threshold": 0.6, "total": 1319},
    "math500": {"threshold": 0.7, "total": 500},
    "humaneval": {"threshold": 0.7, "total": 164},
    "mbpp": {"threshold": 0.8, "total": 500},
}


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_json(path, value):
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
    os.replace(temporary, path)


def load_task(name):
    if name == "gsm8k":
        return load_dataset(
            "/root/lx/ReMix-DLLM/LLaDA/data/gsm8k", "main",
            trust_remote_code=True)["test"]
    if name == "math500":
        return load_dataset(
            "/root/lx/ReMix-DLLM/LLaDA/data/math500",
            trust_remote_code=True)["test"]
    if name == "humaneval":
        return load_dataset(
            "/root/lx/ReMix-DLLM/LLaDA/data/humaneval",
            trust_remote_code=True)["test"]
    if name == "mbpp":
        return list(mbpp_read_data(os.path.join(HERE, "data", "mbpp", "mbpp.jsonl")))
    raise ValueError(name)


def format_example(name, doc):
    if name == "gsm8k":
        context, target = gsm8k_doc_to_text(doc)
        return context, target, ""
    if name == "math500":
        context, target = math500_doc_to_text(doc)
        return context, target, ""
    if name == "humaneval":
        return humaneval_doc_to_text(doc)
    if name == "mbpp":
        return mbpp_doc_to_text(doc)
    raise ValueError(name)


def make_prompt(tokenizer, name, doc):
    context, target, trailing = format_example(name, doc)
    prompt = tokenizer.apply_chat_template(
        context, add_generation_prompt=True, tokenize=False) + trailing
    if name in {"gsm8k", "math500"}:
        prompt += "<reasoning>"
    return tokenizer(prompt, return_tensors="pt").input_ids.cuda(), target


def evaluate_humaneval_rows(rows):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as handle:
        sample_file = handle.name
        for row in rows:
            handle.write(json.dumps({
                "task_id": row["task_id"], "completion": row["completion"]}) + "\n")
    metrics = evaluate_humaneval(sample_file)
    evaluated = read_jsonl(sample_file + "_results.jsonl")
    passed = {row["task_id"]: bool(row["passed"]) for row in evaluated}
    os.unlink(sample_file)
    os.unlink(sample_file + "_results.jsonl")
    return metrics, passed


def evaluate_mbpp_rows(rows):
    problem_file = os.path.join(HERE, "data", "mbpp", "mbpp_test.jsonl")
    problems = mbpp_read_problems(problem_file, dataset_type="humaneval")
    completion_id = Counter()
    results = defaultdict(list)
    temporary_root = tempfile.mkdtemp(prefix="revision_mbpp_eval_")
    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = []
        for row in rows:
            sample = {"task_id": row["task_id"], "completion": row["completion"]}
            task_id = sample["task_id"]
            sample["test_code"] = mbpp_process_test(
                sample, problems, False, True, "python")
            futures.append(executor.submit(
                mbpp_check_correctness, task_id, sample, "python", 10.0,
                os.path.join(temporary_root, "python", "evaluation"),
                completion_id[task_id]))
            completion_id[task_id] += 1
        for future in as_completed(futures):
            result = future.result()
            results[result["task_id"]].append(result)
    passed = {task_id: bool(values[0]["passed"])
              for task_id, values in results.items()}
    return {"pass@1": float(np.mean(list(passed.values())))}, passed


def v6_path(name):
    return os.path.join(
        RESULT_ROOT, "soft_revision_v6", f"{name}_soft_revision_v6.json")


def sample_correct(name, row):
    return bool(row["is_correct"])


def paired(name, v6, rows):
    v6_by_id = {row["index"]: row for row in v6["samples"]}
    ours_by_id = {row["index"]: row for row in rows}
    matched = sorted(set(v6_by_id) & set(ours_by_id))
    recovered = [index for index in matched
                 if not sample_correct(name, v6_by_id[index])
                 and sample_correct(name, ours_by_id[index])]
    regressed = [index for index in matched
                 if sample_correct(name, v6_by_id[index])
                 and not sample_correct(name, ours_by_id[index])]
    return {
        "matched": len(matched), "recovered": len(recovered),
        "regressed": len(regressed),
        "net_recovery": len(recovered) - len(regressed),
        "recovered_sample_ids": recovered, "regressed_sample_ids": regressed,
    }


def compact_diagnostics(diagnostics):
    events = []
    for event in diagnostics["revision_events"]:
        events.append({
            "position": event["position"], "block": event["block"],
            "old_hard_token_a": event["original_hard_token_id"],
            "revision_candidate_b": event.get("current_top1_token_id"),
            "new_token_id": event.get("new_token_id"),
            "commit_reason": event.get("commit_reason"),
            "identity_changed": event.get("identity_changed", False),
            "alpha": event.get("commit_alpha"),
            "soft_distribution_entropy": event.get("commit_soft_entropy"),
            "final_committed_representation_type": event.get(
                "final_committed_representation_type"),
            "interpolated_commit_consumed": event.get(
                "interpolated_commit_consumed"),
            "interpolated_commit_superseded_by_revision": event.get(
                "interpolated_commit_superseded_by_revision", False),
        })
    return {
        "nfe": diagnostics["nfe"],
        "decoding_rounds": diagnostics["decoding_rounds"],
        "num_h_to_soft_revisions": diagnostics["num_h_to_soft_revisions"],
        "num_h_to_s_to_h": diagnostics["num_h_to_s_to_h"],
        "num_h_to_s_to_m": diagnostics["num_h_to_s_to_m"],
        "num_interpolated_identity_commits": diagnostics[
            "num_interpolated_identity_commits"],
        "num_interpolated_commit_forwards_consumed": diagnostics[
            "num_interpolated_commit_forwards_consumed"],
        "unresolved_soft_states": diagnostics["unresolved_soft_states"],
        "revision_events": events,
    }


def finalize(name, alpha, checkpoint, output_dir):
    rows = sorted(read_jsonl(checkpoint), key=lambda row: row["index"])
    total = TASKS[name]["total"]
    if len(rows) != total or [row["index"] for row in rows] != list(range(total)):
        raise RuntimeError(f"Incomplete {name}/alpha={alpha}: {len(rows)}/{total}")
    external = {}
    if name == "humaneval":
        external, passed = evaluate_humaneval_rows(rows)
        for row in rows:
            row["is_correct"] = passed[row["task_id"]]
    elif name == "mbpp":
        external, passed = evaluate_mbpp_rows(rows)
        for row in rows:
            row["is_correct"] = passed[row["task_id"]]
    correct = sum(row["is_correct"] for row in rows)
    accuracy = correct / total
    diagnostics = [row["diagnostics"] for row in rows]
    if any(d["nfe"] != d["decoding_rounds"] for d in diagnostics):
        raise RuntimeError("NFE does not equal normal decoding rounds")
    if any(d["unresolved_soft_states"] for d in diagnostics):
        raise RuntimeError("Unresolved Soft state at termination")
    metrics = {
        **external, "accuracy": accuracy, "correct": correct, "total": total,
        "average_nfe": sum(row["nfe"] for row in rows) / total,
        "average_latency": sum(row["latency"] for row in rows) / total,
        "h_to_soft": sum(d["num_h_to_soft_revisions"] for d in diagnostics),
        "identity_corrections": sum(
            event.get("identity_changed", False)
            for d in diagnostics for event in d["revision_events"]),
        "interpolated_identity_commits": sum(
            d["num_interpolated_identity_commits"] for d in diagnostics),
        "interpolated_commit_forwards_consumed": sum(
            d["num_interpolated_commit_forwards_consumed"] for d in diagnostics),
    }
    with open(v6_path(name), encoding="utf-8") as handle:
        v6 = json.load(handle)
    comparison = {
        "v6_accuracy": v6["metrics"]["accuracy"],
        "v6_average_nfe": v6["metrics"]["average_nfe"],
        "delta_accuracy_vs_v6": accuracy - v6["metrics"]["accuracy"],
        "delta_average_nfe_vs_v6": metrics["average_nfe"] - v6["metrics"]["average_nfe"],
        "paired_vs_v6": paired(name, v6, rows),
    }
    result = {
        "dataset": name, "method": "soft_interpolation_commitment",
        "configuration": {
            "model": MODEL_PATH, "gen_length": 256, "block_length": 128,
            "temperature": 0.0, "threshold": TASKS[name]["threshold"],
            "threshold_back": 0.9, "beta_mix": 0.5,
            "commit_alpha": alpha,
            "commit_soft_distribution": "current shadow posterior",
            "interpolation_lifetime_normal_forwards": 1,
        },
        "metrics": metrics, "comparison": comparison, "samples": rows,
    }
    alpha_tag = str(alpha).replace(".", "p")
    final_path = os.path.join(output_dir, f"{name}_alpha_{alpha_tag}.json")
    atomic_json(final_path, result)
    report = (
        f"Soft Interpolation Commitment: {name}, alpha={alpha}\n"
        f"{'=' * 48}\n"
        f"Accuracy: {accuracy:.4%} ({correct}/{total})\n"
        f"Average NFE: {metrics['average_nfe']:.2f}\n"
        f"Average latency: {metrics['average_latency']:.3f}s\n"
        f"Delta accuracy vs V6: {comparison['delta_accuracy_vs_v6']:+.4%}\n"
        f"Delta average NFE vs V6: {comparison['delta_average_nfe_vs_v6']:+.2f}\n"
        f"V6 wrong -> ours correct: {comparison['paired_vs_v6']['recovered']}\n"
        f"V6 correct -> ours wrong: {comparison['paired_vs_v6']['regressed']}\n"
        f"Net recovery: {comparison['paired_vs_v6']['net_recovery']:+d}\n")
    with open(os.path.join(output_dir, f"report_{name}_alpha_{alpha_tag}.txt"), "w",
              encoding="utf-8") as handle:
        handle.write(report)
    print(report, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=TASKS, required=True)
    parser.add_argument("--alpha", type=float, choices=(0.25, 0.5, 0.75), required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    dataset = load_task(args.dataset)
    total = TASKS[args.dataset]["total"]
    if len(dataset) != total:
        raise RuntimeError(f"Unexpected {args.dataset} size: {len(dataset)}/{total}")
    decoder = decoding.decoding_wino_soft_interpolation_commitment
    alpha_tag = str(args.alpha).replace(".", "p")
    checkpoint = os.path.join(
        args.output_dir, f".{args.dataset}_alpha_{alpha_tag}_samples.jsonl")
    completed = {row["index"] for row in read_jsonl(checkpoint)}
    print(f"{args.dataset}: resuming {len(completed)}/{total}", flush=True)
    model = LLaDAModelLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16).cuda().eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    warm_ids, _ = make_prompt(tokenizer, args.dataset, dataset[0])
    decoder(
        model, warm_ids, gen_length=256, block_length=128, temperature=0.0,
        threshold=TASKS[args.dataset]["threshold"], threshold_back=0.9,
        beta_mix=0.5, commit_alpha=args.alpha)
    with open(checkpoint, "a", encoding="utf-8", buffering=1) as output:
        for index, doc in enumerate(dataset):
            if index in completed:
                continue
            input_ids, target = make_prompt(tokenizer, args.dataset, doc)
            torch.cuda.synchronize()
            started = time.perf_counter()
            generated, steps, diag = decoder(
                model, input_ids, gen_length=256, block_length=128,
                temperature=0.0, threshold=TASKS[args.dataset]["threshold"],
                threshold_back=0.9, beta_mix=0.5,
                return_diagnostics=True, commit_alpha=args.alpha)
            torch.cuda.synchronize()
            latency = time.perf_counter() - started
            response = tokenizer.batch_decode(
                generated[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
            row = {
                "index": index, "full_response": response,
                "nfe": int(steps), "latency": latency,
                "diagnostics": compact_diagnostics(diag),
            }
            if args.dataset == "gsm8k":
                prediction = gsm8k_extract_answer(response)
                row.update({"prediction": prediction,
                            "is_correct": bool(gsm8k_is_correct(prediction, target))})
            elif args.dataset == "math500":
                prediction = math500_extract_answer(response)
                row.update({"prediction": prediction,
                            "is_correct": bool(math500_is_equiv(prediction, target))})
            elif args.dataset == "humaneval":
                row.update({"task_id": target["task_id"],
                            "completion": humaneval_extract_answer(response, target),
                            "is_correct": None})
            elif args.dataset == "mbpp":
                row.update({"task_id": target["task_id"],
                            "completion": mbpp_extract_answer(
                                "```python\n" + response, target["entry_point"]),
                            "is_correct": None})
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"{args.dataset} {index + 1}/{total} nfe={steps}", flush=True)
    finalize(args.dataset, args.alpha, checkpoint, args.output_dir)


if __name__ == "__main__":
    main()
