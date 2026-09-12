"""Frozen WINO/Soft-Revision evaluation on extended benchmarks."""

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

from dataset_utils.humaneval import humaneval_doc_to_text, humaneval_extract_answer
from dataset_utils.math500 import math500_doc_to_text, math500_extract_answer, math500_is_equiv
from dataset_utils.mbpp import mbpp_doc_to_text, mbpp_extract_answer, mbpp_read_data
from dataset_utils.countdown import (
    countdown_doc_to_text, countdown_extract_answer, countdown_is_correct,
    countdown_read_data,
)
from dataset_utils.eval_correctness_mbpp.evaluation import (
    check_correctness as mbpp_check_correctness,
    process_humaneval_test as mbpp_process_test,
    read_dataset as mbpp_read_problems,
)
from decoding import (
    decoding_wino_remask, decoding_wino_soft_revision,
    decoding_wino_soft_revision_v6,
)
try:
    from human_eval.evaluation import evaluate_functional_correctness as evaluate_humaneval
except ModuleNotFoundError:
    # Keep the frozen WINO model environment, and source only the missing
    # official HumanEval evaluator from the existing ReMix environment.
    sys.path.append(
        "/root/lx/ReMix-DLLM/LLaDA/dev/lib/python3.11/site-packages"
    )
    from human_eval.evaluation import evaluate_functional_correctness as evaluate_humaneval
from modeling_llada import LLaDAModelLM


MODEL_PATH = "/root/lx/ReMix-DLLM/LLaDA/models/LLaDA-8B-Instruct"
RESULT_DIR = os.path.join(os.path.dirname(__file__), "results", "soft_revision")
DATASETS = {
    "math500": {
        "data": "/root/lx/ReMix-DLLM/LLaDA/data/math500",
        "threshold": 0.7,
        "total": 500,
    },
    "humaneval": {
        "data": "/root/lx/ReMix-DLLM/LLaDA/data/humaneval",
        "threshold": 0.7,
        "total": 164,
    },
    "mbpp": {
        "data": os.path.join(os.path.dirname(__file__), "data", "mbpp", "mbpp.jsonl"),
        "problem_file": os.path.join(os.path.dirname(__file__), "data", "mbpp", "mbpp_test.jsonl"),
        "threshold": 0.8,
        "total": 500,
    },
    "countdown": {
        "data": os.path.join(
            os.path.dirname(__file__), "data", "countdown", "countdown_cd3_test.jsonl"),
        "threshold": 0.5,
        "total": 256,
    },
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


def load_benchmark(name):
    if name == "math500":
        return load_dataset(DATASETS[name]["data"], trust_remote_code=True)["test"]
    if name == "humaneval":
        return load_dataset(DATASETS[name]["data"], trust_remote_code=True)["test"]
    if name == "mbpp":
        return list(mbpp_read_data(DATASETS[name]["data"]))
    return list(countdown_read_data(DATASETS[name]["data"]))


def format_example(name, doc):
    if name == "math500":
        context, target = math500_doc_to_text(doc)
        return context, target, ""
    if name == "humaneval":
        return humaneval_doc_to_text(doc)
    if name == "mbpp":
        return mbpp_doc_to_text(doc)
    context, _ = countdown_doc_to_text(doc)
    target = {
        "target": int(doc["output"]),
        "numbers": [int(value) for value in doc["input"].split(",")],
    }
    return context, target, ""


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
    """Use the existing MBPP evaluator's own test builder and executor."""
    problems = mbpp_read_problems(DATASETS["mbpp"]["problem_file"], dataset_type="humaneval")
    completion_id = Counter()
    futures = []
    results = defaultdict(list)
    temporary_root = tempfile.mkdtemp(prefix="wino_mbpp_eval_")
    with ThreadPoolExecutor(max_workers=32) as executor:
        for row in rows:
            sample = {"task_id": row["task_id"], "completion": row["completion"]}
            task_id = sample["task_id"]
            sample["test_code"] = mbpp_process_test(
                sample, problems, False, True, "python")
            args = (
                task_id, sample, "python", 10.0,
                os.path.join(temporary_root, "python", "evaluation"),
                completion_id[task_id],
            )
            futures.append(executor.submit(mbpp_check_correctness, *args))
            completion_id[task_id] += 1
        for future in as_completed(futures):
            result = future.result()
            results[result["task_id"]].append(result)
    passed = {task_id: bool(values[0]["passed"]) for task_id, values in results.items()}
    accuracy = float(np.mean(list(passed.values())))
    return {"pass@1": accuracy}, passed


def finalize(dataset_name, method, checkpoint_path, final_path):
    rows = sorted(read_jsonl(checkpoint_path), key=lambda row: row["index"])
    total = DATASETS[dataset_name]["total"]
    if len(rows) != total or [row["index"] for row in rows] != list(range(total)):
        raise RuntimeError(f"Incomplete {dataset_name}/{method}: {len(rows)}/{total}")

    if dataset_name == "humaneval":
        external_metrics, passed = evaluate_humaneval_rows(rows)
        for row in rows:
            row["is_correct"] = passed[row["task_id"]]
    elif dataset_name == "mbpp":
        external_metrics, passed = evaluate_mbpp_rows(rows)
        for row in rows:
            row["is_correct"] = passed[row["task_id"]]
    elif dataset_name == "math500":
        external_metrics = {"accuracy": sum(row["is_correct"] for row in rows) / total}
    else:
        external_metrics = {"accuracy": sum(row["is_correct"] for row in rows) / total}

    correct = sum(row["is_correct"] for row in rows)
    metrics = {
        **external_metrics,
        "accuracy": correct / total,
        "correct": correct,
        "total": total,
        "average_nfe": sum(row["nfe"] for row in rows) / total,
        "average_latency": sum(row["latency"] for row in rows) / total,
        "total_suspicious_hard_tokens": sum(
            row["diagnostics"]["num_suspicious_hard_tokens"] for row in rows),
        "total_h_to_mask_revisions": sum(
            row["diagnostics"]["num_h_to_mask_revisions"] for row in rows),
        "total_h_to_soft_revisions": sum(
            row["diagnostics"]["num_h_to_soft_revisions"] for row in rows),
        "total_h_to_s_to_h": sum(
            row["diagnostics"]["num_h_to_s_to_h"] for row in rows),
        "total_h_to_s_to_m": sum(
            row["diagnostics"]["num_h_to_s_to_m"] for row in rows),
    }
    soft_total = metrics["total_h_to_soft_revisions"]
    metrics["soft_recovery_rate"] = (
        metrics["total_h_to_s_to_h"] / soft_total if soft_total else 0.0)
    result = {
        "dataset": dataset_name,
        "method": method,
        "configuration": {
            "model": MODEL_PATH,
            "gen_length": 256,
            "block_length": 128,
            "temperature": 0.0,
            "threshold": DATASETS[dataset_name]["threshold"],
            "threshold_back": 0.9,
            "beta_mix": 0.5 if method.startswith("soft_revision") else None,
        },
        "metrics": metrics,
        "samples": rows,
    }
    atomic_json(final_path, result)
    if method == "soft_revision_v6":
        trace_path = os.path.join(
            os.path.dirname(final_path), f"{dataset_name}_revision_trace_v6.jsonl")
        with open(trace_path + ".tmp", "w", encoding="utf-8") as handle:
            for row in rows:
                for event in row["diagnostics"]["revision_events"]:
                    handle.write(json.dumps(
                        {"sample_index": row["index"], **event}, ensure_ascii=False) + "\n")
        os.replace(trace_path + ".tmp", trace_path)
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"Saved {final_path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument(
        "--method", choices=("wino_remask", "soft_revision", "soft_revision_v6"),
        required=True)
    parser.add_argument("--result-dir", default=RESULT_DIR)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int)
    parser.add_argument("--worker-name")
    args = parser.parse_args()

    result_dir = args.result_dir
    os.makedirs(result_dir, exist_ok=True)
    worker_suffix = f"_{args.worker_name}" if args.worker_name else ""
    checkpoint_path = os.path.join(
        result_dir, f".{args.dataset}_{args.method}{worker_suffix}_samples.jsonl")
    final_path = os.path.join(result_dir, f"{args.dataset}_{args.method}.json")
    benchmark = load_benchmark(args.dataset)
    if len(benchmark) != DATASETS[args.dataset]["total"]:
        raise RuntimeError(f"Unexpected {args.dataset} size: {len(benchmark)}")
    start_index = max(0, args.start_index)
    end_index = len(benchmark) if args.end_index is None else min(args.end_index, len(benchmark))
    expected_indices = list(range(start_index, end_index))
    completed = {row["index"] for row in read_jsonl(checkpoint_path)}
    print(
        f"{args.dataset}/{args.method}: resuming {len(completed)}/{len(expected_indices)}",
        flush=True)

    model = LLaDAModelLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16).cuda().eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    generation_functions = {
        "wino_remask": decoding_wino_remask,
        "soft_revision": decoding_wino_soft_revision,
        "soft_revision_v6": decoding_wino_soft_revision_v6,
    }
    generation_fn = generation_functions[args.method]
    threshold = DATASETS[args.dataset]["threshold"]

    context, _, trailing = format_example(args.dataset, benchmark[start_index])
    prompt = tokenizer.apply_chat_template(
        context, add_generation_prompt=True, tokenize=False) + trailing
    if args.dataset in {"math500", "countdown"}:
        prompt += "<reasoning>"
    warm_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
    generation_fn(model, warm_ids, gen_length=256, block_length=128,
                  temperature=0.0, threshold=threshold, threshold_back=0.9)

    with open(checkpoint_path, "a", encoding="utf-8", buffering=1) as checkpoint:
        for index in expected_indices:
            doc = benchmark[index]
            if index in completed:
                continue
            context, target, trailing = format_example(args.dataset, doc)
            prompt = tokenizer.apply_chat_template(
                context, add_generation_prompt=True, tokenize=False) + trailing
            if args.dataset in {"math500", "countdown"}:
                prompt += "<reasoning>"
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
            torch.cuda.synchronize()
            started = time.perf_counter()
            generated, steps, diagnostics = generation_fn(
                model, input_ids, gen_length=256, block_length=128,
                temperature=0.0, threshold=threshold, threshold_back=0.9,
                return_diagnostics=True)
            torch.cuda.synchronize()
            latency = time.perf_counter() - started
            response = tokenizer.batch_decode(
                generated[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
            diagnostics.pop("round_trace", None)
            if args.method == "soft_revision_v6":
                for event in diagnostics["revision_events"]:
                    event["original_hard_token_string"] = tokenizer.convert_ids_to_tokens(
                        event["original_hard_token_id"])
                    event["current_top1_token_string"] = tokenizer.convert_ids_to_tokens(
                        event["current_top1_token_id"])
                    event["new_hard_token_string"] = (
                        tokenizer.convert_ids_to_tokens(event["new_token_id"])
                        if event["new_token_id"] is not None else None)
            row = {
                "index": index,
                "full_response": response,
                "steps": int(steps),
                "nfe": int(diagnostics["nfe"]),
                "latency": latency,
                "diagnostics": diagnostics,
            }
            if args.dataset == "math500":
                prediction = math500_extract_answer(response)
                row.update({
                    "prediction": prediction,
                    "is_correct": bool(math500_is_equiv(prediction, target)),
                })
            elif args.dataset == "countdown":
                prediction = countdown_extract_answer(response)
                row.update({
                    "prediction": prediction,
                    "is_correct": bool(countdown_is_correct(
                        prediction, target["target"], target["numbers"])),
                })
            elif args.dataset == "humaneval":
                row.update({
                    "task_id": target["task_id"],
                    "completion": humaneval_extract_answer(response, target),
                    "is_correct": None,
                })
            else:
                response_for_extract = "```python\n" + response
                row.update({
                    "task_id": target["task_id"],
                    "completion": mbpp_extract_answer(
                        response_for_extract, target["entry_point"]),
                    "is_correct": None,
                })
            checkpoint.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(
                f"{args.dataset}/{args.method} index={index} "
                f"nfe={steps} latency={latency:.3f}s", flush=True)

    if start_index != 0 or end_index != len(benchmark) or args.worker_name:
        rows = sorted(read_jsonl(checkpoint_path), key=lambda row: row["index"])
        if len(rows) != len(expected_indices) or [row["index"] for row in rows] != expected_indices:
            raise RuntimeError(
                f"Incomplete worker {args.dataset}/{args.method}: "
                f"{len(rows)}/{len(expected_indices)}")
        print(f"Worker complete: {checkpoint_path}", flush=True)
        return
    finalize(args.dataset, args.method, checkpoint_path, final_path)


if __name__ == "__main__":
    main()
