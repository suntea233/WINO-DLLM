"""Frozen V6 evaluation for Sudoku and ARC using repository task settings."""

import argparse
import glob
import json
import os
import time

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from dataset_utils.arc import arc_doc_to_text, arc_extract_answer, arc_is_correct
from dataset_utils.sudoku import (
    sudoku_doc_to_text, sudoku_evaluate_and_extract, sudoku_read_data)
from decoding import decoding_wino_soft_revision_v6
from modeling_llada import LLaDAModelLM


HERE = os.path.dirname(__file__)
MODEL_PATH = "/root/lx/ReMix-DLLM/LLaDA/models/LLaDA-8B-Instruct"
RESULT_DIR = os.path.join(HERE, "results", "soft_revision_v6_reasoning")
TASKS = {
    "sudoku": {"threshold": 0.7, "split": None, "total": 500},
    "arc_easy": {"threshold": 0.5, "split": "test", "total": 2376},
    "arc_challenge": {"threshold": 0.5, "split": "validation", "total": 299},
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
    if name == "sudoku":
        return sudoku_read_data(os.path.join(
            HERE, "data", "sudoku", "4x4_test_sudoku.csv"))
    config = "ARC-Easy" if name == "arc_easy" else "ARC-Challenge"
    return load_dataset("ai2_arc", config, trust_remote_code=True)[TASKS[name]["split"]]


def prompt_and_target(name, doc):
    if name == "sudoku":
        context, _ = sudoku_doc_to_text(doc)
        return context, doc["Solution"]
    return arc_doc_to_text(doc)


def collect_rows(name, output_dir):
    paths = glob.glob(os.path.join(output_dir, f".{name}*_samples.jsonl"))
    by_index = {}
    for path in paths:
        for row in read_jsonl(path):
            by_index[row["index"]] = row
    return [by_index[index] for index in sorted(by_index)]


def finalize(name, output_dir):
    rows = collect_rows(name, output_dir)
    total = TASKS[name]["total"]
    if len(rows) != total or [row["index"] for row in rows] != list(range(total)):
        raise RuntimeError(f"Incomplete {name}: {len(rows)}/{total}")
    if name == "sudoku":
        total_cells = sum(row["total_cells"] for row in rows)
        correct_cells = sum(row["correct_cells"] for row in rows)
        exact = sum(row["correct_cells"] == row["total_cells"] for row in rows)
        metrics = {
            "accuracy": correct_cells / total_cells,
            "correct_cells": correct_cells,
            "total_empty_cells": total_cells,
            "exact_puzzle_accuracy": exact / total,
            "exact_puzzles": exact,
            "total": total,
        }
    else:
        correct = sum(row["is_correct"] for row in rows)
        metrics = {"accuracy": correct / total, "correct": correct, "total": total}
    metrics.update({
        "average_nfe": sum(row["nfe"] for row in rows) / total,
        "average_latency": sum(row["latency"] for row in rows) / total,
        "total_h_to_s": sum(row["diagnostics"]["num_h_to_soft_revisions"] for row in rows),
        "total_identity_corrections": sum(
            row["diagnostics"]["identity_corrections"] for row in rows),
    })
    result = {
        "dataset": name,
        "method": "soft_revision_v6_identity_correction_only",
        "configuration": {
            "model": MODEL_PATH, "gen_length": 256, "block_length": 128,
            "temperature": 0.0, "threshold": TASKS[name]["threshold"],
            "threshold_back": 0.9, "beta_mix": 0.5,
            "split": TASKS[name]["split"],
        },
        "metrics": metrics,
        "samples": rows,
    }
    atomic_json(os.path.join(output_dir, f"{name}_soft_revision_v6.json"), result)
    print(json.dumps(metrics, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=TASKS, required=True)
    parser.add_argument("--output-dir", default=RESULT_DIR)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int)
    parser.add_argument("--worker-name")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    suffix = f"_{args.worker_name}" if args.worker_name else ""
    checkpoint = os.path.join(
        args.output_dir, f".{args.dataset}{suffix}_samples.jsonl")
    rows = read_jsonl(checkpoint)
    completed = {row["index"] for row in rows}
    dataset = load_task(args.dataset)
    if len(dataset) != TASKS[args.dataset]["total"]:
        raise RuntimeError(f"Unexpected {args.dataset} size: {len(dataset)}")
    start = max(0, args.start_index)
    end = len(dataset) if args.end_index is None else min(args.end_index, len(dataset))
    indices = list(range(start, end))
    print(f"{args.dataset}: resuming {len(completed)}/{len(indices)} "
          f"for [{start}, {end})", flush=True)

    model = LLaDAModelLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16).cuda().eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    context, _ = prompt_and_target(args.dataset, dataset[0])
    prompt = tokenizer.apply_chat_template(
        context, add_generation_prompt=True, tokenize=False) + "<reasoning>"
    warm = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
    decoding_wino_soft_revision_v6(
        model, warm, gen_length=256, block_length=128, temperature=0.0,
        threshold=TASKS[args.dataset]["threshold"], threshold_back=0.9,
        beta_mix=0.5)

    with open(checkpoint, "a", encoding="utf-8", buffering=1) as output:
        for index in indices:
            doc = dataset[index]
            if index in completed:
                continue
            context, target = prompt_and_target(args.dataset, doc)
            prompt = tokenizer.apply_chat_template(
                context, add_generation_prompt=True, tokenize=False) + "<reasoning>"
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
            torch.cuda.synchronize()
            started = time.perf_counter()
            generated, steps, diagnostics = decoding_wino_soft_revision_v6(
                model, input_ids, gen_length=256, block_length=128,
                temperature=0.0, threshold=TASKS[args.dataset]["threshold"],
                threshold_back=0.9, beta_mix=0.5, return_diagnostics=True)
            torch.cuda.synchronize()
            latency = time.perf_counter() - started
            response = tokenizer.batch_decode(
                generated[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
            compact_diagnostics = {
                "num_suspicious_hard_tokens": diagnostics["num_suspicious_hard_tokens"],
                "num_h_to_soft_revisions": diagnostics["num_h_to_soft_revisions"],
                "num_h_to_s_to_h": diagnostics["num_h_to_s_to_h"],
                "num_h_to_s_to_m": diagnostics["num_h_to_s_to_m"],
                "identity_corrections": sum(
                    event.get("commit_reason") == "different_identity_correction"
                    for event in diagnostics["revision_events"]),
            }
            row = {"index": index, "full_response": response,
                   "nfe": int(steps), "latency": latency,
                   "diagnostics": compact_diagnostics}
            if args.dataset == "sudoku":
                correct, total_cells, prediction = sudoku_evaluate_and_extract(
                    response, target, doc["Puzzle"])
                row.update({"prediction": prediction, "correct_cells": int(correct),
                            "total_cells": int(total_cells),
                            "is_exact": bool(correct == total_cells)})
            else:
                prediction = arc_extract_answer(response)
                row.update({"prediction": prediction,
                            "is_correct": bool(arc_is_correct(prediction, target)),
                            "target": target})
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"{args.dataset} {index + 1}/{len(dataset)} nfe={steps}", flush=True)
    collected = collect_rows(args.dataset, args.output_dir)
    if len(collected) == TASKS[args.dataset]["total"]:
        finalize(args.dataset, args.output_dir)
    else:
        print(f"Shard complete; combined coverage {len(collected)}/"
              f"{TASKS[args.dataset]['total']}", flush=True)


if __name__ == "__main__":
    main()
