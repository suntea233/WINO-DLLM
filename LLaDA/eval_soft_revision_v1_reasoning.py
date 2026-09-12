"""Frozen one-round Soft Revision V1 evaluation for Sudoku and ARC."""

import argparse
import json
import os
import time

import torch
from transformers import AutoTokenizer

from decoding import decoding_wino_soft_revision
from eval_soft_revision_v6_reasoning import (
    MODEL_PATH, TASKS, load_task, prompt_and_target)
from dataset_utils.arc import arc_extract_answer, arc_is_correct
from dataset_utils.sudoku import sudoku_evaluate_and_extract
from modeling_llada import LLaDAModelLM


HERE = os.path.dirname(__file__)
RESULT_DIR = os.path.join(HERE, "results", "soft_revision")


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


def finalize(name, checkpoint, output_dir):
    rows = sorted(read_jsonl(checkpoint), key=lambda row: row["index"])
    total = TASKS[name]["total"]
    if len(rows) != total or [row["index"] for row in rows] != list(range(total)):
        raise RuntimeError(f"Incomplete {name}: {len(rows)}/{total}")
    if name == "sudoku":
        total_cells = sum(row["total_cells"] for row in rows)
        correct_cells = sum(row["correct_cells"] for row in rows)
        exact = sum(row["is_exact"] for row in rows)
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
        "total_h_to_s": sum(
            row["diagnostics"]["num_h_to_soft_revisions"] for row in rows),
        "total_h_to_s_to_h": sum(
            row["diagnostics"]["num_h_to_s_to_h"] for row in rows),
        "total_h_to_s_to_m": sum(
            row["diagnostics"]["num_h_to_s_to_m"] for row in rows),
    })
    result = {
        "dataset": name,
        "method": "soft_revision_v1_one_round_restore_original",
        "configuration": {
            "model": MODEL_PATH,
            "gen_length": 256,
            "block_length": 128,
            "temperature": 0.0,
            "threshold": TASKS[name]["threshold"],
            "threshold_back": 0.9,
            "beta_mix": 0.5,
            "split": TASKS[name]["split"],
            "lifecycle": "H(a)->S; next round p_old>=0.9 -> H(a), else MASK",
        },
        "metrics": metrics,
        "samples": rows,
    }
    atomic_json(os.path.join(output_dir, f"{name}_soft_revision.json"), result)
    print(json.dumps(metrics, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=TASKS, required=True)
    parser.add_argument("--output-dir", default=RESULT_DIR)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint = os.path.join(
        args.output_dir, f".{args.dataset}_soft_revision_samples.jsonl")
    completed = {row["index"] for row in read_jsonl(checkpoint)}
    dataset = load_task(args.dataset)
    if len(dataset) != TASKS[args.dataset]["total"]:
        raise RuntimeError(f"Unexpected {args.dataset} size: {len(dataset)}")
    print(f"{args.dataset}/V1: resuming {len(completed)}/{len(dataset)}", flush=True)

    model = LLaDAModelLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16).cuda().eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    context, _ = prompt_and_target(args.dataset, dataset[0])
    prompt = tokenizer.apply_chat_template(
        context, add_generation_prompt=True, tokenize=False) + "<reasoning>"
    warm_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
    decoding_wino_soft_revision(
        model, warm_ids, gen_length=256, block_length=128, temperature=0.0,
        threshold=TASKS[args.dataset]["threshold"], threshold_back=0.9,
        beta_mix=0.5)

    with open(checkpoint, "a", encoding="utf-8", buffering=1) as output:
        for index, doc in enumerate(dataset):
            if index in completed:
                continue
            context, target = prompt_and_target(args.dataset, doc)
            prompt = tokenizer.apply_chat_template(
                context, add_generation_prompt=True, tokenize=False) + "<reasoning>"
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
            torch.cuda.synchronize()
            started = time.perf_counter()
            generated, steps, diagnostics = decoding_wino_soft_revision(
                model, input_ids, gen_length=256, block_length=128,
                temperature=0.0, threshold=TASKS[args.dataset]["threshold"],
                threshold_back=0.9, beta_mix=0.5, return_diagnostics=True)
            torch.cuda.synchronize()
            latency = time.perf_counter() - started
            response = tokenizer.batch_decode(
                generated[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
            compact = {
                "num_suspicious_hard_tokens": diagnostics["num_suspicious_hard_tokens"],
                "num_h_to_soft_revisions": diagnostics["num_h_to_soft_revisions"],
                "num_h_to_s_to_h": diagnostics["num_h_to_s_to_h"],
                "num_h_to_s_to_m": diagnostics["num_h_to_s_to_m"],
            }
            row = {"index": index, "full_response": response, "nfe": int(steps),
                   "latency": latency, "diagnostics": compact}
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
            print(f"{args.dataset}/V1 {index + 1}/{len(dataset)} nfe={steps}", flush=True)
    finalize(args.dataset, checkpoint, args.output_dir)


if __name__ == "__main__":
    main()
