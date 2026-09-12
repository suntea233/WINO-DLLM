"""Evaluate frozen Soft Revision V2 on GSM8K without changing V1 artifacts."""

import argparse
import csv
import json
import os
import time

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from dataset_utils.gsm8k import gsm8k_doc_to_text, gsm8k_extract_answer, gsm8k_is_correct
from decoding import decoding_wino_soft_revision_v2
from modeling_llada import LLaDAModelLM


MODEL_PATH = "/root/lx/ReMix-DLLM/LLaDA/models/LLaDA-8B-Instruct"
DATA_PATH = "/root/lx/ReMix-DLLM/LLaDA/data/gsm8k"
V1_RESULT_DIR = os.path.join(os.path.dirname(__file__), "results", "soft_revision")
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "results", "soft_revision_v2")


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


def finalize(checkpoint_path, output_dir, total):
    rows = sorted(read_jsonl(checkpoint_path), key=lambda row: row["index"])
    if len(rows) != total or [row["index"] for row in rows] != list(range(total)):
        raise RuntimeError(f"Incomplete V2 checkpoint: {len(rows)}/{total}")

    hsh_events = [
        event for row in rows for event in row["diagnostics"]["revision_events"]
        if event.get("outcome") == "H->S->H"]
    hsm_count = sum(
        row["diagnostics"]["num_h_to_s_to_m"] for row in rows)
    same_count = sum(event["same_token"] for event in hsh_events)
    changed_count = len(hsh_events) - same_count
    correct = sum(row["is_correct"] for row in rows)
    metrics = {
        "accuracy": correct / total,
        "correct": correct,
        "total": total,
        "average_nfe": sum(row["nfe"] for row in rows) / total,
        "average_latency": sum(row["latency"] for row in rows) / total,
        "total_h_to_s": sum(
            row["diagnostics"]["num_h_to_soft_revisions"] for row in rows),
        "total_h_to_s_to_h": len(hsh_events),
        "total_h_to_s_to_m": hsm_count,
        "same_token_h_to_s_to_h_count": same_count,
        "same_token_h_to_s_to_h_percentage": (
            100.0 * same_count / len(hsh_events) if hsh_events else 0.0),
        "changed_token_h_to_s_to_h_count": changed_count,
        "changed_token_h_to_s_to_h_percentage": (
            100.0 * changed_count / len(hsh_events) if hsh_events else 0.0),
    }
    result = {
        "dataset": "gsm8k_test",
        "method": "soft_revision_v2",
        "configuration": {
            "model": MODEL_PATH,
            "gen_length": 256,
            "block_length": 128,
            "temperature": 0.0,
            "threshold": 0.6,
            "threshold_back": 0.9,
            "beta_mix": 0.5,
            "pass_action": "commit current verification posterior top-1",
        },
        "metrics": metrics,
        "samples": rows,
    }
    atomic_json(os.path.join(output_dir, "gsm8k_soft_revision_v2.json"), result)

    trace_path = os.path.join(output_dir, "revision_trace_v2.jsonl")
    with open(trace_path + ".tmp", "w", encoding="utf-8") as handle:
        for row in rows:
            for event in row["diagnostics"]["revision_events"]:
                handle.write(json.dumps(
                    {"sample_index": row["index"], **event}, ensure_ascii=False) + "\n")
    os.replace(trace_path + ".tmp", trace_path)

    if total == 1319:
        with open(os.path.join(V1_RESULT_DIR, "gsm8k_wino_remask.json"),
                  encoding="utf-8") as handle:
            wino = json.load(handle)
        with open(os.path.join(V1_RESULT_DIR, "gsm8k_soft_revision.json"),
                  encoding="utf-8") as handle:
            v1 = json.load(handle)
        wino_by_id = {row["index"]: row for row in wino["samples"]}
        recovered = sum(
            not wino_by_id[row["index"]]["is_correct"] and row["is_correct"]
            for row in rows)
        regressed = sum(
            wino_by_id[row["index"]]["is_correct"] and not row["is_correct"]
            for row in rows)
        metrics.update({
            "wino_wrong_to_v2_correct": recovered,
            "wino_correct_to_v2_wrong": regressed,
            "net_recovery_vs_wino": recovered - regressed,
        })
        # Re-write after adding paired metrics.
        result["metrics"] = metrics
        atomic_json(os.path.join(output_dir, "gsm8k_soft_revision_v2.json"), result)
        with open(os.path.join(output_dir, "summary.csv"), "w", newline="",
                  encoding="utf-8") as handle:
            fields = ["Method", "Accuracy", "Avg NFE", "Avg Latency",
                      "H->S", "H->S->H", "H->S->M", "Same", "Changed"]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerow({
                "Method": "WINO", "Accuracy": wino["metrics"]["accuracy"],
                "Avg NFE": wino["metrics"]["average_nfe"],
                "Avg Latency": wino["metrics"]["average_latency"]})
            writer.writerow({
                "Method": "Soft Revision V1", "Accuracy": v1["metrics"]["accuracy"],
                "Avg NFE": v1["metrics"]["average_nfe"],
                "Avg Latency": v1["metrics"]["average_latency"],
                "H->S": v1["metrics"]["total_h_to_soft_revisions"],
                "H->S->H": v1["metrics"]["total_h_to_s_to_h"],
                "H->S->M": v1["metrics"]["total_h_to_s_to_m"],
                "Same": v1["metrics"]["total_h_to_s_to_h"], "Changed": 0})
            writer.writerow({
                "Method": "Soft Revision V2", "Accuracy": metrics["accuracy"],
                "Avg NFE": metrics["average_nfe"],
                "Avg Latency": metrics["average_latency"],
                "H->S": metrics["total_h_to_s"],
                "H->S->H": metrics["total_h_to_s_to_h"],
                "H->S->M": metrics["total_h_to_s_to_m"],
                "Same": same_count, "Changed": changed_count})

        report = f"""Soft Revision V2: GSM8K
=========================

Frozen setting: LLaDA-8B-Instruct, gen_length=256, block_length=128,
temperature=0, threshold=0.6, threshold_back=0.9, beta_mix=0.5.
V2 changes only the passing action to commit the current verifier posterior top-1.

Accuracy: {metrics['accuracy']:.4%} ({correct}/{total})
Average NFE: {metrics['average_nfe']:.2f}
Average latency: {metrics['average_latency']:.3f}s
Total H->S: {metrics['total_h_to_s']}
H->S->H: {metrics['total_h_to_s_to_h']}
H->S->M: {metrics['total_h_to_s_to_m']}
H(a)->S->H(a): {same_count} ({metrics['same_token_h_to_s_to_h_percentage']:.2f}%)
H(a)->S->H(b), a!=b: {changed_count} ({metrics['changed_token_h_to_s_to_h_percentage']:.2f}%)

Paired against WINO
-------------------
WINO wrong -> V2 correct: {recovered}
WINO correct -> V2 wrong: {regressed}
Net recovery: {recovered - regressed:+d}

Reference
---------
WINO: accuracy 77.56%, average NFE 46.25, H->M->H changed-token rate 8.94%.
Soft Revision V1: accuracy 78.32%, average NFE 42.93,
H->S->H changed-token rate 0% by construction.

Interpretation
--------------
The unchanged pass criterion requires posterior probability >= 0.9 for the old
HARD token. Therefore that token is necessarily the posterior top-1 whenever
the pass action executes. V2's requested top-1 assignment cannot change token
identity under this frozen verification rule.
"""
        with open(os.path.join(output_dir, "report_soft_revision_v2.txt"), "w",
                  encoding="utf-8") as handle:
            handle.write(report)
    print(json.dumps(metrics, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-samples", type=int, default=1319)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.output_dir, ".gsm8k_v2_samples.jsonl")
    dataset = load_dataset(DATA_PATH, "main", trust_remote_code=True)["test"]
    total = min(args.num_samples, len(dataset))
    completed = {row["index"] for row in read_jsonl(checkpoint_path)}
    print(f"Soft Revision V2: resuming {len(completed)}/{total}", flush=True)

    model = LLaDAModelLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16).cuda().eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    context, _ = gsm8k_doc_to_text(dataset[0])
    prompt = tokenizer.apply_chat_template(
        context, add_generation_prompt=True, tokenize=False) + "<reasoning>"
    warm_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
    decoding_wino_soft_revision_v2(
        model, warm_ids, gen_length=256, block_length=128, temperature=0.0,
        threshold=0.6, threshold_back=0.9)

    with open(checkpoint_path, "a", encoding="utf-8", buffering=1) as checkpoint:
        for index in range(total):
            if index in completed:
                continue
            context, target = gsm8k_doc_to_text(dataset[index])
            prompt = tokenizer.apply_chat_template(
                context, add_generation_prompt=True, tokenize=False) + "<reasoning>"
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
            torch.cuda.synchronize()
            started = time.perf_counter()
            output, steps, diagnostics = decoding_wino_soft_revision_v2(
                model, input_ids, gen_length=256, block_length=128,
                temperature=0.0, threshold=0.6, threshold_back=0.9,
                return_diagnostics=True)
            torch.cuda.synchronize()
            latency = time.perf_counter() - started
            response = tokenizer.batch_decode(
                output[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
            prediction = gsm8k_extract_answer(response)
            is_correct = bool(gsm8k_is_correct(prediction, target))
            diagnostics.pop("round_trace", None)
            for event in diagnostics["revision_events"]:
                event["original_token_id"] = event["token_id"]
                event["original_token_string"] = tokenizer.convert_ids_to_tokens(
                    event["token_id"])
                if event.get("outcome") == "H->S->H":
                    event["new_token_string"] = tokenizer.convert_ids_to_tokens(
                        event["new_token_id"])
            checkpoint.write(json.dumps({
                "index": index,
                "full_response": response,
                "prediction": prediction,
                "is_correct": is_correct,
                "steps": int(steps),
                "nfe": diagnostics["nfe"],
                "latency": latency,
                "diagnostics": diagnostics,
            }, ensure_ascii=False) + "\n")
            print(f"V2 {index + 1}/{total} nfe={steps}", flush=True)
    finalize(checkpoint_path, args.output_dir, total)


if __name__ == "__main__":
    main()
