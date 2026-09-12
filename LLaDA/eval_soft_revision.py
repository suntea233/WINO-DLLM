"""GSM8K evaluation for WINO remasking and Hard-to-Soft Revision.

This keeps the repository's official GSM8K prompt, answer extraction, model,
and WINO generation settings.  JSONL checkpoints make a long run resumable;
the requested JSON artifacts are written atomically when all samples finish.
"""

import argparse
import json
import os
import time

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from dataset_utils.gsm8k import (
    gsm8k_doc_to_text,
    gsm8k_extract_answer,
    gsm8k_is_correct,
)
from decoding import decoding_wino_remask, decoding_wino_soft_revision
from modeling_llada import LLaDAModelLM


DEFAULT_MODEL = "/root/lx/ReMix-DLLM/LLaDA/models/LLaDA-8B-Instruct"
DEFAULT_DATA = "/root/lx/ReMix-DLLM/LLaDA/data/gsm8k"
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "results", "soft_revision")


def _read_jsonl(path):
    rows = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def _atomic_json(path, value):
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
    os.replace(temporary, path)


def _mean(values):
    return sum(values) / len(values) if values else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("wino_remask", "soft_revision"), required=True)
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--data-path", default=DEFAULT_DATA)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-samples", type=int)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int)
    parser.add_argument("--worker-name")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    worker_suffix = f"_{args.worker_name}" if args.worker_name else ""
    checkpoint_path = os.path.join(
        args.output_dir, f".{args.method}{worker_suffix}_samples.jsonl")
    final_name = ("gsm8k_soft_revision.json" if args.method == "soft_revision"
                  else "gsm8k_wino_remask.json")
    final_path = os.path.join(args.output_dir, final_name)

    dataset = load_dataset(args.data_path, "main", trust_remote_code=True)["test"]
    total = len(dataset) if args.num_samples is None else min(args.num_samples, len(dataset))
    start_index = max(0, args.start_index)
    end_index = total if args.end_index is None else min(args.end_index, total)
    expected_indices = list(range(start_index, end_index))
    existing = [] if args.no_resume else _read_jsonl(checkpoint_path)
    completed = {row["index"] for row in existing}
    if completed:
        print(f"Resuming {args.method}: {len(completed)}/{len(expected_indices)} worker samples already present", flush=True)

    model = LLaDAModelLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16).cuda().eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    generation_fn = (decoding_wino_soft_revision if args.method == "soft_revision"
                     else decoding_wino_remask)

    # Match the existing evaluator's single warm-up, using a real formatted prompt.
    warm_context, _ = gsm8k_doc_to_text(dataset[start_index])
    warm_prompt = tokenizer.apply_chat_template(
        warm_context, add_generation_prompt=True, tokenize=False) + "<reasoning>"
    warm_ids = tokenizer(warm_prompt, return_tensors="pt").input_ids.cuda()
    generation_fn(model, warm_ids, gen_length=256, block_length=128,
                  temperature=0.0, threshold=0.6, threshold_back=0.9)

    mode = "w" if args.no_resume else "a"
    with open(checkpoint_path, mode, encoding="utf-8", buffering=1) as checkpoint:
        for index in expected_indices:
            if index in completed:
                continue
            doc = dataset[index]
            context, target = gsm8k_doc_to_text(doc)
            prompt = tokenizer.apply_chat_template(
                context, add_generation_prompt=True, tokenize=False) + "<reasoning>"
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
            torch.cuda.synchronize()
            started = time.perf_counter()
            generated, steps, diagnostics = generation_fn(
                model, input_ids, gen_length=256, block_length=128,
                temperature=0.0, threshold=0.6, threshold_back=0.9,
                return_diagnostics=True)
            torch.cuda.synchronize()
            latency = time.perf_counter() - started
            response = tokenizer.batch_decode(
                generated[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
            prediction = gsm8k_extract_answer(response)
            is_correct = bool(gsm8k_is_correct(prediction, target))
            diagnostics.pop("round_trace", None)
            row = {
                "index": index,
                "full_response": response,
                "prediction": prediction,
                "is_correct": is_correct,
                "steps": int(steps),
                "nfe": int(diagnostics["nfe"]),
                "latency": latency,
                "diagnostics": diagnostics,
            }
            checkpoint.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(
                f"{args.method} index={index} worker_progress="
                f"{sum(1 for done in expected_indices if done <= index)}/{len(expected_indices)} "
                f"correct={int(is_correct)} "
                f"nfe={steps} latency={latency:.3f}s", flush=True)

    rows = sorted(_read_jsonl(checkpoint_path), key=lambda row: row["index"])
    rows = [row for row in rows if row["index"] in set(expected_indices)]
    if len(rows) != len(expected_indices) or [row["index"] for row in rows] != expected_indices:
        raise RuntimeError(
            f"Incomplete or duplicate checkpoint: found {len(rows)} of {len(expected_indices)}")
    if start_index != 0 or end_index != total or args.worker_name:
        print(f"Worker complete: {checkpoint_path}", flush=True)
        return
    aggregate = {
        "accuracy": sum(row["is_correct"] for row in rows) / total,
        "correct": sum(row["is_correct"] for row in rows),
        "total": total,
        "average_nfe": _mean([row["nfe"] for row in rows]),
        "average_decoding_rounds": _mean([row["steps"] for row in rows]),
        "average_latency": _mean([row["latency"] for row in rows]),
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
        "total_unique_revised_positions_per_sample": sum(
            row["diagnostics"]["num_unique_revised_positions"] for row in rows),
    }
    soft_total = aggregate["total_h_to_soft_revisions"]
    aggregate["soft_recovery_rate"] = (
        aggregate["total_h_to_s_to_h"] / soft_total if soft_total else 0.0)
    result = {
        "method": args.method,
        "dataset": "gsm8k_test",
        "configuration": {
            "model": args.model_path,
            "gen_length": 256,
            "block_length": 128,
            "temperature": 0.0,
            "threshold": 0.6,
            "threshold_back": 0.9,
            "beta_mix": 0.5 if args.method == "soft_revision" else None,
            "prompt_suffix": "<reasoning>",
        },
        "metrics": aggregate,
        "samples": rows,
    }
    _atomic_json(final_path, result)

    if args.method == "soft_revision":
        trace_path = os.path.join(args.output_dir, "revision_trace.jsonl")
        with open(trace_path + ".tmp", "w", encoding="utf-8") as handle:
            for row in rows:
                for event in row["diagnostics"]["revision_events"]:
                    trace_row = {"sample_index": row["index"], **event}
                    handle.write(json.dumps(trace_row, ensure_ascii=False) + "\n")
        os.replace(trace_path + ".tmp", trace_path)
    print(json.dumps(aggregate, indent=2), flush=True)
    print(f"Saved {final_path}", flush=True)


if __name__ == "__main__":
    main()
