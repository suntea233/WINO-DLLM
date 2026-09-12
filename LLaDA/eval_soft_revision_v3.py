"""Evaluate renewable multi-round Soft Revision V3 on GSM8K."""

import argparse
import json
import os
import statistics
import time

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from dataset_utils.gsm8k import gsm8k_doc_to_text, gsm8k_extract_answer, gsm8k_is_correct
from decoding import decoding_wino_soft_revision_v3
from modeling_llada import LLaDAModelLM


MODEL_PATH = "/root/lx/ReMix-DLLM/LLaDA/models/LLaDA-8B-Instruct"
DATA_PATH = "/root/lx/ReMix-DLLM/LLaDA/data/gsm8k"
V1_DIR = os.path.join(os.path.dirname(__file__), "results", "soft_revision")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "results", "soft_revision_v3")


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


def lifetime_summary(events):
    lifetimes = [event["soft_lifetime"] for event in events]
    histogram = {
        "1": sum(value == 1 for value in lifetimes),
        "2": sum(value == 2 for value in lifetimes),
        "3": sum(value == 3 for value in lifetimes),
        "4": sum(value == 4 for value in lifetimes),
        "5+": sum(value >= 5 for value in lifetimes),
    }
    ending_h = sum(event["final_transition"] == "S->H" for event in events)
    ending_m = len(events) - ending_h
    return {
        "episodes": len(events),
        "total_s_to_s": sum(event["num_s_to_s_renewals"] for event in events),
        "total_s_to_h": ending_h,
        "total_s_to_m": ending_m,
        "average_soft_lifetime": statistics.fmean(lifetimes) if lifetimes else 0.0,
        "median_soft_lifetime": statistics.median(lifetimes) if lifetimes else 0.0,
        "lifetime_histogram": histogram,
        "fraction_ending_in_h": ending_h / len(events) if events else 0.0,
        "fraction_ending_in_m": ending_m / len(events) if events else 0.0,
    }


def finalize(checkpoint_path, output_dir, total):
    rows = sorted(read_jsonl(checkpoint_path), key=lambda row: row["index"])
    if len(rows) != total or [row["index"] for row in rows] != list(range(total)):
        raise RuntimeError(f"Incomplete V3 checkpoint: {len(rows)}/{total}")
    if not all(row["nfe"] == row["steps"] for row in rows):
        raise RuntimeError("V3 NFE must equal normal decoding rounds")
    if not all(row["diagnostics"]["unresolved_soft_states"] == 0 for row in rows):
        raise RuntimeError("V3 terminated with unresolved SOFT state")

    events = [event for row in rows for event in row["diagnostics"]["revision_events"]]
    correct_events = [
        event for row in rows if row["is_correct"]
        for event in row["diagnostics"]["revision_events"]]
    wrong_events = [
        event for row in rows if not row["is_correct"]
        for event in row["diagnostics"]["revision_events"]]
    overall = lifetime_summary(events)
    correct_lifetimes = lifetime_summary(correct_events)
    wrong_lifetimes = lifetime_summary(wrong_events)
    correct = sum(row["is_correct"] for row in rows)
    metrics = {
        "accuracy": correct / total,
        "correct": correct,
        "total": total,
        "average_nfe": sum(row["nfe"] for row in rows) / total,
        "average_latency": sum(row["latency"] for row in rows) / total,
        "total_h_to_s": overall["episodes"],
        "total_s_to_s": overall["total_s_to_s"],
        "total_s_to_h": overall["total_s_to_h"],
        "total_s_to_m": overall["total_s_to_m"],
        "stalled_soft_to_mask_fallbacks": sum(
            row["diagnostics"]["num_stalled_soft_to_mask_fallbacks"] for row in rows),
        "average_soft_lifetime": overall["average_soft_lifetime"],
        "median_soft_lifetime": overall["median_soft_lifetime"],
        "lifetime_histogram": overall["lifetime_histogram"],
        "fraction_ending_in_h": overall["fraction_ending_in_h"],
        "fraction_ending_in_m": overall["fraction_ending_in_m"],
    }
    result = {
        "dataset": "gsm8k_test",
        "method": "soft_revision_v3_renewable",
        "configuration": {
            "model": MODEL_PATH,
            "gen_length": 256,
            "block_length": 128,
            "temperature": 0.0,
            "threshold": 0.6,
            "threshold_back": 0.9,
            "beta_mix": 0.5,
            "lifecycle": "p_old>=0.9: S->H; 0.6<=p_old<0.9: S->S; p_old<0.6: S->M",
        },
        "metrics": metrics,
        "correct_sample_lifetimes": correct_lifetimes,
        "wrong_sample_lifetimes": wrong_lifetimes,
        "samples": rows,
    }

    summary = {
        "configuration": result["configuration"],
        "metrics": metrics,
        "correct_sample_lifetimes": correct_lifetimes,
        "wrong_sample_lifetimes": wrong_lifetimes,
    }
    if total == 1319:
        with open(os.path.join(V1_DIR, "gsm8k_wino_remask.json"), encoding="utf-8") as handle:
            wino = json.load(handle)
        with open(os.path.join(V1_DIR, "gsm8k_soft_revision.json"), encoding="utf-8") as handle:
            v1 = json.load(handle)
        wino_by_id = {row["index"]: row for row in wino["samples"]}
        v1_by_id = {row["index"]: row for row in v1["samples"]}
        v1_correct_v3_wrong = sum(
            v1_by_id[row["index"]]["is_correct"] and not row["is_correct"]
            for row in rows)
        v1_wrong_v3_correct = sum(
            not v1_by_id[row["index"]]["is_correct"] and row["is_correct"]
            for row in rows)
        wino_wrong_v3_correct = sum(
            not wino_by_id[row["index"]]["is_correct"] and row["is_correct"]
            for row in rows)
        wino_correct_v3_wrong = sum(
            wino_by_id[row["index"]]["is_correct"] and not row["is_correct"]
            for row in rows)
        paired = {
            "v1_correct_to_v3_wrong": v1_correct_v3_wrong,
            "v1_wrong_to_v3_correct": v1_wrong_v3_correct,
            "net_recovery_vs_v1": v1_wrong_v3_correct - v1_correct_v3_wrong,
            "wino_wrong_to_v3_correct": wino_wrong_v3_correct,
            "wino_correct_to_v3_wrong": wino_correct_v3_wrong,
            "net_recovery_vs_wino": wino_wrong_v3_correct - wino_correct_v3_wrong,
        }
        result["paired"] = paired
        summary["paired"] = paired

    os.makedirs(output_dir, exist_ok=True)
    atomic_json(os.path.join(output_dir, "gsm8k_soft_revision_v3.json"), result)
    atomic_json(os.path.join(output_dir, "summary_v3.json"), summary)
    trace_path = os.path.join(output_dir, "revision_trace_v3.jsonl")
    with open(trace_path + ".tmp", "w", encoding="utf-8") as handle:
        for row in rows:
            for event in row["diagnostics"]["revision_events"]:
                handle.write(json.dumps(
                    {"sample_index": row["index"], **event}, ensure_ascii=False) + "\n")
    os.replace(trace_path + ".tmp", trace_path)

    if total == 1319:
        paired = result["paired"]
        report = f"""Renewable Soft Revision V3: GSM8K
==================================

Frozen setting: LLaDA-8B-Instruct, gen_length=256, block_length=128,
temperature=0, threshold=0.6, threshold_back=0.9, beta_mix=0.5.
No auxiliary backbone forward is used.

Accuracy: {metrics['accuracy']:.4%} ({correct}/{total})
Average NFE: {metrics['average_nfe']:.2f}
Average latency: {metrics['average_latency']:.3f}s

Total H->S: {metrics['total_h_to_s']}
Total S->S: {metrics['total_s_to_s']}
Total S->H: {metrics['total_s_to_h']}
Total S->M: {metrics['total_s_to_m']}
Exact fixed-point S->M termination fallbacks: {metrics['stalled_soft_to_mask_fallbacks']}
Average Soft lifetime: {metrics['average_soft_lifetime']:.3f} rounds
Median Soft lifetime: {metrics['median_soft_lifetime']:.1f} rounds
Lifetime histogram: {json.dumps(metrics['lifetime_histogram'])}
Fraction ending in H: {metrics['fraction_ending_in_h']:.2%}
Fraction ending in M: {metrics['fraction_ending_in_m']:.2%}

Correct samples
---------------
Episodes: {correct_lifetimes['episodes']}
Average/median lifetime: {correct_lifetimes['average_soft_lifetime']:.3f} / {correct_lifetimes['median_soft_lifetime']:.1f}
Histogram: {json.dumps(correct_lifetimes['lifetime_histogram'])}
Fraction ending H/M: {correct_lifetimes['fraction_ending_in_h']:.2%} / {correct_lifetimes['fraction_ending_in_m']:.2%}

Wrong samples
-------------
Episodes: {wrong_lifetimes['episodes']}
Average/median lifetime: {wrong_lifetimes['average_soft_lifetime']:.3f} / {wrong_lifetimes['median_soft_lifetime']:.1f}
Histogram: {json.dumps(wrong_lifetimes['lifetime_histogram'])}
Fraction ending H/M: {wrong_lifetimes['fraction_ending_in_h']:.2%} / {wrong_lifetimes['fraction_ending_in_m']:.2%}

Paired outcomes
---------------
V1 correct -> V3 wrong: {paired['v1_correct_to_v3_wrong']}
V1 wrong -> V3 correct: {paired['v1_wrong_to_v3_correct']}
Net recovery vs V1: {paired['net_recovery_vs_v1']:+d}
WINO wrong -> V3 correct: {paired['wino_wrong_to_v3_correct']}
WINO correct -> V3 wrong: {paired['wino_correct_to_v3_wrong']}
Net recovery vs WINO: {paired['net_recovery_vs_wino']:+d}

References
----------
WINO: accuracy 77.56%, average NFE 46.25.
Soft Revision V1/V2: accuracy 78.32%, average NFE 42.93.
"""
        with open(os.path.join(output_dir, "report_soft_revision_v3.txt"), "w",
                  encoding="utf-8") as handle:
            handle.write(report)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--num-samples", type=int, default=1319)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.output_dir, ".gsm8k_v3_samples.jsonl")
    dataset = load_dataset(DATA_PATH, "main", trust_remote_code=True)["test"]
    total = min(args.num_samples, len(dataset))
    completed = {row["index"] for row in read_jsonl(checkpoint_path)}
    print(f"Soft Revision V3: resuming {len(completed)}/{total}", flush=True)

    model = LLaDAModelLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16).cuda().eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    context, _ = gsm8k_doc_to_text(dataset[0])
    prompt = tokenizer.apply_chat_template(
        context, add_generation_prompt=True, tokenize=False) + "<reasoning>"
    warm_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
    decoding_wino_soft_revision_v3(
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
            output, steps, diagnostics = decoding_wino_soft_revision_v3(
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
                event["original_hard_token_string"] = tokenizer.convert_ids_to_tokens(
                    event["original_hard_token_id"])
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
            print(f"V3 {index + 1}/{total} nfe={steps}", flush=True)
    finalize(checkpoint_path, args.output_dir, total)


if __name__ == "__main__":
    main()
