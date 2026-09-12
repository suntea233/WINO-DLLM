"""Trace token identity across WINO H->MASK->H cycles on GSM8K."""

import argparse
from collections import Counter
import json
import os
import statistics
import time

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from dataset_utils.gsm8k import gsm8k_doc_to_text, gsm8k_extract_answer, gsm8k_is_correct
from decoding import decoding_wino_remask
from modeling_llada import LLaDAModelLM


MODEL_PATH = "/root/lx/ReMix-DLLM/LLaDA/models/LLaDA-8B-Instruct"
DATA_PATH = "/root/lx/ReMix-DLLM/LLaDA/data/gsm8k"
RESULT_DIR = os.path.join(os.path.dirname(__file__), "results", "soft_revision")
BASELINE_PATH = os.path.join(RESULT_DIR, "gsm8k_wino_remask.json")


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


def summarize_events(events):
    same = [event for event in events if event["same_token"]]
    changed = [event for event in events if not event["same_token"]]
    total = len(events)
    pairs = Counter(
        (event["original_token_string"], event["new_token_string"])
        for event in changed)
    return {
        "total_h_to_m_to_h_events": total,
        "same_token_count": len(same),
        "same_token_percentage": 100.0 * len(same) / total if total else 0.0,
        "changed_token_count": len(changed),
        "changed_token_percentage": 100.0 * len(changed) / total if total else 0.0,
        "average_remask_duration_rounds": (
            statistics.fmean(event["rounds_spent_remasked"] for event in events)
            if events else 0.0),
        "most_common_changed_token_pairs": [
            {"old_token": old, "new_token": new, "count": count}
            for (old, new), count in pairs.most_common(10)
        ],
    }


def finalize(checkpoint_path):
    rows = sorted(read_jsonl(checkpoint_path), key=lambda row: row["sample_id"])
    if len(rows) != 1319 or [row["sample_id"] for row in rows] != list(range(1319)):
        raise RuntimeError(f"Incomplete identity trace: {len(rows)}/1319")
    if not all(row["output_matches_existing"] and row["steps_match_existing"]
               and row["correctness_matches_existing"] for row in rows):
        raise RuntimeError("Tracing changed at least one existing WINO result")
    events = [event for row in rows for event in row["events"]]
    if len(events) != 62702:
        raise RuntimeError(f"Expected 62702 remask cycles, found {len(events)}")
    correct_events = [event for row in rows if row["is_correct"] for event in row["events"]]
    wrong_events = [event for row in rows if not row["is_correct"] for event in row["events"]]
    result = {
        "configuration": {
            "dataset": "gsm8k_test",
            "model": MODEL_PATH,
            "gen_length": 256,
            "block_length": 128,
            "temperature": 0.0,
            "threshold": 0.6,
            "threshold_back": 0.9,
            "accuracy": 1023 / 1319,
            "correct": 1023,
            "total": 1319,
            "average_nfe": 46.25094768764215,
        },
        "output_regression_check": {
            "matched_samples": 1319,
            "response_matches": sum(row["output_matches_existing"] for row in rows),
            "decoding_round_matches": sum(row["steps_match_existing"] for row in rows),
            "correctness_matches": sum(row["correctness_matches_existing"] for row in rows),
        },
        "all_samples": summarize_events(events),
        "wino_correct_samples": {
            "sample_count": sum(row["is_correct"] for row in rows),
            **summarize_events(correct_events),
        },
        "wino_wrong_samples": {
            "sample_count": sum(not row["is_correct"] for row in rows),
            **summarize_events(wrong_events),
        },
        "soft_revision_comparison": {
            "h_to_s_to_h_same_token": 12073,
            "h_to_s_to_h_changed_token": 0,
        },
        "samples": rows,
    }
    output_path = os.path.join(RESULT_DIR, "wino_remask_identity_analysis.json")
    atomic_json(output_path, result)

    all_stats = result["all_samples"]
    correct_stats = result["wino_correct_samples"]
    wrong_stats = result["wino_wrong_samples"]
    lines = [
        "WINO remask token-identity analysis",
        "===================================",
        "",
        "Tracing regression check",
        "------------------------",
        "All 1319 traced runs exactly matched the saved WINO response, decoding-round",
        "count, and correctness. Accuracy remained 1023/1319 (77.56%) and average",
        "NFE remained 46.25. Tracing added bookkeeping only and no backbone forward.",
        "",
        "All samples",
        "-----------",
        f"Total H->M->H events: {all_stats['total_h_to_m_to_h_events']}",
        f"Same token H(a)->M->H(a): {all_stats['same_token_count']} ({all_stats['same_token_percentage']:.2f}%)",
        f"Changed token H(a)->M->H(b): {all_stats['changed_token_count']} ({all_stats['changed_token_percentage']:.2f}%)",
        f"Average remask duration: {all_stats['average_remask_duration_rounds']:.2f} rounds",
        "",
        "WINO-correct samples",
        "--------------------",
        f"Samples: {correct_stats['sample_count']}",
        f"Total events: {correct_stats['total_h_to_m_to_h_events']}",
        f"Same token: {correct_stats['same_token_count']} ({correct_stats['same_token_percentage']:.2f}%)",
        f"Changed token: {correct_stats['changed_token_count']} ({correct_stats['changed_token_percentage']:.2f}%)",
        f"Average duration: {correct_stats['average_remask_duration_rounds']:.2f} rounds",
        "",
        "WINO-wrong samples",
        "------------------",
        f"Samples: {wrong_stats['sample_count']}",
        f"Total events: {wrong_stats['total_h_to_m_to_h_events']}",
        f"Same token: {wrong_stats['same_token_count']} ({wrong_stats['same_token_percentage']:.2f}%)",
        f"Changed token: {wrong_stats['changed_token_count']} ({wrong_stats['changed_token_percentage']:.2f}%)",
        f"Average duration: {wrong_stats['average_remask_duration_rounds']:.2f} rounds",
        "",
        "Most common changed-token pairs",
        "-------------------------------",
    ]
    for pair in all_stats["most_common_changed_token_pairs"]:
        lines.append(f"{pair['old_token']!r} -> {pair['new_token']!r}: {pair['count']}")
    lines.extend([
        "",
        "Comparison with Soft Revision",
        "-----------------------------",
        "Soft Revision H->S->H keeps the original committed ID by construction:",
        "same token = 12073, changed token = 0. WINO H->M->H instead permits token",
        "identity to change; the measured frequency above quantifies how often it does.",
    ])
    with open(os.path.join(RESULT_DIR, "report_wino_remask_identity.txt"),
              "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print(json.dumps(result["all_samples"], ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=1319)
    parser.add_argument("--worker-name", default="full")
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    os.makedirs(RESULT_DIR, exist_ok=True)
    checkpoint_path = os.path.join(
        RESULT_DIR, f".wino_remask_identity_{args.worker_name}.jsonl")
    if args.finalize:
        finalize(checkpoint_path)
        return

    baseline = json.load(open(BASELINE_PATH, encoding="utf-8"))
    baseline_by_id = {row["index"]: row for row in baseline["samples"]}
    dataset = load_dataset(DATA_PATH, "main", trust_remote_code=True)["test"]
    expected = list(range(args.start_index, args.end_index))
    existing = {row["sample_id"] for row in read_jsonl(checkpoint_path)}

    model = LLaDAModelLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16).cuda().eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    context, _ = gsm8k_doc_to_text(dataset[expected[0]])
    prompt = tokenizer.apply_chat_template(
        context, add_generation_prompt=True, tokenize=False) + "<reasoning>"
    warm_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
    decoding_wino_remask(
        model, warm_ids, gen_length=256, block_length=128, temperature=0.0,
        threshold=0.6, threshold_back=0.9)

    with open(checkpoint_path, "a", encoding="utf-8", buffering=1) as handle:
        for sample_id in expected:
            if sample_id in existing:
                continue
            context, target = gsm8k_doc_to_text(dataset[sample_id])
            prompt = tokenizer.apply_chat_template(
                context, add_generation_prompt=True, tokenize=False) + "<reasoning>"
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
            torch.cuda.synchronize()
            started = time.perf_counter()
            output, steps, diagnostics = decoding_wino_remask(
                model, input_ids, gen_length=256, block_length=128,
                temperature=0.0, threshold=0.6, threshold_back=0.9,
                return_diagnostics=True)
            torch.cuda.synchronize()
            response = tokenizer.batch_decode(
                output[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
            prediction = gsm8k_extract_answer(response)
            is_correct = bool(gsm8k_is_correct(prediction, target))
            old = baseline_by_id[sample_id]
            events = diagnostics["remask_identity_events"]
            for event in events:
                event["sample_id"] = sample_id
                event["absolute_position"] = event.pop("position")
                event["generation_position"] = (
                    event["absolute_position"] - input_ids.shape[1])
                event["original_token_string"] = tokenizer.convert_ids_to_tokens(
                    event["original_token_id"])
                event["new_token_string"] = tokenizer.convert_ids_to_tokens(
                    event["new_token_id"])
            row = {
                "sample_id": sample_id,
                "is_correct": is_correct,
                "steps": int(steps),
                "nfe": diagnostics["nfe"],
                "latency": time.perf_counter() - started,
                "output_matches_existing": response == old["full_response"],
                "steps_match_existing": int(steps) == old["steps"],
                "correctness_matches_existing": is_correct == old["is_correct"],
                "events": events,
            }
            if not (row["output_matches_existing"] and row["steps_match_existing"]
                    and row["correctness_matches_existing"]):
                raise RuntimeError(f"Trace changed WINO output at sample {sample_id}")
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(
                f"identity {sample_id + 1}/{args.end_index} events={len(events)} "
                f"nfe={steps}", flush=True)


if __name__ == "__main__":
    main()
