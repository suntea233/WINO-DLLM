"""Evaluate one-round identity-correction-only Soft Revision V6 on GSM8K."""

import argparse
import json
import os
import time

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from dataset_utils.gsm8k import gsm8k_doc_to_text, gsm8k_extract_answer, gsm8k_is_correct
from decoding import decoding_wino_soft_revision_v6
from modeling_llada import LLaDAModelLM


MODEL_PATH = "/root/lx/ReMix-DLLM/LLaDA/models/LLaDA-8B-Instruct"
DATA_PATH = "/root/lx/ReMix-DLLM/LLaDA/data/gsm8k"
RESULT_ROOT = os.path.join(os.path.dirname(__file__), "results")
OUTPUT_DIR = os.path.join(RESULT_ROOT, "soft_revision_v6")


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


def identity_stats(rows_by_id, sample_ids):
    counts = [
        sum(event.get("identity_changed", False)
            for event in rows_by_id[index]["diagnostics"]["revision_events"])
        for index in sample_ids]
    return {
        "samples": len(sample_ids),
        "samples_with_identity_correction": sum(count > 0 for count in counts),
        "fraction_samples_with_identity_correction": (
            sum(count > 0 for count in counts) / len(counts) if counts else 0.0),
        "identity_correction_events": sum(counts),
        "identity_correction_events_per_sample": (
            sum(counts) / len(counts) if counts else 0.0),
    }


def paired(reference, rows_by_id):
    reference_by_id = {row["index"]: row for row in reference["samples"]}
    recovered_ids = [
        index for index, row in rows_by_id.items()
        if not reference_by_id[index]["is_correct"] and row["is_correct"]]
    regressed_ids = [
        index for index, row in rows_by_id.items()
        if reference_by_id[index]["is_correct"] and not row["is_correct"]]
    return {
        "recovered": len(recovered_ids),
        "regressed": len(regressed_ids),
        "net_recovery": len(recovered_ids) - len(regressed_ids),
        "recovered_identity_statistics": identity_stats(rows_by_id, recovered_ids),
        "regressed_identity_statistics": identity_stats(rows_by_id, regressed_ids),
        "recovered_sample_ids": recovered_ids,
        "regressed_sample_ids": regressed_ids,
    }


def finalize(checkpoint_path, output_dir, total):
    rows = sorted(read_jsonl(checkpoint_path), key=lambda row: row["index"])
    if len(rows) != total or [row["index"] for row in rows] != list(range(total)):
        raise RuntimeError(f"Incomplete V6 checkpoint: {len(rows)}/{total}")
    if not all(row["nfe"] == row["steps"] for row in rows):
        raise RuntimeError("V6 NFE must equal the number of normal decoding rounds")
    if not all(row["diagnostics"]["unresolved_soft_states"] == 0 for row in rows):
        raise RuntimeError("V6 terminated with unresolved SOFT states")

    events = [
        event for row in rows for event in row["diagnostics"]["revision_events"]]
    if not all(event["soft_lifetime"] == 1 for event in events):
        raise RuntimeError("V6 contains a SOFT episode whose lifetime is not one round")
    high_confidence = [
        event for event in events
        if event["commit_reason"] == "p_old>=threshold_back"]
    corrections = [
        event for event in events
        if event["commit_reason"] == "different_identity_correction"]
    masks = [event for event in events if event["final_transition"] == "S->M"]
    if not all(event["identity_changed"] for event in corrections):
        raise RuntimeError("A V6 identity-correction event kept the old token")
    if any(event["final_transition"] == "S->H" and not (
            event["commit_reason"] == "p_old>=threshold_back"
            or event["identity_changed"]) for event in events):
        raise RuntimeError("V6 performed an illegal same-identity early recommit")

    harden_count = len(high_confidence) + len(corrections)
    correct = sum(row["is_correct"] for row in rows)
    metrics = {
        "accuracy": correct / total,
        "correct": correct,
        "total": total,
        "average_nfe": sum(row["nfe"] for row in rows) / total,
        "average_latency": sum(row["latency"] for row in rows) / total,
        "total_h_to_s": len(events),
        "s_to_h_a_via_p_old_ge_0_9": len(high_confidence),
        "s_to_h_b_identity_correction": len(corrections),
        "s_to_mask": len(masks),
        "identity_change_rate_among_hardening": (
            len(corrections) / harden_count if harden_count else 0.0),
        "identity_change_rate_among_all_soft": (
            len(corrections) / len(events) if events else 0.0),
        "soft_embeddings_consumed": sum(
            row["diagnostics"]["soft_embeddings_consumed"] for row in rows),
    }
    result = {
        "dataset": "gsm8k_test",
        "method": "soft_revision_v6_identity_correction_only",
        "configuration": {
            "model": MODEL_PATH,
            "gen_length": 256,
            "block_length": 128,
            "temperature": 0.0,
            "threshold": 0.6,
            "threshold_back": 0.9,
            "beta_mix": 0.5,
            "soft_lifetime_rounds": 1,
            "arbitration": (
                "p_old>=0.9: S->H(a); else b!=a and p_top1>=0.6: "
                "S->H(b); else S->M"),
        },
        "metrics": metrics,
        "samples": rows,
    }

    if total == 1319:
        reference_paths = {
            "wino": os.path.join(
                RESULT_ROOT, "soft_revision", "gsm8k_wino_remask.json"),
            "v1": os.path.join(
                RESULT_ROOT, "soft_revision", "gsm8k_soft_revision.json"),
            "v4": os.path.join(
                RESULT_ROOT, "soft_revision_v4", "gsm8k_soft_revision_v4.json"),
            "v5": os.path.join(
                RESULT_ROOT, "soft_revision_v5", "gsm8k_soft_revision_v5.json"),
        }
        rows_by_id = {row["index"]: row for row in rows}
        result["paired"] = {}
        for name, path in reference_paths.items():
            with open(path, encoding="utf-8") as handle:
                result["paired"][name] = paired(json.load(handle), rows_by_id)

    os.makedirs(output_dir, exist_ok=True)
    atomic_json(os.path.join(output_dir, "gsm8k_soft_revision_v6.json"), result)
    summary = {
        "configuration": result["configuration"],
        "metrics": metrics,
        "paired": result.get("paired"),
    }
    atomic_json(os.path.join(output_dir, "summary_v6.json"), summary)

    trace_path = os.path.join(output_dir, "revision_trace_v6.jsonl")
    with open(trace_path + ".tmp", "w", encoding="utf-8") as handle:
        for row in rows:
            for event in row["diagnostics"]["revision_events"]:
                handle.write(json.dumps(
                    {"sample_index": row["index"], **event}, ensure_ascii=False) + "\n")
    os.replace(trace_path + ".tmp", trace_path)

    if total == 1319:
        p_v1 = result["paired"]["v1"]
        p_v4 = result["paired"]["v4"]
        p_v5 = result["paired"]["v5"]
        rec = p_v1["recovered_identity_statistics"]
        reg = p_v1["regressed_identity_statistics"]
        report = f"""Soft Revision V6: GSM8K
=========================

Frozen setting: LLaDA-8B-Instruct, gen_length=256, block_length=128,
temperature=0, threshold=0.6, threshold_back=0.9, beta_mix=0.5.
SOFT lifetime is exactly one normal forward and no auxiliary forward is used.

Accuracy: {metrics['accuracy']:.4%} ({correct}/{total})
Average NFE: {metrics['average_nfe']:.2f}
Average latency: {metrics['average_latency']:.3f}s

Total H->S: {metrics['total_h_to_s']}
S->H(a) via p_old>=0.9: {len(high_confidence)}
S->H(b), b!=a identity corrections: {len(corrections)}
S->MASK: {len(masks)}
Identity-change rate among S->H: {metrics['identity_change_rate_among_hardening']:.2%}
Identity-change rate among all H->S: {metrics['identity_change_rate_among_all_soft']:.2%}
Same-identity early recommits: 0

Paired against V1
-----------------
V1 wrong -> V6 correct: {p_v1['recovered']}
V1 correct -> V6 wrong: {p_v1['regressed']}
Net recovery: {p_v1['net_recovery']:+d}
Identity corrections in recovered samples: {rec['identity_correction_events']} ({rec['identity_correction_events_per_sample']:.2f}/sample)
Identity corrections in regressed samples: {reg['identity_correction_events']} ({reg['identity_correction_events_per_sample']:.2f}/sample)
Samples with corrections, recovered/regressed: {rec['fraction_samples_with_identity_correction']:.2%} / {reg['fraction_samples_with_identity_correction']:.2%}

Paired against V4
-----------------
V4 wrong -> V6 correct: {p_v4['recovered']}
V4 correct -> V6 wrong: {p_v4['regressed']}
Net recovery: {p_v4['net_recovery']:+d}

Paired against V5
-----------------
V5 wrong -> V6 correct: {p_v5['recovered']}
V5 correct -> V6 wrong: {p_v5['regressed']}
Net recovery: {p_v5['net_recovery']:+d}

References
----------
WINO: accuracy 77.56%, average NFE 46.25.
Soft Revision V1: accuracy 78.32%, average NFE 42.93.
Soft Revision V4: accuracy 78.09%, average NFE 40.00.
Soft Revision V5: accuracy 77.79%, average NFE 40.71.
"""
        with open(os.path.join(output_dir, "report_soft_revision_v6.txt"), "w",
                  encoding="utf-8") as handle:
            handle.write(report)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--num-samples", type=int, default=1319)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.output_dir, ".gsm8k_v6_samples.jsonl")
    dataset = load_dataset(DATA_PATH, "main", trust_remote_code=True)["test"]
    total = min(args.num_samples, len(dataset))
    completed = {row["index"] for row in read_jsonl(checkpoint_path)}
    print(f"Soft Revision V6: resuming {len(completed)}/{total}", flush=True)

    model = LLaDAModelLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16).cuda().eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    context, _ = gsm8k_doc_to_text(dataset[0])
    prompt = tokenizer.apply_chat_template(
        context, add_generation_prompt=True, tokenize=False) + "<reasoning>"
    warm_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
    decoding_wino_soft_revision_v6(
        model, warm_ids, gen_length=256, block_length=128, temperature=0.0,
        threshold=0.6, threshold_back=0.9, beta_mix=0.5)

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
            output, steps, diagnostics = decoding_wino_soft_revision_v6(
                model, input_ids, gen_length=256, block_length=128,
                temperature=0.0, threshold=0.6, threshold_back=0.9,
                beta_mix=0.5, return_diagnostics=True)
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
                event["current_top1_token_string"] = tokenizer.convert_ids_to_tokens(
                    event["current_top1_token_id"])
                if "soft_top1_token_id" in event:
                    event["soft_top1_token_string"] = tokenizer.convert_ids_to_tokens(
                        event["soft_top1_token_id"])
                    event["shadow_top1_token_string"] = tokenizer.convert_ids_to_tokens(
                        event["shadow_top1_token_id"])
                event["new_hard_token_string"] = (
                    tokenizer.convert_ids_to_tokens(event["new_token_id"])
                    if event["new_token_id"] is not None else None)
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
            print(f"V6 {index + 1}/{total} nfe={steps}", flush=True)
    finalize(checkpoint_path, args.output_dir, total)


if __name__ == "__main__":
    main()
