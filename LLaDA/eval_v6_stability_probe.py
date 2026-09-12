"""Run a diagnostic-only V6 stability probe on paired GSM8K samples."""

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


HERE = os.path.dirname(__file__)
MODEL_PATH = "/root/lx/ReMix-DLLM/LLaDA/models/LLaDA-8B-Instruct"
DATA_PATH = "/root/lx/ReMix-DLLM/LLaDA/data/gsm8k"
V6_PATH = os.path.join(HERE, "results", "soft_revision_v6", "gsm8k_soft_revision_v6.json")
CONTROL_PATH = os.path.join(
    HERE, "results", "soft_revision_v6_control_v1", "gsm8k_soft_revision.json")
DEFAULT_OUTPUT = os.path.join(
    HERE, "results", "soft_revision_v6", "stability_probe_samples.jsonl")


def load(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args()

    original = load(V6_PATH)
    control = load(CONTROL_PATH)
    original_rows = {row["index"]: row for row in original["samples"]}
    control_rows = {row["index"]: row for row in control["samples"]}
    selected = sorted(
        index for index in original_rows
        if bool(original_rows[index]["is_correct"])
        != bool(control_rows[index]["is_correct"]))
    if args.max_samples is not None:
        selected = selected[:args.max_samples]
    completed = {row["index"] for row in read_jsonl(args.output)}
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    dataset = load_dataset(DATA_PATH, "main", trust_remote_code=True)["test"]
    model = LLaDAModelLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16).cuda().eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    context, _ = gsm8k_doc_to_text(dataset[selected[0]])
    prompt = tokenizer.apply_chat_template(
        context, add_generation_prompt=True, tokenize=False) + "<reasoning>"
    warm_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
    decoding_wino_soft_revision_v6(
        model, warm_ids, gen_length=256, block_length=128, temperature=0.0,
        threshold=0.6, threshold_back=0.9, beta_mix=0.5)

    with open(args.output, "a", encoding="utf-8", buffering=1) as checkpoint:
        for ordinal, index in enumerate(selected, 1):
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
                beta_mix=0.5, return_diagnostics=True, stability_probe=True)
            torch.cuda.synchronize()
            latency = time.perf_counter() - started
            response = tokenizer.batch_decode(
                output[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
            prediction = gsm8k_extract_answer(response)
            is_correct = bool(gsm8k_is_correct(prediction, target))
            expected = original_rows[index]
            if response != expected["full_response"]:
                raise RuntimeError(f"Probe changed output for sample {index}")
            if int(steps) != expected["nfe"] or is_correct != expected["is_correct"]:
                raise RuntimeError(f"Probe changed NFE/correctness for sample {index}")
            diagnostics.pop("round_trace", None)
            checkpoint.write(json.dumps({
                "index": index,
                "group": "recovered" if is_correct else "regressed",
                "full_response": response,
                "prediction": prediction,
                "is_correct": is_correct,
                "nfe": int(steps),
                "latency": latency,
                "diagnostics": diagnostics,
            }, ensure_ascii=False) + "\n")
            print(f"probe {ordinal}/{len(selected)} sample={index} group="
                  f"{'recovered' if is_correct else 'regressed'} nfe={steps}", flush=True)

    rows = read_jsonl(args.output)
    if {row["index"] for row in rows} != set(selected):
        raise RuntimeError("Stability probe is incomplete")
    print(f"complete: {len(rows)} paired samples; exact outputs and NFE verified", flush=True)


if __name__ == "__main__":
    main()
