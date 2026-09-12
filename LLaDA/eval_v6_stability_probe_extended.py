"""Diagnostic-only V6 stability probe on paired flips of existing benchmarks."""

import argparse
import json
import os
import time

import torch
from transformers import AutoTokenizer

from decoding import decoding_wino_soft_revision_v6
from eval_soft_revision_extended import DATASETS, format_example, load_benchmark
from modeling_llada import LLaDAModelLM


HERE = os.path.dirname(__file__)
MODEL_PATH = "/root/lx/ReMix-DLLM/LLaDA/models/LLaDA-8B-Instruct"


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
    parser.add_argument("--dataset", required=True,
                        choices=("math500", "humaneval", "mbpp"))
    parser.add_argument("--output-dir", default=os.path.join(
        HERE, "results", "soft_revision_v6", "stability_cross_benchmark"))
    args = parser.parse_args()

    v1 = load(os.path.join(
        HERE, "results", "soft_revision", f"{args.dataset}_soft_revision.json"))
    v6 = load(os.path.join(
        HERE, "results", "soft_revision_v6", f"{args.dataset}_soft_revision_v6.json"))
    v1_rows = {row["index"]: row for row in v1["samples"]}
    v6_rows = {row["index"]: row for row in v6["samples"]}
    selected = sorted(index for index in v6_rows
                      if bool(v1_rows[index]["is_correct"])
                      != bool(v6_rows[index]["is_correct"]))
    os.makedirs(args.output_dir, exist_ok=True)
    output = os.path.join(args.output_dir, f"{args.dataset}_probe.jsonl")
    completed = {row["index"] for row in read_jsonl(output)}
    benchmark = load_benchmark(args.dataset)

    model = LLaDAModelLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16).cuda().eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    context, _, trailing = format_example(args.dataset, benchmark[selected[0]])
    prompt = tokenizer.apply_chat_template(
        context, add_generation_prompt=True, tokenize=False) + trailing
    if args.dataset == "math500":
        prompt += "<reasoning>"
    warm_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
    decoding_wino_soft_revision_v6(
        model, warm_ids, gen_length=256, block_length=128, temperature=0.0,
        threshold=DATASETS[args.dataset]["threshold"], threshold_back=0.9,
        beta_mix=0.5)

    with open(output, "a", encoding="utf-8", buffering=1) as checkpoint:
        for ordinal, index in enumerate(selected, 1):
            if index in completed:
                continue
            context, _, trailing = format_example(args.dataset, benchmark[index])
            prompt = tokenizer.apply_chat_template(
                context, add_generation_prompt=True, tokenize=False) + trailing
            if args.dataset == "math500":
                prompt += "<reasoning>"
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
            torch.cuda.synchronize()
            started = time.perf_counter()
            generated, steps, diagnostics = decoding_wino_soft_revision_v6(
                model, input_ids, gen_length=256, block_length=128,
                temperature=0.0, threshold=DATASETS[args.dataset]["threshold"],
                threshold_back=0.9, beta_mix=0.5, return_diagnostics=True,
                stability_probe=True)
            torch.cuda.synchronize()
            response = tokenizer.batch_decode(
                generated[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
            expected = v6_rows[index]
            excluded = False
            if response != expected["full_response"] or int(steps) != expected["nfe"]:
                control_generated, control_steps, _ = decoding_wino_soft_revision_v6(
                    model, input_ids, gen_length=256, block_length=128,
                    temperature=0.0, threshold=DATASETS[args.dataset]["threshold"],
                    threshold_back=0.9, beta_mix=0.5, return_diagnostics=True,
                    stability_probe=False)
                control_response = tokenizer.batch_decode(
                    control_generated[:, input_ids.shape[1]:],
                    skip_special_tokens=True)[0]
                if control_response != response or int(control_steps) != int(steps):
                    raise RuntimeError(
                        f"Probe itself changed output/NFE for {args.dataset} {index}")
                excluded = True
                print(f"excluding {args.dataset} sample {index}: probe on/off agree, "
                      "but the saved V6 trajectory is not reproduced", flush=True)
            diagnostics.pop("round_trace", None)
            checkpoint.write(json.dumps({
                "index": index,
                "group": "recovered" if expected["is_correct"] else "regressed",
                "is_correct": bool(expected["is_correct"]),
                "excluded_saved_trajectory_mismatch": excluded,
                "nfe": int(steps),
                "latency": time.perf_counter() - started,
                "diagnostics": diagnostics,
            }, ensure_ascii=False) + "\n")
            print(f"{args.dataset} probe {ordinal}/{len(selected)} sample={index} "
                  f"group={'recovered' if expected['is_correct'] else 'regressed'}", flush=True)

    if {row["index"] for row in read_jsonl(output)} != set(selected):
        raise RuntimeError(f"Incomplete {args.dataset} probe")
    print(f"complete: {args.dataset}, {len(selected)} paired flips", flush=True)


if __name__ == "__main__":
    main()
