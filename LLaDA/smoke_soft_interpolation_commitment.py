#!/usr/bin/env python3
"""Deterministic alpha=1 parity and interpolation lifecycle smoke test."""

import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from dataset_utils.gsm8k import gsm8k_doc_to_text
from decoding import (
    decoding_wino_soft_interpolation_commitment,
    decoding_wino_soft_revision_v6,
)
from modeling_llada import LLaDAModelLM


MODEL = "/root/lx/ReMix-DLLM/LLaDA/models/LLaDA-8B-Instruct"
DATA = "/root/lx/ReMix-DLLM/LLaDA/data/gsm8k"


def transfer_trace(diag):
    return [row["transferred_positions"] for row in diag["round_trace"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    data = load_dataset(DATA, "main", trust_remote_code=True)["test"]
    model = LLaDAModelLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16).cuda().eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    rows = []
    for index in range(min(args.samples, len(data))):
        context, _ = gsm8k_doc_to_text(data[index])
        prompt = tokenizer.apply_chat_template(
            context, add_generation_prompt=True, tokenize=False) + "<reasoning>"
        ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
        common = dict(gen_length=256, block_length=128, temperature=0.0,
                      threshold=0.6, threshold_back=0.9, beta_mix=0.5,
                      return_diagnostics=True)
        v6, v6_steps, v6_d = decoding_wino_soft_revision_v6(model, ids, **common)
        control, control_steps, control_d = decoding_wino_soft_interpolation_commitment(
            model, ids, commit_alpha=1.0, **common)
        if not torch.equal(v6, control) or v6_steps != control_steps:
            raise RuntimeError(f"alpha=1 differs from V6 at sample {index}")
        if transfer_trace(v6_d) != transfer_trace(control_d):
            raise RuntimeError(f"alpha=1 transfer trace differs at sample {index}")
        alpha_rows = {}
        for alpha in (0.25, 0.5, 0.75):
            output, steps, diag = decoding_wino_soft_interpolation_commitment(
                model, ids, commit_alpha=alpha, **common)
            if steps != diag["nfe"]:
                raise RuntimeError(f"alpha={alpha} NFE mismatch at sample {index}")
            if diag["unresolved_soft_states"]:
                raise RuntimeError(f"alpha={alpha} unresolved Soft at sample {index}")
            commits = [e for e in diag["revision_events"]
                       if e.get("commit_reason") == "different_identity_correction"]
            if len(commits) != diag["num_interpolated_identity_commits"]:
                raise RuntimeError(f"alpha={alpha} commit counter mismatch")
            for event in commits:
                if event["commit_alpha"] != alpha:
                    raise RuntimeError("Wrong alpha in event trace")
                if event["final_committed_representation_type"] != (
                        "one_round_hard_soft_interpolation"):
                    raise RuntimeError("Wrong commit representation trace")
            alpha_rows[str(alpha)] = {
                "steps": steps,
                "output_equal_to_v6": bool(torch.equal(output, v6)),
                "interpolated_identity_commits": len(commits),
                "commit_forwards_consumed": diag[
                    "num_interpolated_commit_forwards_consumed"],
            }
        row = {"index": index, "v6_steps": v6_steps,
               "alpha_1_exact_v6": True, "alphas": alpha_rows}
        rows.append(row)
        print(json.dumps(row), flush=True)
    summary = {
        "samples": len(rows), "alpha_1_exact_v6": True,
        "alpha_1_transfer_trace_exact_v6": True,
        "nfe_accounting_valid": True, "unresolved_soft_states": 0,
        "interpolated_commits": {
            alpha: sum(r["alphas"][alpha]["interpolated_identity_commits"]
                       for r in rows) for alpha in ("0.25", "0.5", "0.75")},
        "samples_with_output_change": {
            alpha: sum(not r["alphas"][alpha]["output_equal_to_v6"]
                       for r in rows) for alpha in ("0.25", "0.5", "0.75")},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary": summary, "samples": rows}, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
