"""Deterministic one-decision V6 candidate oracle; no policy changes after the override."""

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from decoding import decoding_wino_soft_revision_v6
from modeling_llada import LLaDAModelLM
from dataset_utils.gsm8k import gsm8k_doc_to_text, gsm8k_extract_answer, gsm8k_is_correct
from dataset_utils.math500 import math500_extract_answer, math500_is_equiv
from dataset_utils.humaneval import humaneval_extract_answer
from dataset_utils.mbpp import mbpp_extract_answer
from eval_soft_revision_extended import (
    DATASETS, load_benchmark, format_example, evaluate_humaneval,
    evaluate_mbpp_rows,
)

HERE = Path(__file__).resolve().parent
RESULTS = Path('/root/lx/WINO-DLLM/LLaDA/results/v6_revision_candidate_oracle')
MODEL = '/root/lx/ReMix-DLLM/LLaDA/models/LLaDA-8B-Instruct'
BRANCHES = ('B1', 'B2', 'B3', 'B4', 'OLD')
V6_ROOT = Path('/root/lx/WINO-DLLM/LLaDA/results/soft_revision_v6')
V1_ROOT = Path('/root/lx/WINO-DLLM/LLaDA/results/soft_revision')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_jsonl(path):
    path = Path(path)
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line] if path.exists() else []


def append_jsonl(path, row):
    with open(path, 'a', encoding='utf-8', buffering=1) as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + '\n')


def paths(dataset):
    v6 = V6_ROOT / f'{dataset}_soft_revision_v6.json'
    v1 = (Path('/root/lx/WINO-DLLM/LLaDA/results/soft_revision_v6_control_v1/gsm8k_soft_revision.json')
          if dataset == 'gsm8k' else V1_ROOT / f'{dataset}_soft_revision.json')
    return v1, v6


def prepare(include_both_wrong=True):
    RESULTS.mkdir(parents=True, exist_ok=True)
    all_events = []
    manifest = {}
    for dataset in ('gsm8k', 'math500', 'mbpp', 'humaneval'):
        v1_path, v6_path = paths(dataset)
        v1, v6 = read_json(v1_path), read_json(v6_path)
        for key in ('model', 'gen_length', 'block_length', 'temperature',
                    'threshold', 'threshold_back', 'beta_mix'):
            assert v1['configuration'][key] == v6['configuration'][key], (dataset, key)
        old = {int(row['index']): row for row in v1['samples']}
        new = {int(row['index']): row for row in v6['samples']}
        assert old.keys() == new.keys()
        groups = {'recovered': [], 'regressed': [], 'both_wrong': [], 'both_correct': []}
        for index in sorted(new):
            a, b = bool(old[index]['is_correct']), bool(new[index]['is_correct'])
            group = ('both_correct' if a and b else 'recovered' if not a and b
                     else 'regressed' if a and not b else 'both_wrong')
            groups[group].append(index)
        selected = {key: groups[key] for key in ('recovered', 'regressed')}
        if include_both_wrong:
            selected['both_wrong'] = [index for index in groups['both_wrong']
                if any(event.get('commit_reason') == 'different_identity_correction'
                       for event in new[index]['diagnostics']['revision_events'])][:20]
        for group, indices in selected.items():
            for index in indices:
                events = [event for event in new[index]['diagnostics']['revision_events']
                          if event.get('commit_reason') == 'different_identity_correction']
                if not events:
                    continue
                event = min(events, key=lambda e: (e['resolved_round'], e['event_id']))
                all_events.append({
                    'dataset': dataset, 'sample_id': index, 'group': group,
                    'event_id': event['event_id'], 'round': event['resolved_round'],
                    'block': event['block'], 'position': event['position'],
                    'old_token_id': event['original_hard_token_id'],
                    'historical_b1_token_id': event['new_token_id'],
                    'historical_v6_correct': bool(new[index]['is_correct']),
                    'historical_v6_response': new[index]['full_response'],
                    'historical_v6_nfe': int(new[index]['nfe']),
                })
        manifest[dataset] = {
            'v1_file': str(v1_path), 'v6_file': str(v6_path),
            'total': len(new), 'v6_correct': sum(bool(r['is_correct']) for r in new.values()),
            'groups': {key: len(value) for key, value in groups.items()},
            'selected_identity_events': sum(e['dataset'] == dataset for e in all_events),
        }
    (RESULTS / 'v6_revision_candidate_oracle_events.jsonl').write_text(
        ''.join(json.dumps(event, ensure_ascii=False) + '\n' for event in all_events),
        encoding='utf-8')
    (RESULTS / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest, indent=2))


def benchmark(dataset):
    if dataset == 'gsm8k':
        return load_dataset('/root/lx/ReMix-DLLM/LLaDA/data/gsm8k', 'main',
                            trust_remote_code=True)['test']
    return load_benchmark(dataset)


def prompt_target(dataset, doc, tokenizer):
    if dataset == 'gsm8k':
        context, target = gsm8k_doc_to_text(doc)
        trailing = '<reasoning>'
    else:
        context, target, trailing = format_example(dataset, doc)
        if dataset == 'math500':
            trailing += '<reasoning>'
    text = tokenizer.apply_chat_template(
        context, add_generation_prompt=True, tokenize=False) + trailing
    return tokenizer(text, return_tensors='pt').input_ids.cuda(), target


def extract(dataset, response, target):
    if dataset == 'gsm8k':
        prediction = gsm8k_extract_answer(response)
        return {'prediction': prediction,
                'is_correct': bool(gsm8k_is_correct(prediction, target))}
    if dataset == 'math500':
        prediction = math500_extract_answer(response)
        return {'prediction': prediction,
                'is_correct': bool(math500_is_equiv(prediction, target))}
    if dataset == 'humaneval':
        return {'task_id': target['task_id'],
                'completion': humaneval_extract_answer(response, target),
                'is_correct': None}
    return {'task_id': target['task_id'],
            'completion': mbpp_extract_answer('```python\n' + response, target['entry_point']),
            'is_correct': None}


def evaluate_humaneval_subset_rows(rows):
    """Official HumanEval tests, allowing the selected oracle subset to be incomplete."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as handle:
        sample_file = handle.name
        for row in rows:
            handle.write(json.dumps({'task_id': row['task_id'],
                                     'completion': row['completion']}) + '\n')
    try:
        metrics = evaluate_humaneval(sample_file, ignore_incomplete=True)
        passed = {row['task_id']: bool(row['passed'])
                  for row in read_jsonl(sample_file + '_results.jsonl')}
        return metrics, passed
    finally:
        for path in (sample_file, sample_file + '_results.jsonl'):
            if os.path.exists(path):
                os.unlink(path)


def run(dataset, max_samples=None):
    events = [e for e in read_jsonl(RESULTS / 'v6_revision_candidate_oracle_events.jsonl')
              if e['dataset'] == dataset]
    if max_samples is not None:
        events = events[:max_samples]
    if not events:
        print(f'{dataset}: no eligible identity-changing samples', flush=True)
        return
    device = os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')
    print(f'{dataset}: {len(events)} samples on CUDA_VISIBLE_DEVICES={device}', flush=True)
    model = LLaDAModelLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).cuda().eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    data = benchmark(dataset)
    checkpoint = RESULTS / f'.{dataset}_branch_results.jsonl'
    complete = {(row['sample_id'], row['branch']) for row in read_jsonl(checkpoint)}
    threshold = 0.6 if dataset == 'gsm8k' else DATASETS[dataset]['threshold']
    first = True
    for item in events:
        prompt, target = prompt_target(dataset, data[item['sample_id']], tokenizer)
        base = {key: item[key] for key in ('event_id', 'position', 'block', 'round')}
        for branch in BRANCHES:
            if (item['sample_id'], branch) in complete:
                continue
            override = dict(base, branch=branch)
            torch.cuda.synchronize()
            started = time.perf_counter()
            output, steps, diag = decoding_wino_soft_revision_v6(
                model, prompt, gen_length=256, block_length=128, temperature=0.,
                threshold=threshold, threshold_back=0.9, beta_mix=0.5,
                return_diagnostics=True, oracle_branch=override)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            selected = diag['oracle_event']
            assert selected is not None, (dataset, item['sample_id'], branch)
            assert selected['candidate_ids'][0] == item['historical_b1_token_id']
            if first and branch == 'B1':
                control, control_steps, _ = decoding_wino_soft_revision_v6(
                    model, prompt, gen_length=256, block_length=128, temperature=0.,
                    threshold=threshold, threshold_back=0.9, beta_mix=0.5,
                    return_diagnostics=True)
                assert torch.equal(output, control) and steps == control_steps, (
                    'Oracle B1 must match disabled V6 token-for-token', dataset, item['sample_id'])
                first = False
            response = tokenizer.batch_decode(
                output[:, prompt.shape[1]:], skip_special_tokens=True)[0]
            if branch == 'B1':
                assert response == item['historical_v6_response'], (
                    'Historical B1 output mismatch', dataset, item['sample_id'])
                assert steps == item['historical_v6_nfe'], (
                    'Historical B1 NFE mismatch', dataset, item['sample_id'])
            row = {
                'dataset': dataset, 'sample_id': item['sample_id'],
                'group': item['group'], 'branch': branch,
                'event': selected, 'final_output': response,
                'nfe': int(steps), 'latency': elapsed,
                **extract(dataset, response, target),
            }
            append_jsonl(checkpoint, row)
            print(f"{dataset} sample={item['sample_id']} {branch} "
                  f"nfe={steps} latency={elapsed:.2f}s", flush=True)


def finalize():
    manifest = read_json(RESULTS / 'manifest.json')
    selected_events = read_jsonl(RESULTS / 'v6_revision_candidate_oracle_events.jsonl')
    all_rows = []
    for dataset in manifest:
        rows = read_jsonl(RESULTS / f'.{dataset}_branch_results.jsonl')
        expected = [e for e in selected_events
                    if e['dataset'] == dataset]
        assert len(rows) == len(expected) * len(BRANCHES), (dataset, len(rows), len(expected))
        for branch in BRANCHES:
            branch_rows = [r for r in rows if r['branch'] == branch]
            if dataset == 'humaneval':
                _, passed = evaluate_humaneval_subset_rows(branch_rows)
                for row in branch_rows:
                    row['is_correct'] = passed[row['task_id']]
            elif dataset == 'mbpp':
                _, passed = evaluate_mbpp_rows(branch_rows)
                for row in branch_rows:
                    row['is_correct'] = passed[row['task_id']]
        all_rows.extend(rows)
    with open(RESULTS / 'v6_revision_candidate_branch_results.jsonl', 'w', encoding='utf-8') as handle:
        for row in all_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    summary = {}
    for dataset, info in manifest.items():
        dataset_rows = [r for r in all_rows if r['dataset'] == dataset]
        by_sample = {}
        for row in dataset_rows:
            by_sample.setdefault(row['sample_id'], {})[row['branch']] = row
        for sample, branches in by_sample.items():
            assert branches['B1']['is_correct'] == next(
                e['historical_v6_correct'] for e in selected_events
                if e['dataset'] == dataset and e['sample_id'] == sample)
        regressions = [b for b in by_sample.values() if b['B1']['group'] == 'regressed']
        recovered = [b for b in by_sample.values() if b['B1']['group'] == 'recovered']
        both_wrong = [b for b in by_sample.values() if b['B1']['group'] == 'both_wrong']
        rank_rescues = {branch: sum(b[branch]['is_correct'] for b in regressions)
                        for branch in ('B2', 'B3', 'B4')}
        any_alt = sum(any(b[branch]['is_correct'] for branch in ('B2', 'B3', 'B4'))
                      for b in regressions)
        summary[dataset] = {
            **info, 'eligible_recovered': len(recovered), 'eligible_regressed': len(regressions),
            'rank_rescues': rank_rescues, 'any_alternative_rescue': any_alt,
            'old_rescue': sum(b['OLD']['is_correct'] for b in regressions),
            'oracle_alt_accuracy': (info['v6_correct'] + any_alt) / info['total'],
            'oracle_gain_pp': 100 * any_alt / info['total'],
            'recovered_top1_only': sum(not any(b[branch]['is_correct']
                for branch in ('B2', 'B3', 'B4')) for b in recovered),
            'recovered_b1_plus_alternative_correct': sum(any(b[branch]['is_correct']
                for branch in ('B2', 'B3', 'B4')) for b in recovered),
            'recovered_multiple_alternatives_correct': sum(sum(bool(b[branch]['is_correct'])
                for branch in ('B2', 'B3', 'B4')) >= 2 for b in recovered),
            'recovered_old_correct': sum(b['OLD']['is_correct'] for b in recovered),
            'both_wrong_alt_rescue': sum(any(b[branch]['is_correct']
                for branch in ('B2', 'B3', 'B4')) for b in both_wrong),
            'both_wrong_old_rescue': sum(b['OLD']['is_correct'] for b in both_wrong),
        }
    (RESULTS / 'v6_revision_candidate_oracle_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    selected_map = {(e['dataset'], e['sample_id']): e for e in selected_events}
    for row in all_rows:
        if row['branch'] != 'B1':
            continue
        event = selected_map[(row['dataset'], row['sample_id'])]
        event['shadow_top4_candidate_ids'] = row['event']['candidate_ids']
        event['shadow_top4_probabilities'] = row['event']['candidate_probabilities']
    (RESULTS / 'v6_revision_candidate_oracle_events.jsonl').write_text(
        ''.join(json.dumps(e, ensure_ascii=False) + '\n' for e in selected_events),
        encoding='utf-8')
    lines = ['# V6 Revision Candidate Oracle', '',
             'Each branch changes exactly one S->H decision and then resumes frozen V6.',
             'Only final official benchmark correctness is used. Oracle-Alt is an upper bound, not a decoder.',
             'GSM8K pairing uses the same-run V1 control (1044/1319), whose output differs from the older archived V1 run (1033/1319) despite matching saved configuration. This avoids mixing run vintages; the paired groups should not be conflated.', '']
    lines += ['| Dataset | Eligible recovered | Eligible regressed | Any Alt Rescue | OLD Rescue |',
              '|---|---:|---:|---:|---:|']
    for d, v in summary.items():
        lines.append(f"| {d} | {v['eligible_recovered']} | {v['eligible_regressed']} | {v['any_alternative_rescue']} | {v['old_rescue']} |")
    lines += ['', '| Dataset | Rank2 | Rank3 | Rank4 | Any Alt |', '|---|---:|---:|---:|---:|']
    for d, v in summary.items():
        ranks = v['rank_rescues']
        lines.append(f"| {d} | {ranks['B2']} | {ranks['B3']} | {ranks['B4']} | {v['any_alternative_rescue']} |")
    lines += ['', '| Dataset | V6 accuracy | Oracle-Alt accuracy | Oracle gain (pp) |',
              '|---|---:|---:|---:|']
    for d, v in summary.items():
        lines.append(f"| {d} | {v['v6_correct']/v['total']:.2%} | {v['oracle_alt_accuracy']:.2%} | {v['oracle_gain_pp']:.2f} |")
    lines += ['', '| Dataset | Eligible recovered | Top1-only | B1 + alternative correct | Multiple alternatives correct | OLD correct |',
              '|---|---:|---:|---:|---:|---:|']
    for d, v in summary.items():
        lines.append(f"| {d} | {v['eligible_recovered']} | {v['recovered_top1_only']} | {v['recovered_b1_plus_alternative_correct']} | {v['recovered_multiple_alternatives_correct']} | {v['recovered_old_correct']} |")
    alt_total = sum(v['any_alternative_rescue'] for v in summary.values())
    old_total = sum(v['old_rescue'] for v in summary.values())
    lines += ['', f'Across eligible regressions, alternative ranks rescue {alt_total} samples and OLD rescues {old_total}.']
    lines += ['', 'MBPP and other samples without an identity-changing V6 event are excluded from replay.',
              'Outcome groups are sample-level; one earliest identity-changing event is replayed per eligible sample.']
    (RESULTS / 'report_v6_revision_candidate_oracle.md').write_text(
        '\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare', 'run', 'finalize'))
    parser.add_argument('--dataset', choices=('gsm8k', 'math500', 'mbpp', 'humaneval'))
    parser.add_argument('--max-samples', type=int)
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare()
    elif args.action == 'run':
        assert args.dataset
        run(args.dataset, args.max_samples)
    else:
        finalize()
