"""Replay an existing WINO result with identity-only H->M->H tracing."""
import argparse, json, os, time
from pathlib import Path
import torch
from transformers import AutoTokenizer
from decoding import decoding_wino_remask
from modeling_llada import LLaDAModelLM
from eval_soft_revision_extended import DATASETS, load_benchmark, format_example
from eval_v6_candidate_oracle import extract

MODEL='/root/lx/ReMix-DLLM/LLaDA/models/LLaDA-8B-Instruct'
OUT=Path('/root/lx/WINO-DLLM/LLaDA/results/wino_v6_revision_analysis')
BASE=Path('/root/lx/WINO-DLLM/LLaDA/results/soft_revision')

def rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s] if Path(path).exists() else []

def main():
    ap=argparse.ArgumentParser();ap.add_argument('dataset',choices=('math500','mbpp','humaneval'));a=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True); checkpoint=OUT/f'.wino_{a.dataset}_identity.jsonl'
    baseline=json.loads((BASE/f'{a.dataset}_wino_remask.json').read_text())
    old={int(r['index']):r for r in baseline['samples']}; done={r['sample_id'] for r in rows(checkpoint)}
    data=load_benchmark(a.dataset); threshold=DATASETS[a.dataset]['threshold']
    model=LLaDAModelLM.from_pretrained(MODEL,torch_dtype=torch.bfloat16).cuda().eval()
    tok=AutoTokenizer.from_pretrained(MODEL,trust_remote_code=True)
    warm_context,_,warm_trailing=format_example(a.dataset,data[0])
    if a.dataset=='math500':warm_trailing+='<reasoning>'
    warm_text=tok.apply_chat_template(warm_context,add_generation_prompt=True,tokenize=False)+warm_trailing
    warm_ids=tok(warm_text,return_tensors='pt').input_ids.cuda()
    decoding_wino_remask(model,warm_ids,gen_length=256,block_length=128,
                         temperature=0.,threshold=threshold,threshold_back=.9)
    with checkpoint.open('a',encoding='utf-8',buffering=1) as f:
      for i,doc in enumerate(data):
        if i in done:continue
        context,target,trailing=format_example(a.dataset,doc)
        if a.dataset=='math500':trailing+='<reasoning>'
        text=tok.apply_chat_template(context,add_generation_prompt=True,tokenize=False)+trailing
        ids=tok(text,return_tensors='pt').input_ids.cuda();torch.cuda.synchronize();t=time.perf_counter()
        out,steps,diag=decoding_wino_remask(model,ids,gen_length=256,block_length=128,temperature=0.,threshold=threshold,threshold_back=.9,return_diagnostics=True)
        torch.cuda.synchronize();response=tok.batch_decode(out[:,ids.shape[1]:],skip_special_tokens=True)[0]
        evaluated=extract(a.dataset,response,target)
        events=[]
        for e in diag['remask_identity_events']:
          e=dict(e);e['absolute_position']=e.pop('position');e['sample_id']=i
          e['original_token_string']=tok.convert_ids_to_tokens(e['original_token_id'])
          e['new_token_string']=tok.convert_ids_to_tokens(e['new_token_id']);events.append(e)
        f.write(json.dumps({'sample_id':i,'is_correct':evaluated.pop('is_correct'),
          'nfe':int(steps),'latency':time.perf_counter()-t,'events':events,
          'full_response':response,'output_matches_historical':response==old[i]['full_response'],
          'nfe_matches_historical':int(steps)==int(old[i]['nfe']),**evaluated},ensure_ascii=False)+'\n')
        print(f'{a.dataset} {i+1}/{len(data)} events={len(events)}',flush=True)

if __name__=='__main__':main()
