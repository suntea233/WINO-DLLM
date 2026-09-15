"""Replay only OLD; historical V6 is NEW after a per-task deterministic control."""
import argparse,json,os,time
from pathlib import Path
import torch
from transformers import AutoTokenizer
from decoding import decoding_wino_soft_revision_v6
from modeling_llada import LLaDAModelLM
from eval_v6_candidate_oracle import benchmark,prompt_target,extract,DATASETS,MODEL

OUT=Path('/root/lx/WINO-DLLM/LLaDA/results/wino_v6_revision_analysis')
def jl(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s] if Path(p).exists() else []
def main():
 a=argparse.ArgumentParser();a.add_argument('dataset',choices=('gsm8k','math500','mbpp','humaneval'));z=a.parse_args()
 items=[e for e in jl(OUT/'analysis2_manifest.jsonl') if e['dataset']==z.dataset]
 ck=OUT/f'.analysis2_{z.dataset}_old.jsonl';done={r['sample_id'] for r in jl(ck)}
 model=LLaDAModelLM.from_pretrained(MODEL,torch_dtype=torch.bfloat16).cuda().eval();tok=AutoTokenizer.from_pretrained(MODEL,trust_remote_code=True);data=benchmark(z.dataset)
 threshold=.6 if z.dataset=='gsm8k' else DATASETS[z.dataset]['threshold']; checked=(OUT/f'.{z.dataset}_new_determinism_ok').exists()
 with ck.open('a',encoding='utf-8',buffering=1) as f:
  for e in items:
   if e['sample_id'] in done:continue
   ids,target=prompt_target(z.dataset,data[e['sample_id']],tok);base={k:e[k] for k in ('event_id','position','block','round')}
   if not checked:
    out,steps,diag=decoding_wino_soft_revision_v6(model,ids,gen_length=256,block_length=128,temperature=0.,threshold=threshold,threshold_back=.9,beta_mix=.5,return_diagnostics=True,oracle_branch=dict(base,branch='B1'))
    response=tok.batch_decode(out[:,ids.shape[1]:],skip_special_tokens=True)[0]
    assert response==e['historical_v6_response'] and int(steps)==e['historical_v6_nfe'] and diag['oracle_event']['selected_token_id']==e['new_token_id']
    (OUT/f'.{z.dataset}_new_determinism_ok').write_text(str(e['sample_id']));checked=True
   torch.cuda.synchronize();t=time.perf_counter()
   out,steps,diag=decoding_wino_soft_revision_v6(model,ids,gen_length=256,block_length=128,temperature=0.,threshold=threshold,threshold_back=.9,beta_mix=.5,return_diagnostics=True,oracle_branch=dict(base,branch='OLD'))
   torch.cuda.synchronize();response=tok.batch_decode(out[:,ids.shape[1]:],skip_special_tokens=True)[0]
   assert diag['oracle_event']['selected_token_id']==e['old_token_id']
   row={'dataset':z.dataset,'sample_id':e['sample_id'],'old_token_id':e['old_token_id'],'new_token_id':e['new_token_id'],'new_final_output':e['historical_v6_response'],'new_correct':e['historical_v6_correct'],'new_nfe':e['historical_v6_nfe'],'old_final_output':response,'old_nfe':int(steps),'latency':time.perf_counter()-t,**{('old_'+k if k=='is_correct' else k):v for k,v in extract(z.dataset,response,target).items()}}
   f.write(json.dumps(row,ensure_ascii=False)+'\n');print(f"{z.dataset} {len(done)+1}/{len(items)} sample={e['sample_id']}",flush=True);done.add(e['sample_id'])
if __name__=='__main__':main()
