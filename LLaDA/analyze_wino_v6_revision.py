"""Offline Analysis 1 and preparation/finalization for Analysis 2."""
import json, statistics
from collections import defaultdict
from pathlib import Path
from eval_v6_candidate_oracle import evaluate_humaneval_subset_rows,evaluate_mbpp_rows

ROOT=Path('/root/lx/WINO-DLLM/LLaDA/results'); OUT=ROOT/'wino_v6_revision_analysis'
DATASETS=('gsm8k','math500','mbpp','humaneval')

def load(path):return json.loads(Path(path).read_text())
def jl(path):return [json.loads(s) for s in Path(path).read_text().splitlines() if s]
def mean(x):return statistics.fmean(x) if x else 0.
def median(x):return statistics.median(x) if x else 0.

def wino_rows(d):
 evaluated=OUT/f'.wino_{d}_identity_evaluated.jsonl'
 p=(ROOT/'soft_revision/.wino_remask_identity_full.jsonl' if d=='gsm8k' else
    evaluated if evaluated.exists() else OUT/f'.wino_{d}_identity.jsonl')
 return jl(p)

def method_histories(d,method):
 if method=='WINO':
  src=wino_rows(d); result={r['sample_id']:r for r in src}; event_key='events'
 else:
  x=load(ROOT/f'soft_revision_v6/{d}_soft_revision_v6.json');result={r['index']:r for r in x['samples']};event_key=None
 histories=[]; sample_stats={}
 for sid,row in result.items():
  if method=='WINO':
   events=sorted(row[event_key],key=lambda e:(e['remask_round'],e['event_id']))
   norm=[{'challenge_round':e['remask_round'],'return_round':e['rehardening_round'],'position':e.get('absolute_position',e.get('position')),'old':e['original_token_id'],'new':e['new_token_id'],'same':e['same_token']} for e in events]
  else:
   events=sorted(row['diagnostics']['revision_events'],key=lambda e:(e['created_round'],e['event_id']))
   norm=[{'challenge_round':e['created_round'],'return_round':e['resolved_round'],'position':e['position'],'old':e['original_hard_token_id'],'new':e.get('new_token_id'),'same':e.get('final_action')=='S->H(a)'} for e in events]
  bypos=defaultdict(list)
  for e in norm:bypos[e['position']].append(e)
  same_events=repeat_events=repeated_challenges=next_round=0
  for pos,es in bypos.items():
   same_by_id=defaultdict(int)
   for k,e in enumerate(es):
    if not e['same']:continue
    same_events+=1;same_by_id[e['old']]+=1
    prior=same_by_id[e['old']]-1
    if prior:repeat_events+=1
    later=es[k+1] if k+1<len(es) else None
    if later is not None:
     repeated_challenges+=1
     if later['challenge_round']<=e['return_round']+1:next_round+=1
   same_count=sum(e['same'] for e in es)
   histories.append({'dataset':d,'method':method,'sample_id':sid,'position':pos,'challenges':len(es),'same_identity_returns':same_count,'repeated_same_identity_events':sum(max(0,n-1) for n in same_by_id.values()),'repeated_challenges_after_same_return':sum(1 for k,e in enumerate(es) if e['same'] and k+1<len(es)),'challenged_next_round_after_same_return':sum(1 for k,e in enumerate(es) if e['same'] and k+1<len(es) and es[k+1]['challenge_round']<=e['return_round']+1),'rounds_first_challenge_to_stability':es[-1]['return_round']-es[0]['challenge_round'],'events':es})
  sample_stats[sid]={'same_events':same_events,'repeat_events':repeat_events,'repeated_challenges':repeated_challenges,'next_round':next_round,'nfe':int(row['nfe'] if method=='WINO' else row['diagnostics']['nfe']),'correct':bool(row['is_correct'])}
 return histories,sample_stats

def analysis1():
 OUT.mkdir(parents=True,exist_ok=True); allh=[]; summary={'datasets':{},'group_comparison':[]}
 for d in DATASETS:
  if d in ('mbpp','humaneval'):
   traced=wino_rows(d)
   _,passed=(evaluate_mbpp_rows(traced) if d=='mbpp' else evaluate_humaneval_subset_rows(traced))
   for row in traced:row['is_correct']=passed[row['task_id']]
   corrected=OUT/f'.wino_{d}_identity_evaluated.jsonl'
   with corrected.open('w') as f:
    for row in traced:f.write(json.dumps(row)+'\n')
  wh,ws=method_histories(d,'WINO');vh,vs=method_histories(d,'V6');allh+=wh+vh
  src_rows=wino_rows(d);summary['datasets'][d]={'trace_provenance':('historical_exact' if d=='gsm8k' else 'logging_only_replay'),'historical_output_match_rate':sum(r.get('output_matches_historical',True) for r in src_rows)/len(src_rows),'historical_nfe_match_rate':sum(r.get('nfe_matches_historical',True) for r in src_rows)/len(src_rows)}
  for method,h,s in [('WINO',wh,ws),('V6',vh,vs)]:
   ev=sum(x['challenges'] for x in h);same=sum(x['same_identity_returns'] for x in h);affected=[x for x in h if x['same_identity_returns']]
   repeated=[x for x in affected if x['repeated_same_identity_events']]
   summary['datasets'][d][method]={'samples':len(s),'challenged_positions':len(h),'challenge_events':ev,'same_identity_returns':same,'same_identity_return_rate':same/ev if ev else 0,'repeated_loop_positions':len(repeated),'repeated_loop_rate':len(repeated)/len(affected) if affected else 0,'mean_same_returns_per_affected_position':mean([x['same_identity_returns'] for x in affected]),'median_same_returns_per_affected_position':median([x['same_identity_returns'] for x in affected]),'mean_repeated_challenges_per_affected_position':mean([x['repeated_challenges_after_same_return'] for x in affected]),'median_repeated_challenges_per_affected_position':median([x['repeated_challenges_after_same_return'] for x in affected]),'challenged_next_round_fraction':sum(x['challenged_next_round_after_same_return'] for x in affected)/same if same else 0,'repeated_same_identity_events':sum(x['repeated_same_identity_events'] for x in h),'same_identity_events_per_sample':same/len(s),'repeated_same_identity_events_per_sample':sum(x['repeated_same_identity_events'] for x in h)/len(s),'mean_rounds_to_stability':mean([x['rounds_first_challenge_to_stability'] for x in affected]),'median_rounds_to_stability':median([x['rounds_first_challenge_to_stability'] for x in affected]),'mean_nfe':mean([x['nfe'] for x in s.values()])}
  ids=sorted(ws.keys()&vs.keys())
  for label,pred in [('both_correct',lambda a,b:a and b),('WINO_wrong_to_V6_correct',lambda a,b:not a and b),('WINO_correct_to_V6_wrong',lambda a,b:a and not b),('both_wrong',lambda a,b:not a and not b)]:
   g=[i for i in ids if pred(ws[i]['correct'],vs[i]['correct'])]
   summary['group_comparison'].append({'dataset':d,'group':label,'n':len(g),'wino_same_events_mean':mean([ws[i]['same_events'] for i in g]),'v6_same_events_mean':mean([vs[i]['same_events'] for i in g]),'wino_repeated_challenges_mean':mean([ws[i]['repeated_challenges'] for i in g]),'v6_repeated_challenges_mean':mean([vs[i]['repeated_challenges'] for i in g]),'wino_nfe_mean':mean([ws[i]['nfe'] for i in g]),'v6_nfe_mean':mean([vs[i]['nfe'] for i in g])})
 (OUT/'analysis1_same_identity_summary.json').write_text(json.dumps(summary,indent=2))
 with (OUT/'analysis1_position_histories.jsonl').open('w') as f:
  for x in allh:f.write(json.dumps(x)+'\n')
 return summary

def prepare2():
 events=[]
 for d in DATASETS:
  x=load(ROOT/f'soft_revision_v6/{d}_soft_revision_v6.json')
  for row in x['samples']:
   es=[e for e in row['diagnostics']['revision_events'] if e.get('commit_reason')=='different_identity_correction']
   if not es:continue
   e=min(es,key=lambda z:(z['resolved_round'],z['event_id']))
   events.append({'dataset':d,'sample_id':row['index'],'event_id':e['event_id'],'round':e['resolved_round'],'block':e['block'],'position':e['position'],'old_token_id':e['original_hard_token_id'],'new_token_id':e['new_token_id'],'historical_v6_correct':bool(row['is_correct']),'historical_v6_response':row['full_response'],'historical_v6_nfe':int(row['nfe'])})
 with (OUT/'analysis2_manifest.jsonl').open('w') as f:
  for e in events:f.write(json.dumps(e)+'\n')
 print({d:sum(e['dataset']==d for e in events) for d in DATASETS})

if __name__=='__main__':
 import argparse;a=argparse.ArgumentParser();a.add_argument('action',choices=('analysis1','prepare2'));z=a.parse_args();analysis1() if z.action=='analysis1' else prepare2()
