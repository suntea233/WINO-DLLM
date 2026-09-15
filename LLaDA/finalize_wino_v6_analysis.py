import json,os
from pathlib import Path
from transformers import AutoTokenizer
from eval_v6_candidate_oracle import evaluate_humaneval_subset_rows,evaluate_mbpp_rows

OUT=Path('/root/lx/WINO-DLLM/LLaDA/results/wino_v6_revision_analysis');DS=('gsm8k','math500','mbpp','humaneval')
def jl(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s]
def pc(n,d):return f'{n} ({100*n/d:.1f}%)' if d else '0'

def main():
 a1=json.loads((OUT/'analysis1_same_identity_summary.json').read_text()); rows=[]
 manifest=jl(OUT/'analysis2_manifest.jsonl')
 for d in DS:
  x=jl(OUT/f'.analysis2_{d}_old.jsonl'); expected=sum(e['dataset']==d for e in manifest);assert len(x)==expected,(d,len(x),expected)
  if d=='mbpp':
   _,passed=evaluate_mbpp_rows(x)
   for r in x:r['old_correct']=passed[r['task_id']]
  elif d=='humaneval':
   _,passed=evaluate_humaneval_subset_rows(x)
   for r in x:r['old_correct']=passed[r['task_id']]
  rows+=x
 with (OUT/'analysis2_branch_results.jsonl').open('w') as f:
  for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
 summary={'datasets':{},'aggregate':{k:0 for k in ('samples','new_only','old_only','both_correct','both_wrong')},'by_v6_correctness':{}}
 for d in DS:
  x=[r for r in rows if r['dataset']==d];cats={'new_only':0,'old_only':0,'both_correct':0,'both_wrong':0}
  for r in x:
   k='both_correct' if r['new_correct'] and r['old_correct'] else 'new_only' if r['new_correct'] else 'old_only' if r['old_correct'] else 'both_wrong';cats[k]+=1;r['category']=k
  summary['datasets'][d]={'samples':len(x),**cats,'percentages':{k:(v/len(x) if x else 0) for k,v in cats.items()}}
  summary['aggregate']['samples']+=len(x)
  for k,v in cats.items():summary['aggregate'][k]+=v
 for label,val in [('V6_correct',True),('V6_wrong',False)]:
  x=[r for r in rows if r['new_correct']==val];cats={k:sum(r['category']==k for r in x) for k in ('new_only','old_only','both_correct','both_wrong')};summary['by_v6_correctness'][label]={'samples':len(x),**cats}
 (OUT/'analysis2_identity_change_summary.json').write_text(json.dumps(summary,indent=2))
 tok=AutoTokenizer.from_pretrained('/root/lx/ReMix-DLLM/LLaDA/models/LLaDA-8B-Instruct',trust_remote_code=True)
 lines=['# WINO vs V6 Revision Analysis','','## TABLE 1 — Same-Identity Behavior','', '| Dataset | WINO Same-ID Events/Sample | V6 Same-ID Events/Sample | WINO Repeat Rate | V6 Repeat Rate | WINO NFE | V6 NFE |','|---|---:|---:|---:|---:|---:|---:|']
 for d in DS:
  w=a1['datasets'][d]['WINO'];v=a1['datasets'][d]['V6'];lines.append(f"| {d} | {w['same_identity_events_per_sample']:.2f} | {v['same_identity_events_per_sample']:.2f} | {w['repeated_loop_rate']:.1%} | {v['repeated_loop_rate']:.1%} | {w['mean_nfe']:.2f} | {v['mean_nfe']:.2f} |")
 lines+=['','## TABLE 2 — Identity Change Necessity','', '| Dataset | Identity-Change Samples | NEW-only | OLD-only | Both Correct | Both Wrong |','|---|---:|---:|---:|---:|---:|']
 for d in DS:
  s=summary['datasets'][d];lines.append(f"| {d} | {s['samples']} | {pc(s['new_only'],s['samples'])} | {pc(s['old_only'],s['samples'])} | {pc(s['both_correct'],s['samples'])} | {pc(s['both_wrong'],s['samples'])} |")
 lines+=['','## Analysis 1 details','', '| Dataset | Method | Same-ID Return Rate | Repeated Loop Rate | Same-ID Events/Sample | Mean/median returns per affected position | Mean/median repeated challenges | Next-round challenge | Mean rounds to stability | NFE |','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
 for d in DS:
  for m in ('WINO','V6'):
   s=a1['datasets'][d][m];lines.append(f"| {d} | {m} | {s['same_identity_return_rate']:.1%} | {s['repeated_loop_rate']:.1%} | {s['same_identity_events_per_sample']:.2f} | {s['mean_same_returns_per_affected_position']:.2f}/{s['median_same_returns_per_affected_position']:.1f} | {s['mean_repeated_challenges_per_affected_position']:.2f}/{s['median_repeated_challenges_per_affected_position']:.1f} | {s['challenged_next_round_fraction']:.1%} | {s['mean_rounds_to_stability']:.2f} | {s['mean_nfe']:.2f} |")
 lines+=['','### Matched outcome groups','', '| Dataset | Group | N | WINO Same-ID Events | V6 Same-ID Events | WINO repeated challenges | V6 repeated challenges | WINO NFE | V6 NFE |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
 for g in a1['group_comparison']:lines.append(f"| {g['dataset']} | {g['group']} | {g['n']} | {g['wino_same_events_mean']:.2f} | {g['v6_same_events_mean']:.2f} | {g['wino_repeated_challenges_mean']:.2f} | {g['v6_repeated_challenges_mean']:.2f} | {g['wino_nfe_mean']:.2f} | {g['v6_nfe_mean']:.2f} |")
 agg=summary['aggregate'];lines+=['','## Analysis 2 aggregate','',f"Across {agg['samples']} samples: NEW-only {pc(agg['new_only'],agg['samples'])}, OLD-only {pc(agg['old_only'],agg['samples'])}, both correct {pc(agg['both_correct'],agg['samples'])}, both wrong {pc(agg['both_wrong'],agg['samples'])}.",'','| Historical outcome | N | NEW-only | OLD-only | Both correct | Both wrong |','|---|---:|---:|---:|---:|---:|']
 for k,s in summary['by_v6_correctness'].items():lines.append(f"| {k} | {s['samples']} | {s['new_only']} | {s['old_only']} | {s['both_correct']} | {s['both_wrong']} |")
 lines+=['','## Representative cases']
 for d in DS:
  lines+=['',f'### {d}']
  for cat,title in [('new_only','NEW-only'),('old_only','OLD-only')]:
   found=next((r for r in rows if r['dataset']==d and r['category']==cat),None)
   if not found:lines.append(f'- {title}: none observed.')
   else:
    old=tok.convert_ids_to_tokens(found['old_token_id']);new=tok.convert_ids_to_tokens(found['new_token_id'])
    lines.append(f"- {title}, sample {found['sample_id']}: `{old}` → `{new}`. NEW final correctness={found['new_correct']}; OLD final correctness={found['old_correct']}. The single forced identity choice changes the continuation and only the stated branch passes the final evaluator.")
 # Conclusions intentionally only two.
 wins=sum(a1['datasets'][d]['V6']['repeated_same_identity_events_per_sample']<a1['datasets'][d]['WINO']['repeated_same_identity_events_per_sample'] for d in DS)
 answer='YES' if wins==4 else 'PARTIALLY' if wins else 'NO'
 dominant=max(('necessary','harmful','redundant','ineffective'),key=lambda x:agg[{'necessary':'new_only','harmful':'old_only','redundant':'both_correct','ineffective':'both_wrong'}[x]])
 lines+=['','## Conclusions','',f"1. Does Soft improve WINO repeated same-identity revision behavior? **{answer}**. This refers to observed loop frequency/duration, not token correctness.",f"2. Across first identity-changing events, the most common observed category is **{dominant}** ({agg[{'necessary':'new_only','harmful':'old_only','redundant':'both_correct','ineffective':'both_wrong'}[dominant]]}/{agg['samples']})."]
 (OUT/'report_wino_v6_revision_analysis.md').write_text('\n'.join(lines)+'\n')
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
