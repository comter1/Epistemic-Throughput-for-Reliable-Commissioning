#!/usr/bin/env python3
"""Reproduce the finite-delay study figures from a result ZIP, without rerunning experiments.

Usage:
    python analyze_results.py --input results.zip --out analysis_output

Only reads CSV/JSON/log members. Never extracts or executes archived Python code.
Confidence intervals are pointwise normal intervals, not multiplicity-corrected.
"""
from pathlib import Path
import argparse
import contextlib
import hashlib
import io
import json
import platform
import zipfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def validate_design(configs):
    """Avoid silently pooling different model definitions or duplicate seed runs."""
    expected = {"base", "cal_cost_1", "cal_cost_4", "noise_low", "noise_high",
                "offset_small", "offset_large", "prior_nonuniform", "shared_noise"}
    if {c["condition"] for c in configs} != expected:
        raise ValueError("Expected all nine sweep conditions")
    keys = [(c["condition"], c["seed"]) for c in configs]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate condition/seed: do not mix runs or horizons")
    all_seeds = {c["seed"] for c in configs}
    if len({c["n"] for c in configs}) != 1:
        raise ValueError("Expected equal case counts in this study")
    fields = ["noise_mode", "noise", "offset", "prior", "calibration_cost", "rounds", "bundle_size", "max_budget"]
    for condition in expected:
        group = [c for c in configs if c["condition"] == condition]
        if {c["seed"] for c in group} != all_seeds:
            raise ValueError(f"Missing seed in {condition}")
        if any(any(c[f] != group[0][f] for f in fields) for c in group):
            raise ValueError(f"Model parameters differ across seeds in {condition}")


def analyze(src, out):
    """Audit original summaries, then recompute pooled and paired statistics."""
    z=zipfile.ZipFile(src)
    names=z.namelist()
    roots = [n.rsplit('/', 1)[0] for n in names if n.endswith('/run.log')]
    if len(roots) != 1:
     raise ValueError('ZIP must contain exactly one run directory with run.log')
    root = roots[0]
    log=z.read(root+'/run.log').decode()
    print('LOG errors:',[line for line in log.splitlines() if any(k in line for k in ['Traceback','Error','Exception'])])
    print('LOG tail:', '\n'.join(log.splitlines()[-5:]))
    configs=[]; raws=[]; aggs=[]; bundles=[]; checks=[]
    for name in sorted(n for n in names if n.endswith('/config.json')):
     base=name.rsplit('/',1)[0]; condition=base.split('/')[-1]; seed=int(base.split('/')[-2].split('_')[-1])
     cfg=json.loads(z.read(name))
     assert cfg['seed']==seed
     assert cfg['rounds']==2 and cfg['max_budget']==6 and cfg['bundle_size']==4, 'This plotting layout expects rounds=2, max_budget=6, bundle_size=4'
     configs.append(dict(condition=condition,**cfg))
     def read(file): return pd.read_csv(io.BytesIO(z.read(base+'/'+file)))
     raw=read('per_case_results.csv'); agg=read('main_results.csv'); bundle=read('bundle_points.csv'); paired=read('paired_comparisons.csv')
     for frame in (raw,agg,bundle): frame['condition']=condition; frame['seed']=seed
     assert not raw.duplicated(['policy','budget','case_id']).any()
     assert len(raw)==52*cfg['n'] and len(agg)==49 and len(bundle)==3
     assert np.isfinite(raw[['gain_bits','cost','rounds']].to_numpy()).all()
     assert (raw.cost<=raw.budget).all() and (raw['rounds']<=cfg['rounds']).all()
     assert raw.groupby('case_id').y.nunique().max()==1 and raw.groupby('case_id').x0.nunique().max()==1
     recompute=raw.groupby(['policy','budget']).agg(mean=('gain_bits','mean'),sd=('gain_bits','std'),n=('gain_bits','size'),cost=('cost','mean'),rounds=('rounds','mean'))
     for row in pd.concat([agg,bundle]).itertuples():
      a=recompute.loc[(row.policy,row.budget)]
      assert a.n==row.n==cfg['n']
      np.testing.assert_allclose([a['mean'],a['mean']-1.96*a.sd/np.sqrt(a.n),a['mean']+1.96*a.sd/np.sqrt(a.n),a.cost,a['rounds']], [row.information_gain_bits,row.ci95_low,row.ci95_high,row.mean_cost,row.mean_rounds],atol=1e-12)
     pivot=raw.pivot(index=['budget','case_id'],columns='policy',values='gain_bits')
     for row in paired.itertuples():
      d=(pivot.loc[row.budget,row.policy_a]-pivot.loc[row.budget,row.policy_b]).dropna()
      np.testing.assert_allclose([d.mean(),d.mean()-1.96*d.std()/np.sqrt(len(d)),d.mean()+1.96*d.std()/np.sqrt(len(d))],[row.mean_difference_bits,row.ci95_low,row.ci95_high],atol=1e-12)
     checks.append(dict(condition=condition,seed=seed,records=len(raw),checked=True))
     raws.append(raw); aggs.append(agg); bundles.append(bundle)
    raw=pd.concat(raws,ignore_index=True); agg=pd.concat(aggs,ignore_index=True); bundle=pd.concat(bundles,ignore_index=True)
    validate_design(configs)
    pd.DataFrame(configs).to_json(out/'run_configs.json',orient='records',indent=2,force_ascii=False)
    pd.DataFrame(checks).to_csv(out/'integrity_checks.csv',index=False)
    summary=raw.groupby(['condition','policy','budget']).agg(n=('gain_bits','size'),mean=('gain_bits','mean'),sd=('gain_bits','std'),mean_cost=('cost','mean'),mean_rounds=('rounds','mean'),negative_fraction=('gain_bits',lambda s:(s< -1e-12).mean())).reset_index()
    summary['ci_low']=summary['mean']-1.96*summary.sd/np.sqrt(summary.n); summary['ci_high']=summary['mean']+1.96*summary.sd/np.sqrt(summary.n)
    seedmeans=raw.groupby(['condition','policy','budget','seed']).gain_bits.mean().reset_index()
    seedstats=seedmeans.groupby(['condition','policy','budget']).gain_bits.agg(seed_min='min',seed_max='max',seed_sd='std').reset_index()
    summary=summary.merge(seedstats,on=['condition','policy','budget'])
    summary.to_csv(out/'pooled_summary.csv',index=False);seedmeans.to_csv(out/'seed_means.csv',index=False)
    pairs=[('greedy','greedy_no_calibration'),('greedy','greedy_single'),('greedy_single','greedy_single_no_calibration'),('greedy_no_calibration','greedy_single_no_calibration'),('greedy','repeat'),('greedy','fixed'),('full_bundle','greedy')]
    contrasts=[]
    for condition,group in raw.groupby('condition'):
     p=group.pivot(index=['seed','case_id','budget'],columns='policy',values='gain_bits')
     for a,b in pairs:
      d=(p[a]-p[b]).dropna()
      for budget,g in d.groupby(level='budget'):
       se=g.std()/np.sqrt(len(g)); sm=g.groupby(level='seed').mean()
       contrasts.append(dict(condition=condition,policy_a=a,policy_b=b,budget=budget,n=len(g),difference=g.mean(),ci_low=g.mean()-1.96*se,ci_high=g.mean()+1.96*se,seed_min=sm.min(),seed_max=sm.max()))
    contrasts=pd.DataFrame(contrasts);contrasts.to_csv(out/'paired_effects.csv',index=False)
    dpnames=[n for n in names if n.endswith('/dp_reference.csv')]
    print('DP files:',dpnames)
    if len(dpnames)>1:
     raise ValueError('Multiple DP reference files: compare scopes explicitly before combining')
    if dpnames:
     dp=pd.read_csv(io.BytesIO(z.read(dpnames[0])));dp.to_csv(out/'dp_reference.csv',index=False);print(dp.to_string(index=False))
    print('COUNTS', len(configs),len(raw),'images',sum(n.endswith('.png') for n in names))
    print('BASE ALL\n',summary[(summary.condition=='base') & (summary.policy.isin(['repeat','fixed','greedy','greedy_no_calibration','greedy_single','full_bundle']))].round(5).to_string(index=False))
    print('B6 CONDITION POLICY\n',summary[(summary.budget==6)&summary.policy.isin(['greedy','repeat','greedy_no_calibration','greedy_single'])].pivot(index='condition',columns='policy',values='mean').round(5).to_string())
    print('PAIRED B6\n',contrasts[contrasts.budget==6].round(5).to_string(index=False))
    print('BUNDLES\n',summary[summary.policy.isin(['full_bundle','local_s','local_q'])].round(5).to_string(index=False))
    print('CONFIGS\n',pd.DataFrame(configs)[['condition','seed','n','noise_mode','noise','offset','prior','calibration_cost','rounds','bundle_size','dp','dp_max_budget']].to_string(index=False))
    (out/'archive_audit.txt').write_text(f"Source: {src.name}\nSHA256: {hashlib.sha256(src.read_bytes()).hexdigest()}\n{len(configs)} configurations, {sum(n.endswith('.png') for n in names)} plots, {len(raw)} per-case evaluation rows.\nRecomputed all means, intervals, costs, rounds and supplied paired comparisons. No mismatch.\n"+ '\n'.join(log.splitlines()[-5:]))
    # Fixed case draw distributions: summarize only once per condition/seed (not repeated policies or budgets).
    cases=raw[['condition','seed','case_id','y','x0']].drop_duplicates()
    print('Case counts',cases.groupby('condition').size().to_dict())
    # Initial posterior under each condition: exact enumeration independent of experimental code.
    entropy_rows=[]
    for condition in sorted(cases.condition.unique()):
     cfg=next(c for c in configs if c['condition']==condition)
     delays=[[(0,0)],[(1,0),(2,0)],[(0,1),(0,2)],[(1,1),(1,2),(2,1),(2,2)]]
     joint=np.zeros((7,4));noise=[cfg['noise'],1-2*cfg['noise'],cfg['noise']]
     for y,ds in enumerate(delays):
      for s,q in ds:
       for e,prob in zip([-1,0,1],noise):joint[s+q+e+1,y]+=cfg['prior'][y]/len(ds)*prob
     px=joint.sum(axis=1);post=np.divide(joint,px[:,None],out=np.zeros_like(joint),where=px[:,None]>0)
     ent=-np.sum(np.where(post>0,post*np.log2(np.maximum(post,1e-300)),0),axis=1)
     initial=float(px@ent)
     these=cases[cases.condition==condition]
     prob=post[these.x0.to_numpy()+1,these.y.to_numpy()]
     sample_nll=float(-np.log2(prob).mean())
     entropy_rows.append(dict(condition=condition,exact_initial_entropy=initial,sample_initial_nll=sample_nll))
    pd.DataFrame(entropy_rows).to_csv(out/'initial_entropy.csv',index=False)
    print('INITIAL ENTROPY\n',pd.DataFrame(entropy_rows).to_string(index=False))

    z.close()


def make_plots(out):
    """Plot original study layouts and recover sample NLL from the initial model."""
    s=pd.read_csv(out/'pooled_summary.csv');d=pd.read_csv(out/'paired_effects.csv');seeds=pd.read_csv(out/'seed_means.csv');ent=pd.read_csv(out/'initial_entropy.csv')
    sample_size = int(s['n'].iloc[0])
    seed_list = ', '.join(str(x) for x in sorted(seeds.seed.unique()))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold','figure.facecolor':'white'})
    colors={'greedy':'#1565C0','greedy_no_calibration':'#E58A17','greedy_single':'#8B5CA6','repeat':'#26866A','fixed':'#747C87'}
    labels={'greedy':'Greedy: calibration + bundles','greedy_no_calibration':'Greedy: no calibration','greedy_single':'Greedy: single measurements','repeat':'Repeat total delay','fixed':'Fixed Os then Oq'}
    fig,axs=plt.subplots(2,2,figsize=(14,10),layout='constrained')
    a=axs[0,0]
    for p in colors:
     q=s[(s.condition=='base')&(s.policy==p)].sort_values('budget')
     a.plot(q.budget,q['mean'],marker='o',ms=4,lw=2,color=colors[p],label=labels[p])
     a.fill_between(q.budget,q.ci_low,q.ci_high,color=colors[p],alpha=.08)
    b=s[(s.condition=='base')&(s.policy=='full_bundle')].iloc[0]
    a.errorbar(b.budget,b['mean'],yerr=[[b['mean']-b.ci_low],[b.ci_high-b['mean']]],fmt='D',color='black',capsize=4,label='Fixed full bundle (one round)')
    a.set(title='A. Base condition',xlabel='Budget limit',ylabel='Information gain (bits)',xticks=range(7));a.legend(fontsize=8,loc='upper left');a.grid(alpha=.15)
    a=axs[0,1]
    for c,label,col in [('cal_cost_1','Calibration price = 1','#26866A'),('base','Calibration price = 2','#1565C0'),('cal_cost_4','Calibration price = 4','#C44848')]:
     q=d[(d.condition==c)&(d.policy_a=='greedy')&(d.policy_b=='greedy_no_calibration')].sort_values('budget')
     a.plot(q.budget,q.difference,marker='o',ms=4,color=col,label=label)
     a.fill_between(q.budget,q.ci_low,q.ci_high,color=col,alpha=.12)
    a.axhline(0,color='#555555',ls='--',lw=1)
    a.set(title='B. Effect of allowing calibration',xlabel='Budget limit',ylabel='Paired gain: Greedy minus no-calibration (bits)',xticks=range(7));a.legend(fontsize=8);a.grid(alpha=.15)
    a=axs[1,0]
    order=['prior_nonuniform','offset_small','base','noise_high','offset_large','cal_cost_1','noise_low','cal_cost_4','shared_noise']
    cnames={'base':'Base','cal_cost_1':'Calibration price 1','cal_cost_4':'Calibration price 4','noise_high':'High noise','noise_low':'Low noise','offset_small':'Small offsets','offset_large':'Large offsets','prior_nonuniform':'Nonuniform prior','shared_noise':'Shared-round noise'}
    for i,c in enumerate(order):
     q=d[(d.condition==c)&(d.budget==6)&(d.policy_a=='greedy')&(d.policy_b=='greedy_no_calibration')].iloc[0]
     col='#1565C0' if q.ci_low>0 else '#C44848' if q.ci_high<0 else '#747C87'
     a.errorbar(q.difference,i,xerr=[[q.difference-q.ci_low],[q.ci_high-q.difference]],fmt='o',color=col,capsize=3)
    a.axvline(0,color='#555555',ls='--',lw=1);a.set(yticks=range(len(order)),yticklabels=[cnames[c] for c in order],xlabel='Paired calibration effect at budget 6 (bits)',title='C. Calibration effect depends on condition');a.grid(axis='x',alpha=.15)
    a=axs[1,1]
    for c,col in [('cal_cost_1','#26866A'),('base','#1565C0'),('cal_cost_4','#C44848')]:
     q=s[(s.condition==c)&(s.policy=='greedy')].sort_values('budget');a.plot(q.budget,q['mean'],marker='o',ms=4,color=col,label=cnames[c])
     b=s[(s.condition==c)&(s.policy=='full_bundle')].iloc[0]
     a.errorbar(b.budget,b['mean'],yerr=[[b['mean']-b.ci_low],[b.ci_high-b['mean']]],fmt='D',color=col,capsize=4)
     a.annotate(f'Full bundle: cost {int(b.budget)}',(b.budget,b['mean']),xytext=(0,-24 if c!='base' else 20),textcoords='offset points',ha='center',fontsize=8,color=col)
    a.set(xlim=(-.2,10.8),xticks=range(0,11,2),title='D. Full-bundle cost is shown explicitly',xlabel='Budget limit / actual cost for diamond points',ylabel='Information gain (bits)');a.legend(fontsize=8,loc='upper left');a.grid(alpha=.15)
    fig.suptitle(f'Finite delay-and-clock benchmark: results across seeds {seed_list}',fontsize=16)
    fig.supxlabel(f'{sample_size:,} cases per condition. Shading/error bars: pointwise 95% case-level normal CIs; no multiplicity correction. Two-round horizon.',fontsize=10)
    fig.savefig(out/'results_overview.png',dpi=180)
    fig.savefig(out/'results_overview.svg')
    plt.close(fig)
    # Separate condition charts with consistent scale and seed variation in summary tables.
    fig,axs=plt.subplots(3,3,figsize=(15,11),layout='constrained')
    order=['base','cal_cost_1','cal_cost_4','noise_low','noise_high','shared_noise','offset_small','offset_large','prior_nonuniform']
    for ax,c in zip(axs.flat,order):
     for p in ['greedy','greedy_no_calibration','greedy_single','repeat']:
      q=s[(s.condition==c)&(s.policy==p)].sort_values('budget')
      ax.plot(q.budget,q['mean'],marker='o',ms=3,lw=1.7,color=colors[p],label=labels[p]);ax.fill_between(q.budget,q.ci_low,q.ci_high,color=colors[p],alpha=.08)
     b=s[(s.condition==c)&(s.policy=='full_bundle')].iloc[0]
     ax.errorbar(b.budget,b['mean'],yerr=[[b['mean']-b.ci_low],[b.ci_high-b['mean']]],fmt='D',color='black',capsize=3,label='Fixed full bundle')
     ax.set(title=cnames[c],ylim=(-.03,1.4),xlabel='Budget / full-bundle cost',ylabel='Information gain (bits)',xticks=range(0,11 if c=='cal_cost_4' else 7,2));ax.grid(alpha=.15)
    handles,legend=axs[0,0].get_legend_handles_labels();fig.legend(handles,legend,loc='outside lower center',ncol=3,fontsize=10)
    fig.suptitle('All nine conditions: pooled means and case-level 95% CIs',fontsize=16)
    fig.savefig(out/'all_conditions.png',dpi=170)
    fig.savefig(out/'all_conditions.svg')
    plt.close(fig)
    # Recover sample Bayes NLL using exact initial probabilities and stored per-case log ratios.
    nll=s.merge(ent,on='condition')
    nll['final_sample_nll_bits']=nll.sample_initial_nll-nll['mean']
    nll[['condition','policy','budget','n','sample_initial_nll','mean','final_sample_nll_bits','exact_initial_entropy']].to_csv(out/'recovered_nll.csv',index=False)
    print('Graphs and recovered NLL saved.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Original result ZIP")
    parser.add_argument("--out", type=Path, default=Path("analysis_output"), help="New or empty output directory")
    args = parser.parse_args()
    if not __debug__:
        parser.error("Do not use python -O: integrity assertions must remain enabled")
    if not args.input.is_file():
        parser.error(f"Input does not exist: {args.input}")
    if args.out.exists() and any(args.out.iterdir()):
        parser.error("Output directory is not empty; choose a new --out to preserve previous results")
    args.out.mkdir(parents=True, exist_ok=True)
    print("Reading and checking original case-level results...", flush=True)
    with (args.out / "analysis_details.txt").open("w", encoding="utf-8") as log:
        with contextlib.redirect_stdout(log):
            analyze(args.input, args.out)
            make_plots(args.out)
    environment = {"python": platform.python_version(), "numpy": np.__version__,
                   "pandas": pd.__version__, "matplotlib": matplotlib.__version__,
                   "source_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
                   "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (args.out / "analysis_environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    print(f"Done. Figures and tables: {args.out.resolve()}")
    print("results_overview.png / .svg; all_conditions.png / .svg")
    print("Original ZIP unchanged. No new simulation cases generated.")


if __name__ == "__main__":
    main()
