"""Complete denominators, paired family/object bootstrap, and measured figures."""
from __future__ import annotations
import itertools
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .protocol import METHODS,REGISTERED,write_json,file_hash

PALETTE=['#777777','#4477AA','#66CCEE','#228833','#CCBB44','#EE6677']

def offline_rank_one(root,case_id,base_id):
    """Registered source precision diagnostic; never an online policy decision.

Each candidate uses identical deterministic draws, so independent candidate jobs
produce exactly the same score array as one 256/32 batch. No held-out data enter.
The earlier ten-second attempt is retained as a timeout, not called a completed check.
"""
    import time
    from .protocol import Ledger,digest
    from .cases import policy_view
    from .evaluate import setup
    from .prediction import Clock
    from .shared_decision import deployment_choices
    from .policies import Selector
    root=Path(root);path=root/'source_high_precision'/case_id/(base_id+'.json')
    if path.exists(): return json.loads(path.read_text())
    ledger=Ledger(root,'cpu'); token=ledger.reserve(600,unit=case_id+'_'+base_id,kind='source_256_32_offline_diagnostic');begin=time.process_time();wall=time.perf_counter()
    case=next(c for c in json.loads((root/'private/source.json').read_text()) if c['case_id']==case_id)
    row=json.loads((root/'jobs/branches'/(case_id+'.json')).read_text())['result'];pub=policy_view(case)
    ctx,est,pred,bank,_=setup(root,pub,'adaptive_work_decision')
    mean=np.asarray(row['post_B0_estimate']['joint_mean17']);cov=np.asarray(row['post_B0_estimate']['joint_covariance17'])
    with np.load(root/'raw/branches'/case_id/'prefix/public.npz') as z: start=z['q'][-1]
    choices,_,_=deployment_choices(pred,pub['configured_payload'],row['post_B0_estimate']['payload_estimate'],start,mean,cov,Clock(120),False,True)
    screen=next(s for s in row['screens'] if s['base_id']==base_id and s['admitted'])
    entry=bank['actions'][f'{base_id}@{screen["speed"]}'];candidate=dict(entry['action'],information17=entry['information17'])
    selector=Selector('adaptive_work_decision',bank,pub['configured_payload'],est.parameters['posterior_covariance_inflation'])
    scores=selector.score(mean,cov,[candidate],pred,choices,Clock(120),kind='decision',samples=256,outcomes=32)
    result=dict(case_id=case_id,context=case['context'],score=scores[0],parameter_samples=256,outcomes=32,
                cpu_seconds=time.process_time()-begin,wall_seconds=time.perf_counter()-wall,offline_diagnostic=True,
                completed_after_primary_freeze=True,retuning=False,source_history_sha256=file_hash(root/'jobs/branches'/(case_id+'.json')))
    ledger.complete(token,time.process_time()-begin);write_json(path,result);return result

def load_results(root,stage):
    return [json.loads(p.read_text())['result'] for p in sorted((Path(root)/'jobs'/stage).glob('*.json'))]

def scalar_frame(rows):
    return pd.DataFrame([{k:(json.dumps(v,separators=(',',':')) if isinstance(v,(dict,list)) else v)
                          for k,v in r.items() if k not in ('steps','blocks','audits','estimate')} for r in rows])

def comparisons(df):
    pairs=[('adaptive_work_decision','plan_after_b0'),('plan_after_b0','fixed_prior_plan'),
           ('adaptive_work_decision','simple_parameter_rule'),('adaptive_work_decision','adaptive_variance'),
           ('adaptive_work_decision','b0_verify_only')]
    keys=['case_id','family_id','object_id','context','noise_seed']; output=[]; all_paired=[]
    for treatment,control in pairs:
        columns=keys+['normal_correct','false_release','physical_violation','identification_seconds','physical_seconds','serial_seconds','capped_serial_seconds']
        paired=df[df.method==treatment][columns].merge(df[df.method==control][columns],on=keys,suffixes=('_t','_c'))
        for m in ['normal_correct','false_release','physical_violation','capped_serial_seconds','serial_seconds']:
            paired[m+'_difference']=paired[m+'_t'].astype(float)-paired[m+'_c'].astype(float)
        both=paired.normal_correct_t&paired.normal_correct_c
        for m in ['identification_seconds','physical_seconds']:
            paired[m+'_difference']=np.where(both,paired[m+'_t']-paired[m+'_c'],np.nan)
        paired['both_correct_normal']=both; paired['treatment']=treatment; paired['control']=control
        all_paired.append(paired)
        metrics=['normal_correct','false_release','physical_violation','identification_seconds','physical_seconds','serial_seconds','capped_serial_seconds']
        cases=paired.groupby(['family_id','object_id','context'])[[m+'_difference' for m in metrics]].mean().reset_index()
        # Contexts of an object are never sampled independently.
        objects=cases.groupby(['family_id','object_id'])[[m+'_difference' for m in metrics]].mean()
        families=sorted(objects.index.get_level_values(0).unique())
        blocks=[objects.loc[f].to_numpy() for f in families]
        rng=np.random.default_rng(REGISTERED['bootstrap_seed']); draws=[]
        for _ in range(2000):
            values=[]
            for i in rng.integers(0,len(blocks),len(blocks)):
                b=blocks[i]; values.append(b[rng.integers(0,len(b),len(b))])
            draws.append(np.nanmean(np.concatenate(values),axis=0))
        draws=np.asarray(draws)
        for j,m in enumerate(metrics):
            vals=objects[m+'_difference'].to_numpy()
            output.append(dict(treatment=treatment,control=control,metric=m,difference=float(np.nanmean(vals)),
                lower_95=float(np.nanquantile(draws[:,j],.025)),upper_95=float(np.nanquantile(draws[:,j],.975)),
                one_sided_lower_95=float(np.nanquantile(draws[:,j],.05)),one_sided_upper_95=float(np.nanquantile(draws[:,j],.95)),
                paired_runs=len(paired),paired_normal_success_runs=int(both.sum()),objects=len(objects),families=len(families),replicates=2000,
                primary=(treatment,control)==pairs[0]))
    return pd.DataFrame(output),pd.concat(all_paired,ignore_index=True)

def tex_plot(path,csv_name,series,x='x',y='y',xlabel='',ylabel='',extra=''):
    colors=[]; commands=[]
    for i,(label,filename,col) in enumerate(series):
        hx=PALETTE[i%len(PALETTE)].lstrip('#'); rgb=','.join(str(int(hx[k:k+2],16)) for k in (0,2,4))
        colors.append('\\definecolor{c'+str(i)+'}{RGB}{'+rgb+'}')
        mark='*' if 'only marks' in extra else 'none'
        commands.append('\\addplot[color=c'+str(i)+',mark='+mark+'] table[x='+x+',y='+col+',col sep=comma] {'+filename+'};\n\\addlegendentry{'+label.replace('_','\\_')+'}')
    path.write_text('\\documentclass[tikz,border=4pt]{standalone}\n\\usepackage{pgfplots}\n\\pgfplotsset{compat=1.18}\n'+'\n'.join(colors)+
        '\n\\begin{document}\n\\begin{tikzpicture}\n\\begin{axis}[width=12cm,height=7cm,xlabel={'+xlabel+'},ylabel={'+ylabel+'},legend style={font=\\tiny},'+extra+']\n'+
        '\n'.join(commands)+'\n\\end{axis}\n\\end{tikzpicture}\n\\end{document}\n')

def figures(root,df,paired,source,source_predictions,test_predictions,examples,ablations):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from .trajectories import grid,work,library
    from payload_change_v1.physics import Dynamics
    root=Path(root); out=root/'figures'; out.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    def save(fig,name):
        fig.savefig(out/(name+'.pdf'),bbox_inches='tight'); fig.savefig(out/(name+'.png'),dpi=180,bbox_inches='tight'); plt.close(fig)
    contexts=json.loads((root/'configs/workstations.json').read_text()); kin=Dynamics(None)
    for name,ctx in contexts.items():
        fig=plt.figure(figsize=(5,4)); ax=fig.add_subplot(111,projection='3d'); rows=[]; edges=[]
        for action in [work(ctx),*library()]:
            xyz=np.array([kin.fk(q) for q in grid(action,hz=10)['q']])
            is_work=action['id']=='work'
            ax.plot(*xyz.T,color='black' if is_work else PALETTE[0],alpha=1 if is_work else .25,lw=2 if is_work else .6)
            rows.extend(dict(path=action['id'],point=i,x=p[0],y=p[1],z=p[2]) for i,p in enumerate(xyz))
        for f in ctx['fixtures']:
            center=np.asarray(f['center']); half=np.asarray(f['size'])/2
            vertices=np.array([center+half*np.array(s) for s in itertools.product((-1,1),repeat=3)])
            for i,j in itertools.combinations(range(8),2):
                if sum(a!=b for a,b in zip(list(itertools.product((-1,1),repeat=3))[i],list(itertools.product((-1,1),repeat=3))[j]))==1:
                    ax.plot(*vertices[[i,j]].T,color=PALETTE[1],alpha=.7)
                    edges.append(dict(x=vertices[i,0],y=vertices[i,1],z=vertices[i,2],x2=vertices[j,0],y2=vertices[j,1],z2=vertices[j,2]))
        ax.set(xlabel='x (m)',ylabel='y (m)',zlabel='z (m)',title=name+': public hand paths and fixture edges\nBlack: job; gray: 10 test patterns (geometry plot)')
        table=pd.DataFrame(rows); table.to_csv(out/f'01_{name}_geometry.csv',index=False); pd.DataFrame(edges).to_csv(out/f'01_{name}_fixtures.csv',index=False)
        commands=[]
        for path,g in table.groupby('path',sort=False):
            file=f'01_{name}_{path}.csv';g.to_csv(out/file,index=False)
            commands.append('\\addplot3[mark=none] table[x=x,y=y,z=z,col sep=comma] {'+file+'};')
        for edge in edges:
            commands.append('\\addplot3[blue,mark=none] coordinates {('+','.join(str(edge[k]) for k in ('x','y','z'))+') ('+','.join(str(edge[k]) for k in ('x2','y2','z2'))+')};')
        (out/f'01_{name}.tex').write_text('\\documentclass[tikz,border=4pt]{standalone}\n\\usepackage{pgfplots}\n\\pgfplotsset{compat=1.18}\n\\begin{document}\\begin{tikzpicture}\\begin{axis}[xlabel={x (m)},ylabel={y (m)},zlabel={z (m)}]\n'+'\n'.join(commands)+'\n\\end{axis}\\end{tikzpicture}\\end{document}')
        save(fig,f'01_{name}')
    action_ids=[a['id'] for a in library()]; history_ids=sorted(source.case_id.unique()); heat=[]
    for i,cid in enumerate(history_ids):
        s=source[(source.case_id==cid)&(source.method=='adaptive_work_decision')]
        for j,a in enumerate(action_ids):
            r=s[s.base_id==a]
            heat.append(dict(history=i,action=j,case_id=cid,base_id=a,score=float(r.score.iloc[0]) if len(r) else np.nan,
                             speed=float(r.speed.iloc[0]) if len(r) else 0))
    heat=pd.DataFrame(heat);heat.to_csv(out/'02_action_map.csv',index=False)
    matrix=heat.pivot(index='history',columns='action',values='score').to_numpy()
    fig,ax=plt.subplots(figsize=(7,7));im=ax.imshow(matrix,aspect='auto',cmap='viridis');fig.colorbar(im,ax=ax,label='Expected decision bits per physical second')
    ax.set(xticks=range(10),xticklabels=action_ids,yticks=range(len(history_ids)),yticklabels=[str(i+1) for i in range(len(history_ids))],ylabel='Actual source B0 history (n=24)',title='Forced-continue scores; release already eligible is kept explicit')
    for r in heat.itertuples(): ax.text(r.action,r.history,str(r.speed) if r.speed else 'x',ha='center',va='center',fontsize=5,color='white')
    save(fig,'02_action_map')
    (out/'02_action_map.tex').write_text('\\documentclass[tikz,border=4pt]{standalone}\n\\usepackage{pgfplots}\n\\pgfplotsset{compat=1.18}\n\\begin{document}\\begin{tikzpicture}\\begin{axis}[colorbar,xlabel={Base action index},ylabel={Source history index}]\\addplot[matrix plot*,mesh/cols=10,point meta=explicit] table[x=action,y=history,meta=score,col sep=comma] {02_action_map.csv};\\end{axis}\\end{tikzpicture}\\end{document}')
    for state,label in [('normal_correct','normal'),('half_correct','half')]:
        times=np.arange(0,80.01,.25); curve=pd.DataFrame({'time':times}); fig,ax=plt.subplots(figsize=(7,4)); series=[]
        for i,(m,name) in enumerate(METHODS.items()):
            g=df[df.method==m];curve[m]=[float(((g[state])&(g.physical_seconds<=t)).mean()) for t in times]
            ax.step(times,curve[m],where='post',label=name,color=PALETTE[i],ls=['-',':','--','-.',':','--'][i]);series.append((name,f'03_{label}_curve.csv',m))
        curve.to_csv(out/f'03_{label}_curve.csv',index=False)
        ax.set(xlabel='Actual physical commissioning time (s)',ylabel=f'Correct {label}-speed release / all runs',ylim=(-.02,1.02),title='576 runs per method; 64 objects, 3 contexts, 3 noise seeds')
        ax.legend(fontsize=6); save(fig,f'03_{label}_curve')
        tex_plot(out/f'03_{label}_curve.tex','',series,x='time',xlabel='Physical commissioning time (s)',ylabel='Correct releases / all runs',extra='ymin=0,ymax=1')
    aggregate=df.groupby('method').agg(coverage=('normal_correct','mean'),capped_time=('capped_serial_seconds','mean'),half=('half_correct','mean'),unresolved=('release',lambda x:1-x.mean()),false=('false_release','mean')).reindex(METHODS).reset_index()
    aggregate['method_index']=range(6);aggregate.to_csv(out/'04_coverage_time.csv',index=False)
    fig,ax=plt.subplots(figsize=(7,4))
    for i,r in aggregate.iterrows(): ax.scatter(r.capped_time,r.coverage,color=PALETTE[i],label=METHODS[r.method],s=65)
    ax.set(xlabel='All-case capped serial time (s); unsuccessful normal release = 160 s',ylabel='Correct normal-speed coverage',ylim=(0,1.05));ax.legend(fontsize=6);save(fig,'04_coverage_time')
    tex_plot(out/'04_coverage_time.tex','',[('Six methods','04_coverage_time.csv','coverage')],x='capped_time',xlabel='Capped serial time (s)',ylabel='Correct normal coverage',extra='only marks')
    primary=paired[(paired.treatment=='adaptive_work_decision')&(paired.control=='plan_after_b0')]
    directions=primary.groupby(['family_id','context']).agg(difference=('physical_seconds_difference','mean'),n=('both_correct_normal','sum')).reset_index();directions['index']=range(len(directions));directions.to_csv(out/'05_paired_time.csv',index=False)
    fig,ax=plt.subplots(figsize=(8,3));ax.axhline(0,color='black',lw=.7);ax.scatter(directions['index'],directions.difference,color=PALETTE[-1]);ax.set(xlabel='Family/context stratum (8 families x 3 contexts)',ylabel='Adaptive minus plan-once (s)',title='Matched correct normal-speed releases');save(fig,'05_paired_time')
    tex_plot(out/'05_paired_time.tex','',[('Paired physical time','05_paired_time.csv','difference')],x='index',xlabel='Family/context stratum',ylabel='Adaptive minus plan-once (s)',extra='only marks')
    for kind,xcol,ycol in [('tracking','predicted_true_peak_m','actual_peak_m'),('torque','predicted_true_torque_Nm','actual_torque_Nm'),('clearance','predicted_true_clearance_m','actual_clearance_m')]:
        data=source_predictions.copy()
        if kind=='clearance': data=data[data.context!='W0']
        data[['case_id','context','motion_kind','speed',xcol,ycol]].to_csv(out/f'06_source_{kind}.csv',index=False)
        fig,ax=plt.subplots(figsize=(4,4));ax.scatter(data[xcol],data[ycol],s=5,alpha=.35,color=PALETTE[1]);mx=float(max(data[xcol].max(),data[ycol].max()));ax.plot([0,mx],[0,mx],'k--',lw=.7)
        ax.set(xlabel='Public first-order prediction at source label',ylabel='Actual private physical measurement',title=f'{kind}: source validation, n={len(data)}');save(fig,f'06_source_{kind}')
        tex_plot(out/f'06_source_{kind}.tex','',[(kind,f'06_source_{kind}.csv',ycol)],x=xcol,xlabel='Public prediction at source label',ylabel='Actual measurement',extra='only marks,mark size=0.6pt')
    if len(test_predictions):
        test_predictions.to_csv(out/'06_test_work_bounds.csv',index=False)
        fig,ax=plt.subplots(figsize=(4,4));ax.scatter(test_predictions.predicted_peak_m,test_predictions.actual_peak_m,s=5,alpha=.3,color=PALETTE[-1]);mx=max(test_predictions.predicted_peak_m.max(),test_predictions.actual_peak_m.max());ax.plot([0,mx],[0,mx],'k--',lw=.7);ax.set(xlabel='Predicted joint-set peak bound (m)',ylabel='Actual peak error (m)',title=f'Test verification: n={len(test_predictions)}');save(fig,'06_test_work_bounds')
        tex_plot(out/'06_test_work_bounds.tex','',[('Test work','06_test_work_bounds.csv','actual_peak_m')],x='predicted_peak_m',xlabel='Predicted peak bound (m)',ylabel='Actual peak (m)',extra='only marks,mark size=0.6pt')
    for i,ex in enumerate(examples):
        rows=[];fig,ax=plt.subplots(figsize=(7,2.5))
        for level,m in enumerate(('plan_after_b0','adaptive_work_decision')):
            record=ex['runs'][m]; t=0.
            for b in record['blocks']:
                duration=b['physical_seconds'];rows.append(dict(method=m,start=t,end=t+duration,kind=b['kind'],action=b['action']['id'],peak_error_m=b['metrics']['peak_error_m']))
                ax.barh(level,duration,left=t,color=PALETTE[2] if b['kind']=='identification' else PALETTE[3],edgecolor='white');ax.text(t+duration/2,level,b['action']['id'],ha='center',va='center',fontsize=8);t+=duration
        data=pd.DataFrame(rows);data.to_csv(out/f'07_timeline_{i+1}.csv',index=False)
        ax.set(yticks=[0,1],yticklabels=['Plan once after B0','Rechoose decisions'],xlabel='Actual physical commissioning time (s)',title=ex['description']);save(fig,f'07_timeline_{i+1}')
        (out/f'07_timeline_{i+1}.tex').write_text('\\documentclass[tikz,border=4pt]{standalone}\n\\begin{document}\\begin{tikzpicture}[x=.22cm,y=.7cm]\n'+'\n'.join('\\draw[fill='+('cyan!40' if r['kind']=='identification' else 'green!35')+'] ('+str(r['start'])+','+str(0 if r['method']=='plan_after_b0' else 1)+') rectangle ('+str(r['end'])+','+str(.7 if r['method']=='plan_after_b0' else 1.7)+');' for r in rows)+'\n\\end{tikzpicture}\\end{document}')
    ablation_rows=[]
    for row in ablations:
        actual=next(v for v in row['variants'] if v['variant']=='actual').get('selected')
        for variant in row['variants']:
            ablation_rows.append(dict(case_id=row['case_id'],context=row['context'],variant=variant['variant'],selected=variant.get('selected'),actual=actual,changed=variant.get('selected')!=actual))
    abl=pd.DataFrame(ablation_rows);abl.to_csv(out/'08_priority_interventions.csv',index=False)
    agg=abl[abl.variant!='actual'].groupby(['context','variant']).changed.agg(['mean','count']).reset_index();agg['index']=range(len(agg));agg.to_csv(out/'08_priority_fractions.csv',index=False)
    fig,ax=plt.subplots(figsize=(7,3));ax.bar(agg['index'],agg['mean'],color=[PALETTE[2] if v=='freeze_mean_to_prior' else PALETTE[4] for v in agg.variant]);ax.set(xticks=agg['index'],xticklabels=[r.context+' / '+r.variant.replace('_to_prior','').replace('freeze_','') for r in agg.itertuples()],ylabel='Next choice changed / 8 source histories',ylim=(0,1),title='Fixed admitted set and fixed work-deployment set');save(fig,'08_priority_interventions')
    tex_plot(out/'08_priority_interventions.tex','',[('Score interventions','08_priority_fractions.csv','mean')],x='index',xlabel='Context/intervention',ylabel='Choice changed fraction',extra='ybar,ymin=0,ymax=1')

def report(root):
    root=Path(root); rows=load_results(root,'evaluation'); references=load_results(root,'reference')
    assert len(rows)==3456 and len(references)==576,(len(rows),len(references))
    # Reproduction also completes the registered larger-sample source diagnostic
    # if its bounded online attempt timed out. This is analysis, never retuning.
    rank_jobs=[]
    for p in (root/'jobs/branches').glob('*.json'):
        r=json.loads(p.read_text())['result']
        rank_jobs.extend((str(root),r['case_id'],s['base_id']) for s in r['screens'] if s['admitted']
                         and not (root/'source_high_precision'/r['case_id']/(s['base_id']+'.json')).exists())
    if rank_jobs:
        from concurrent.futures import ProcessPoolExecutor,as_completed
        with ProcessPoolExecutor(max_workers=8) as pool:
            for future in as_completed([pool.submit(offline_rank_one,*j) for j in rank_jobs]): future.result()
    df=scalar_frame(rows);df['physical_violation']=(df.identification_violations+df.verification_violations+df.audit_violations)>0
    assert df.groupby('method').size().eq(576).all()
    df.to_parquet(root/'per_run_results.parquet',index=False);df.to_csv(root/'per_run_results.csv',index=False)
    steps=[];scores=[];prediction_rows=[];thresholds=[]
    for r in rows:
        key={k:r[k] for k in ('case_id','object_id','family_id','context','method','noise_seed')}
        for s in r['steps']:
            steps.append(key|{k:(json.dumps(v,separators=(',',':')) if isinstance(v,(dict,list)) else v) for k,v in s.items()})
            for score in s.get('candidate_scores',[]):scores.append(key|dict(step=s['step'],role='online_next_action',**score))
            for node in s.get('plan_trace',[]):
                for score in node.get('scores',[]): scores.append(key|dict(step=s['step'],role='fixed_priority_search',search_depth=node.get('depth'),**score))
            if 'chosen_work' in s:
                actual=next((b['metrics'] for b in r['blocks'] if b['kind']=='verification'),None)
                if actual:
                    p=s['chosen_work']['prediction'];prediction_rows.append(key|dict(predicted_peak_m=p['peak_m'],actual_peak_m=actual['peak_error_m'],
                        predicted_hold_m=p['hold_m'],actual_hold_m=actual['final_hold_peak_m'],predicted_torque_Nm=max(p['torque_Nm']),
                        actual_torque_Nm=max(actual['peak_requested_Nm']),predicted_clearance_m=p['clearance_m'],actual_clearance_m=actual['minimum_clearance_m']))
        for label,rms,peak,hold in [('V1_work',.02,.05,.02),('intermediate',.01,.02,.005),('V2_primary',.005,.01,.002)]:
            passed=sum(a['metrics']['rms_error_m']<=rms and a['metrics']['peak_error_m']<=peak and a['metrics']['final_hold_peak_m']<=hold
                       and not a['metrics']['hard_abort'] and not a['metrics']['unexpected_contact'] and not a['metrics']['joint_limit_violation']
                       and a['metrics']['saturation_fraction']<=.01+1e-12 and a['metrics']['max_continuous_saturation_seconds']<=.1+1e-12 for a in r['audits'])
            thresholds.append(key|dict(threshold=label,audit_pass_count=passed,approved=r['release'],speed=r['final_speed']))
    pd.DataFrame(steps).to_parquet(root/'per_step_actions.parquet',index=False)
    pd.DataFrame(scores,columns=None if scores else ['case_id','method','step','base_id','score','role']).to_parquet(root/'candidate_scores.parquet',index=False)
    pd.DataFrame(thresholds).to_csv(root/'descriptive_threshold_sweep.csv',index=False)
    categories=[]
    for r in references:
        row={k:r[k] for k in ('case_id','object_id','family_id','context','group','noise_seed','category')}
        row.update({a['condition']+'_pass':a['metrics']['pass'] for a in r['conditions']});categories.append(row)
    ref=pd.DataFrame(categories);ref.to_csv(root/'work_reference_categories.csv',index=False)
    update_rows=pd.DataFrame([{k:r[k] for k in ('case_id','noise_seed','method','normal_correct')}|
                             dict(setting_updated=bool(r['updates'])) for r in rows])
    update_rows.merge(ref[['case_id','noise_seed','initial_normal_pass','true_normal_pass']],on=['case_id','noise_seed']).to_csv(root/'setting_updates_vs_reference.csv',index=False)
    primary_stats,paired=comparisons(df);primary_stats.to_csv(root/'paired_method_comparisons.csv',index=False)
    paired.to_parquet(root/'all_paired_runs.parquet',index=False)
    pairs=df[df.method=='adaptive_work_decision'][['case_id','noise_seed','motion_order','motion_speeds']].merge(df[df.method=='plan_after_b0'][['case_id','noise_seed','motion_order','motion_speeds']],on=['case_id','noise_seed'],suffixes=('_adaptive','_once'))
    pairs['identical']=(pairs.motion_order_adaptive==pairs.motion_order_once)&(pairs.motion_speeds_adaptive==pairs.motion_speeds_once);pairs.to_csv(root/'identical_schedule_pairs.csv',index=False)
    summary=df.groupby('method').agg(runs=('normal_correct','size'),normal_correct=('normal_correct','sum'),half_correct=('half_correct','sum'),release=('release','sum'),false_release=('false_release','sum'),
        physical_violations=('physical_violation','sum'),identification_tracking_failures=('identification_tracking_failures','sum'),identification_s=('identification_seconds','mean'),verification_s=('verification_seconds','mean'),serial_s=('serial_seconds','mean'),capped_serial_s=('capped_serial_seconds','mean')).reindex(METHODS).reset_index()
    summary['false_release_over_all']=summary.false_release/summary.runs;summary['false_release_over_released']=summary.false_release/summary.release.replace(0,np.nan)
    summary['unresolved_or_aborted']=summary.runs-summary.release
    summary.to_csv(root/'constraint_and_false_release_counts.csv',index=False)
    timings=[]
    for r in rows:
        # Travel/hold are components of recorded test trajectories, never extra free costs.
        travel=0.;hold=0.
        for b in r['blocks']:
            if b['kind']!='identification': continue
            a=b['action'];duration=b['physical_seconds']
            if a['kind']=='initial': travel+=min(duration,4.);hold+=max(0.,duration-4.)
            elif a['kind']=='static': travel+=duration*.75;hold+=duration*.25
            else: travel+=duration
        timings.append({k:r[k] for k in ('case_id','method','noise_seed','identification_seconds','verification_seconds','physical_seconds','serial_seconds','observed_wall_based_serial_seconds')}|r['compute']|
            {k+'_wall':v for k,v in r.get('compute_wall',{}).items()}|dict(test_travel_component_s=travel,test_hold_settling_component_s=hold))
    pd.DataFrame(timings).to_csv(root/'time_breakdown.csv',index=False)
    branch=load_results(root,'branches');ranking=[]
    for r in branch:
        ranking.extend(r['rankings'])
        for m in ('fixed_prior_plan','plan_after_b0'):
            for rank,a in enumerate(r['selectors'][m].get('order') or []):
                sc=next((s for s in r['screens'] if s['base_id']==a and s['admitted']),None)
                ranking.append(dict(case_id=r['case_id'],context=r['context'],method=m,base_id=a,score=-rank,speed=sc['speed'] if sc else 0.,score_kind='fixed_priority_rank'))
    source=pd.DataFrame(ranking);source.to_csv(root/'source_action_rankings.csv',index=False)
    branch_records=[]
    for r in branch:
        for b in r['branches']:
            outcome=json.loads(Path(b['result_path']).read_text());entry=dict(b)
            choice=next((s['chosen_work']['prediction'] for s in outcome['steps'] if 'chosen_work' in s),None)
            actual=next((block['metrics'] for block in outcome['blocks'] if block['kind']=='verification'),None)
            entry.update(eventual_commissioning_physical_seconds=outcome['physical_seconds'],final_speed=outcome['final_speed'],
                         stop_reason=outcome['stop_reason'],audit_violations=outcome['audit_violations'])
            if choice and actual:
                entry.update(work_peak_prediction_error_m=actual['peak_error_m']-choice['peak_m'],
                             work_hold_prediction_error_m=actual['final_hold_peak_m']-choice['hold_m'])
            branch_records.append(entry)
    branch_outcomes=pd.DataFrame(branch_records);branch_outcomes.to_csv(root/'source_actual_branch_outcomes.csv',index=False)
    actions=pd.DataFrame([dict(case_id=r['case_id'],context=r['context'],would_release_after_B0=r['would_release_after_B0'],base_id=s['base_id'],speed=s['speed'],admitted=s['admitted'],reason=s['reason'],
        torque_margin_Nm=float(min(np.array([87]*4+[12]*3)-np.asarray(s['prediction']['torque_Nm']))) if 'prediction' in s else np.nan,
        clearance_margin_m=s.get('prediction',{}).get('clearance_m',np.nan)) for r in branch for s in r['screens']]);actions.to_csv(root/'source_motion_admissibility.csv',index=False)
    abl=load_results(root,'source_ablation')
    candidates=paired[(paired.treatment=='adaptive_work_decision')&(paired.control=='plan_after_b0')].sort_values(['physical_seconds_difference','case_id','noise_seed'])
    selected=[]
    for label,table in [('largest observed saving',candidates),('largest observed loss',candidates.iloc[::-1]),('identical schedule',candidates)]:
        row=next(x for x in table.itertuples() if (x.case_id,int(x.noise_seed)) not in [(s['case_id'],s['noise_seed']) for s in selected])
        delta=float(row.physical_seconds_difference)
        description=label+f': adaptive minus plan-once = {delta:g} s'
        if delta==0: description+=' (tie; no saving/loss demonstrated)'
        runs={m:next(r for r in rows if r['case_id']==row.case_id and r['noise_seed']==row.noise_seed and r['method']==m) for m in ('plan_after_b0','adaptive_work_decision')}
        selected.append(dict(case_id=row.case_id,noise_seed=int(row.noise_seed),description=description,runs=runs))
    write_json(root/'real_timeline_examples.json',selected)
    figures(root,df,paired,source,pd.read_csv(root/'source_prediction_validation.csv'),pd.DataFrame(prediction_rows),selected,abl)
    selection=json.loads((root/'configs/validation_selection.json').read_text());totals=json.loads((root/'resources/totals.json').read_text())
    old_objects=int(ref.groupby('object_id').initial_normal_pass.all().sum());normal_objects=int(ref.groupby('object_id').true_normal_pass.all().sum())
    actual_choices={m:{a:sum(r['selectors'][m].get('chosen')==a for r in branch) for a in sorted({r['selectors'][m].get('chosen') for r in branch},key=str)} for m in ('fixed_prior_plan','plan_after_b0','adaptive_work_decision')}
    rank_rows=[]; concrete=[]
    for r in branch:
        high_files=list((root/'source_high_precision'/r['case_id']).glob('*.json'))
        scores=[json.loads(p.read_text())['score'] for p in high_files] if high_files else [x for x in r['rank_check'] if 'score' in x]
        high=None
        if scores:
            high=(min(scores,key=lambda x:(-x['score'],x['base_id']))['base_id'] if max(x['score'] for x in scores)>0 and max(x['score'] for x in scores)-min(x['score'] for x in scores)>1e-7 else r['frozen_reference_suffix'][0])
        actual=r['selectors']['adaptive_work_decision'].get('chosen')
        rank_rows.append(dict(case_id=r['case_id'],context=r['context'],choice_128_16=actual,choice_256_32=high,agrees=actual==high,available=high is not None))
        if actual!=r['selectors']['plan_after_b0'].get('chosen'):
            with np.load(root/'raw/branches'/r['case_id']/'prefix/public.npz') as z:
                ts=z['time']-z['time'][0]; torque=z['torque'];home=np.median(torque[(ts>=1)&(ts<2)],axis=0);tilt=np.median(torque[(ts>=4)&(ts<5)],axis=0)
            observed=r['post_B0_estimate'];ablation=next(x for x in abl if x['case_id']==r['case_id'])
            concrete.append(dict(case_id=r['case_id'],context=r['context'],home_torque_median_Nm=home.tolist(),tilted_torque_median_Nm=tilt.tolist(),
                estimated_mass_kg=observed['payload_estimate']['mass'],estimated_com_m=observed['payload_estimate']['com'],
                chosen=actual,once_chosen=r['selectors']['plan_after_b0']['chosen'],choice_256_32=high,
                scores=[x for x in r['rankings'] if x['method']=='adaptive_work_decision'],mean_cov_interventions=ablation['variants'],
                already_eligible_to_release=r['would_release_after_B0']))
    pd.DataFrame(rank_rows).to_csv(root/'source_MC_rank_stability.csv',index=False)
    write_json(root/'concrete_measurement_choice_examples.json',concrete)
    main=summary.set_index('method');ad=main.loc['adaptive_work_decision'];once=main.loc['plan_after_b0']
    text=f'''# V2 实测结果

完整测试为 64 个不同对象 × 3 个公开工作站 × 3 个噪声种子，每种方法 576 次，六种方法共 3456 次；另有 576 组真实参数参考。验证为全部 2304 次条件组合。未按结果删除对象、失败或中止。

## 首先看真实工作能力

旧设置在 {int(ref.initial_normal_pass.sum())}/576 个参考运行中通过；{old_objects}/64 个对象在全部工作站与种子上均通过。真实设置正常速度通过 {int(ref.true_normal_pass.sum())}/576；{normal_objects}/64 个对象在全部环境与种子上均通过。分类见 work_reference_categories.csv；半速和正常速度分列，不把未放行当作完成。

## 主比较

Rechoose for Work Decisions 正常速度正确放行 {int(ad.normal_correct)}/576，Plan Once After Initial Measurements 为 {int(once.normal_correct)}/576。误放行分别为 {int(ad.false_release)} 和 {int(once.false_release)}；接触、关节、力矩约束违规运行分别为 {int(ad.physical_violations)} 和 {int(once.physical_violations)}。

平均识别物理时间分别为 {ad.identification_s:.6f} s 和 {once.identification_s:.6f} s，验证时间分别为 {ad.verification_s:.6f} s 和 {once.verification_s:.6f} s。实测串行时间分别为 {ad.serial_s:.6f} s 和 {once.serial_s:.6f} s。两者 {int(pairs.identical.sum())}/576 对的运动顺序与速度完全相同。

统计先平均三个种子，再按载荷家族、家族内对象分层配对重采样 2000 次，三个工作站始终随对象一起移动。差值区间及单侧边界见 paired_method_comparisons.csv。全相同结果可能给出退化的零宽自助区间；64 个对象和零错误不构成 1% 总体风险保证。主要结论按实际物理时间差判断，不能用数值很小的缓存计时差替代机器人时间收益。

## 测量是否改变选择

预定 24 个源历史中，{sum(r['would_release_after_B0'] for r in branch)}/24 个在 B0 后已经满足工作预测。为研究机制，仍离线强制继续，完成 {len(branch_outcomes)} 个匹配噪声分支。下一动作计数：{json.dumps(actual_choices,ensure_ascii=False)}。

因此，模型内部出现不同选择，与部署中需要继续测量、以及实际节省时间是三件不同的事。source_action_rankings.csv 保存原分数；source_motion_admissibility.csv 保存同一公共筛选器的速度、力矩和间隙余量；source_actual_branch_outcomes.csv 同时保存即时参数变化和最终正确放行时间。source_ablation 保持候选运动集合及本轮部署配置集合不变，分别冻结均值或协方差，仅用于解释模型分数，不当作实际收益。

## 公共筛选器与误差来源

源对象的 0..5 号用于拟合，6..7 号用于校准；三个上下文不跨对象分区。保留 V1 的 17 维联合参数/持久偏置估计。公开模型以冻结的 100 Hz 控制器、500 Hz 物理步长积分，并每 50 ms 更新离散动力学/控制导数；误差传播每 2 ms 进行。预测的名义跟踪误差没有被置零。预测模型没有未知对象指针，公开二次检查单独记账。

共同膨胀倍率选择为 {selection['multiplier']}，固定模板为 {selection['fixed_template']}，经验资格检查结果 {selection['qualified']}。全部三个模板、全部倍率均保留。这个近似线性模型及有限源校准不是安全认证。工作准确度阈值也用于描述识别运动的误差，但识别期接触/关节/力矩违规另列；不能将工作容差的描述性超限隐藏成通过。

六种方法使用相同更新、可行性、速度、停止和一次实际验证规则。最有利/不利/相同顺序示例按固定规则选择；若没有正负时间差，图中明确标为平局。没有失败例时不会制造失败轨迹。

## 计算、缓存与工程记录

仅使用 2072 和 2379。初始最多 8 个进程完成 20 个源运动块测速，随后使用 136+44 个工作进程；未训练 GPU 模型。当前对象仿真账本 {totals['plant']['completed']:.3f} s，公开预测积分 {totals['prediction']['completed']:.3f} s，已记账工作进程 CPU {totals['cpu']['completed']/3600:.3f} 核时。封包后最终资源数见 resources/totals.json。

公开确定性预测按完整公开输入、模型和代码版本精确缓存，所有方法共用同一规则。串行 CPU 数字是实际缓存状态下测量，不是统一冷启动延迟；每次缓存命中另有记录。未把任何预测记录加入真实测量历史。原始大数组在外部存储，ZIP 中的索引只提供位置、大小和 SHA256，明确不是原始数组本身。

早期 .75 倍速采样舍入引发两次记账失败：8/.75 秒不是 100 Hz 的整数样本周期。现有实现先向上对齐采样周期并收费；失败尝试和费用保留，部分失败尾块在异常前尚未保存原始数组，ACCOUNTING_FIX/SECOND_ATTEMPT 文件明确记录该限制。源版本修订补充工作运动预测校准、即时分支参数误差，以及固定计划超时不改变公共 B0 的检查；不修改控制器、载荷范围或测试验收规则。
'''
    if concrete:
        c=concrete[0];intervention=next(x for x in c['mean_cov_interventions'] if x['variant']=='freeze_mean_to_prior')
        text+=f'''\n## 一个确切的测量与选择实例\n\n源历史 `{c['case_id']}`（{c['context']}）：HOME 静止段七关节测得力矩中位数为 {np.round(c['home_torque_median_Nm'],4).tolist()} Nm；倾斜静止段为 {np.round(c['tilted_torque_median_Nm'],4).tolist()} Nm。B0 后估计质量为 {c['estimated_mass_kg']:.6f} kg，质心为 {np.round(c['estimated_com_m'],6).tolist()} m。重复决策模型选择 {c['chosen']}，一次规划选择 {c['once_chosen']}。仅把本轮均值换回先验、保持协方差、可行运动和部署配置集合不变时，模型选择 {intervention.get('selected')}。256/32 采样检查选择 {c['choice_256_32']}；详细分数及蒙特卡洛变异见 concrete_measurement_choice_examples.json。这里没有隔离某一个传感器通道的因果贡献，而且该历史在 B0 后本来就可放行，所以它不能作为多测一次更快的证据。\n'''
    text+=f'\n识别阶段的精度单列：每种方法有 {int(ad.identification_tracking_failures)}/576 次 B0 未达到最终工作的严格跟踪/终端保持容差；这不是接触、关节或力矩违规，也不是任务已放行。全部方法用 B0 后的共同更新和实际验证规则决定工作放行，不能把“无安全约束违规”写成“所有识别轨迹都达到了工作精度”。\n'
    text+='\n计时修订：首次完整验证的固定先验决策搜索有 288 次超过 10 s 上限，最大约 0.310 s；随后增加批次内检查和 20 ms 收尾余量，在测试封存前重新执行了完整 2304 次验证。旧结果位于 engineering_runtime_revision_02，未混入正式矩阵。\n'
    text+='\n256/32 源排序检查最初受到 10 s 在线计时器限制而未完成。补充检查将相同随机样本下各候选分开并行计算，使用源历史完成更高采样数的离线诊断，记录在 source_high_precision。它在主实验封存后补完，没有读取测试结果来改变分数、参数、控制器或选择规则；在线 10 s 上限保持不变。\n'
    text+='\n间隙字段说明：夹具距离在 1 m 截断；W0 没有夹具，其 clearance=1 是“无近距离夹具”的数值哨兵，不是测得恰好 1 m。夹具间隙散点图因此仅画 W1/W2。地面和自接触仍由实际物理接触/中止标志单独判断。\n'
    (root/'FINDINGS_CN.md').write_text(text)
    (root/'REPRODUCE.md').write_text('''# Reproduce

From experiments/commissioning_generalization, use the pinned Python environment
with PYTHONPATH=.:src, PYTHONDONTWRITEBYTECODE=1 and one BLAS thread. Generated data
stay outside Git. The shared parent artifact and pinned Panda assets must exist
at the paths in payload_adaptive_tests_v2/protocol.py and the inherited V1 backend.

```bash
for stage in audit qualify-source fit branch-checks validate seal evaluate report package; do
  python -m payload_adaptive_tests_v2.runner --stage "$stage" --run-root RUN_ROOT || exit 1
done
```

Heavy workers are dispatched only to leiyo@ssh.cloud.sdu.dk ports 2072 and 2379.
Initial profiling uses 8 workers. The current allocation supports 136+44 workers;
reduce these if the scheduler allocation changes. Review resources/profile.json,
stage forecasts and immutable protocol_locked.json. Do not overwrite an existing
seal to hide changed science. A new root regenerates fresh public/private manifests
from the same declared seeds and parent family partition. Validation consumes all
three inflation choices and all three fixed templates. Test code/config hashes
are verified before and after dispatch. Failure and exit-code records are retained.

The source case ZIP is not all raw data: raw_artifact_index.jsonl.gz contains
absolute references and SHA256, and representative_traces/manifest.json lists the
included actual 100 Hz public/500 Hz private arrays at their original relative
raw/ paths. Exact command logs and complete per-run JSON
include all outcomes. Figure CSV and standalone PGFPlots/TikZ sources accompany
PDF/PNG. Build a .tex file from within figures/ using a TeX distribution with
standalone and pgfplots. No physical robot or external API is used.
''')
    write_json(root/'checks/result_matrix.json',dict(passed=True,main_runs=len(rows),reference_cases=len(references),objects=64,contexts=3,seeds=3,
        methods=list(METHODS),same_schedule_pairs=int(pairs.identical.sum()),source_branch_histories=24,source_actual_branches=len(branch_outcomes)))
    return dict(main_runs=3456,reference_cases=576,summary=summary.to_dict('records'),reference_categories=ref.category.value_counts().to_dict())
