import json
from pathlib import Path
import pandas as pd
import pytest
from cgx.reporting.results import validate_result_matrix,paired_bootstrap,write_result_hashes
from cgx.provenance import file_sha256


def fixture():
    config={'methods':{'learned':['gnn_adaptive','transformer_adaptive'],'nonlearned':['default_only']},'seeds':{'training':[11,23]},'evaluation':{'main_execution_cap':4},'resources':{'byte_fractions':[0.,.25]}}
    cases=[{'case_id':'a','configuration_id':'x','split':'test_connections'},{'case_id':'b','configuration_id':'y','split':'test_connections'}]
    methods=[(m,s) for m in config['methods']['learned']+['cost_aware_adaptation'] for s in [11,23]]+[('default_only',None)]
    rows=[dict(method=m,policy_seed=s,byte_fraction=f,execution_cap=4,case_id=c['case_id'],configuration_id=c['configuration_id'],suite=c['split']) for m,s in methods for f in [0.,.25] for c in cases]
    return pd.DataFrame(rows),cases,config


def test_E05_matrix_catches_duplicated_case_despite_matching_count():
    frame,cases,config=fixture();assert validate_result_matrix(frame,cases,config)
    frame.iloc[1]=frame.iloc[0]
    with pytest.raises(ValueError,match='Duplicated'):validate_result_matrix(frame,cases,config)


def test_E05_missing_budget_or_seed_is_not_a_zero_result():
    frame,cases,config=fixture();frame=frame[~((frame.method=='gnn_adaptive')&(frame.policy_seed==23)&(frame.byte_fraction==.25))]
    with pytest.raises(ValueError,match='matrix differs'):validate_result_matrix(frame,cases,config)


def test_E04_bootstrap_weights_systems_not_number_of_case_rows():
    rows=[]
    for cfg,count,difference in [('small',1,2.),('large',50,0.)]:
        for case in range(count):
            rows.extend([dict(configuration_id=cfg,case_id=str(case),method='a',final_nll_bits=difference),dict(configuration_id=cfg,case_id=str(case),method='b',final_nll_bits=0.)])
    result=paired_bootstrap(pd.DataFrame(rows),'a','b',200,19)
    assert result['difference_nll_bits']==1. and result['configurations']==2 and result['cases']==51


def test_E11_figure_hashes_refresh_without_self_reference(tmp_path):
    output=tmp_path/'results';(output/'figures').mkdir(parents=True)
    table=output/'table.csv';table.write_text('actual,measurement\n1,2\n')
    first=write_result_hashes(tmp_path)
    figure=output/'figures'/'F1.pdf';figure.write_bytes(b'actual-figure-fixture')
    second=write_result_hashes(tmp_path)
    assert 'results/figures/F1.pdf' in second and 'results/hashes.json' not in second
    assert second['results/figures/F1.pdf']==file_sha256(figure)
    assert len(second)==len(first)+1


def test_E02_figure_deliverables_and_provenance_on_explicit_test_fixture(tmp_path,monkeypatch):
    from cgx.reporting.results import figures
    from cgx.provenance import atomic_json
    output=tmp_path/'results';output.mkdir()
    rows=[]
    methods=['gnn_adaptive','transformer_adaptive','gnn_fixed_plan','full_recording_unrestricted_reference']
    for suite in ['test_seen','test_connections']:
        for method in methods:
            for fraction in [0.,.25,1.]:
                rows.append(dict(suite=suite,method=method,policy_seed=11,byte_fraction=fraction,execution_cap=4,configuration_id='fixture_system',final_nll_bits=2.,correct_joint=.5,bytes_used=fraction*1000000,executions_used=fraction*4,selection_seconds=.1,inference_seconds=.2))
    frame=pd.DataFrame(rows);frame.to_csv(output/'per_configuration.csv',index=False);frame.drop(columns='configuration_id').to_csv(output/'per_seed.csv',index=False)
    pd.DataFrame([dict(suite='test_connections',method='gnn_adaptive',policy_seed=11,coverage=.5,risk=.1,mean_total_deployed_executions=5.,yield_fraction=.45)]).to_csv(output/'repair_summary.csv',index=False)
    atomic_json(tmp_path/'models/thresholds.json',dict(curves={'gnn_adaptive:11':[dict(threshold=.5,coverage=.7,risk=.2),dict(threshold=.9,coverage=.2,risk=0.)]}))
    atomic_json(tmp_path/'models/readers/final/complete.json',dict(state='passed',technical_smoke=True,actual_updates=2,parameter_count=100,elapsed_seconds=1.,peak_gpu_memory_bytes=1024,selected_checkpoint_hash='fixture'))
    independent=frame.drop(columns=['configuration_id']).rename(columns={'final_nll_bits':'final_nll_bits_primary'});independent['final_nll_bits_independent']=2.1;independent.to_csv(output/'independent_reader_per_seed.csv',index=False)
    import matplotlib.figure
    labels=[];save=matplotlib.figure.Figure.savefig
    def inspect_and_save(self,*args,**kwargs):
        labels.extend(ax.get_ylabel() for ax in self.axes)
        return save(self,*args,**kwargs)
    monkeypatch.setattr(matplotlib.figure.Figure,'savefig',inspect_and_save)
    result=figures(tmp_path,{})
    assert any('NLL' in label for label in labels)
    assert all('exact ET' not in label and 'mutual information' not in label for label in labels)
    expected=['F1_actual_bytes_nll_accuracy','F1_budget_nll','F2_all_systems','F5_validation_curve_heldout_points','F6_training_resources','F7_independent_reader']
    for name in expected:
        assert (output/'figures'/f'{name}.png').is_file()
        assert (output/'figures'/f'{name}.pdf').is_file()
    hashes=json.loads((output/'hashes.json').read_text())
    assert all(f'results/figures/{name}.pdf' in hashes for name in expected)
    assert result['state']=='passed'


def test_E12_release_crc_marker_only_after_actual_archive(tmp_path):
    from cgx.reporting.results import export
    from cgx.provenance import atomic_json
    import zipfile
    for name in ['results/hashes.json','protocol_lock.json','environment.json','models/frozen.json','models/selection.json','models/thresholds.json']:
        atomic_json(tmp_path/name,{'technical_smoke_fixture':True})
    (tmp_path/'software').mkdir();(tmp_path/'software/source.tar.gz').write_bytes(b'fixture-pinned-source')
    output=tmp_path/'release/result.zip';result=export(tmp_path,output)
    marker=json.loads((tmp_path/'release/complete.json').read_text())
    assert marker['sha256']==file_sha256(output) and result['archive_crc_checked']
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None and 'software/source.tar.gz' in archive.namelist()


def test_E10_evaluator_strata_keep_healthy_abstentions_and_small_impact_faults(tmp_path):
    from cgx.reporting.diagnostics import enrich_evaluator_results,write_diagnostic_tables
    cases=[dict(case_id='h0',configuration_id='a',fault_family='healthy',fault_node=None,severity=None),
           dict(case_id='h1',configuration_id='a',fault_family='healthy',fault_node=None,severity=None),
           dict(case_id='f0',configuration_id='b',fault_family='adapter_scale',fault_node='adapter',severity=-.05)]
    graphs={'a':{'nodes':[{'type':'cache'}]*8},'b':{'nodes':[{'type':'adapter'}]*24}}
    rows=[]
    for case,predicted,nll,abstained in zip(cases,['healthy','n:cache_stale','healthy'],[1.,2.,3.],[False,True,False]):
        rows.append(dict(case_id=case['case_id'],configuration_id=case['configuration_id'],suite='test_connections',method='gnn_adaptive',policy_seed=11,
            byte_fraction=.25,execution_cap=4,true_target_key='healthy' if case['fault_family']=='healthy' else 'adapter:adapter_scale',
            predicted_target_key=predicted,final_nll_bits=nll,bytes_used=100,executions_used=1,calibrations_used=0,
            correct_joint=predicted==('healthy' if case['fault_family']=='healthy' else 'adapter:adapter_scale'),
            timeout=False,abstained=abstained,initial_task_pass=None if case['case_id']=='h1' else True,initial_hold_error_m=.001,feasible=True))
    frame=enrich_evaluator_results(pd.DataFrame(rows),cases,graphs)
    assert pd.isna(frame.loc[frame.case_id.eq('h0'),'severity']).all()
    fault=frame[frame.case_id.eq('f0')].iloc[0]
    assert fault.severity==-.05 and fault.severity_magnitude==.05 and fault.component_count==24 and fault.initial_task_impact=='passed'
    target=dict(source='validation',rule='selected_final_reader_mean_validation_nll',target_nll_bits=2.1)
    write_diagnostic_tables(tmp_path,frame,target)
    healthy=pd.read_csv(tmp_path/'results/healthy_false_alarm_per_seed.csv')
    assert healthy.healthy_cases.iloc[0]==2 and healthy.healthy_false_alarm_rate.iloc[0]==.5
    assert healthy.healthy_abstention_rate.iloc[0]==.5
    impact=pd.read_csv(tmp_path/'results/fault_impact_strata.csv')
    assert impact.cases.sum()==3 and 'unknown' in set(impact.initial_task_impact)
    points=pd.read_csv(tmp_path/'results/cost_quality_recorded_points.csv')
    assert points.final_nll_bits.iloc[0]==2.25 # systems equally weighted, not2.0 case-pooled
    crossed=pd.read_csv(tmp_path/'results/cost_to_quality_per_seed.csv')
    assert crossed.status.iloc[0]=='not_reached' and pd.isna(crossed.actual_bytes.iloc[0])
    confusion=pd.read_csv(tmp_path/'results/confusion_by_mechanism.csv')
    assert 'count_queue_hold' in confusion and confusion.count_queue_hold.sum()==0


def test_quality_crossing_uses_only_first_recorded_passing_budget():
    from cgx.reporting.diagnostics import cost_to_quality
    points=pd.DataFrame([dict(suite='test_seen',method='gnn_adaptive',policy_seed=11,execution_cap=4,byte_fraction=f,
        final_nll_bits=loss,bytes_used=used,executions_used=1,calibrations_used=0,all_cases_feasible=True)
        for f,loss,used in [(0.,3.,0),(.25,1.9,250),(.5,1.0,500)]])
    result=cost_to_quality(points,dict(source='validation',rule='selected_final_reader_mean_validation_nll',target_nll_bits=2.)).iloc[0]
    assert result.status=='reached' and result.byte_fraction==.25 and result.actual_bytes==250
    assert not result.interpolated and not result.extrapolated
    with pytest.raises(ValueError,match='source validation'):
        cost_to_quality(points,dict(source='test_connections',rule='selected_final_reader_mean_validation_nll',target_nll_bits=2.))


def test_quality_target_traces_to_selected_validation_checkpoint(tmp_path):
    from cgx.reporting.diagnostics import validation_quality_target
    from cgx.provenance import atomic_json
    root=tmp_path/'models/readers/final';root.mkdir(parents=True);checkpoint=root/'checkpoint_0005000.pt';checkpoint.write_bytes(b'technical-unit-fixture')
    atomic_json(root/'complete.json',dict(state='passed',technical_smoke=False,selected_checkpoint=checkpoint.name,selected_checkpoint_hash=file_sha256(checkpoint)))
    atomic_json(root/'validation.json',[dict(iteration=5000,mean_final_nll_bits=2.),dict(iteration=10000,mean_final_nll_bits=2.5)])
    target=validation_quality_target(tmp_path)
    assert target['source']=='validation' and target['target_nll_bits']==2. and target['selected_iteration']==5000
    assert target['reader_checkpoint_hash']==file_sha256(checkpoint)
    checkpoint.write_bytes(b'changed')
    with pytest.raises(ValueError,match='hash mismatch'):validation_quality_target(tmp_path)
