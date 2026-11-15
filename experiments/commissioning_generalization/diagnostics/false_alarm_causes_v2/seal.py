"""Freeze all source-selected choices before any fresh confirmation acquisition."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
from cgx.provenance import atomic_json, file_sha256, read_json
from cgx.config import EXPERIMENT
from .ledger import Ledger


def run(run_dir, enable_live=False):
    run=Path(run_dir);destination=run/'manifests/source_variant_selection.json'
    if destination.exists():raise RuntimeError('Selection already sealed; do not overwrite it')
    assert not list((run/'trajectories/confirmation/cases').glob('*.json.gz'))
    fresh=read_json(run/'manifests/fresh_cases.json')
    assert all(not (run/'cache/initial'/f"{c['case_id']}.json.gz").exists() for c in fresh)
    source=read_json(run/'trajectories/source/complete.json')
    assert source['state']=='passed' and source['trajectories_present']==1920
    assert read_json(run/'head_data_complete.json')['state']=='passed'
    assert read_json(run/'head_replay_source_complete.json')['state']=='passed'
    heads=read_json(run/'head_selection.json')
    reader=read_json(run/'manifests/source_reader_selection.json')
    assert reader['state']=='passed' and reader['source_validation_only']
    assert not reader['reads_fresh_outcomes'] and not heads['heldout_outcomes_read']
    ledger=Ledger(run).snapshot();counts=ledger['counts']
    assert len(ledger['fits'])==24 and all(f['updates']==2000 for f in ledger['fits'].values())
    original=read_json(run/'source_checks.json')
    for path,sha in original['manifest_hashes'].items():assert file_sha256(run/path)==sha
    for choice in heads['methods'].values():
        assert choice['step'] in (500,1000,1500,2000)
        for saved in choice['checkpoints'].values():assert file_sha256(saved['path'])==saved['sha256']
    combined=sum(counts[k] for k in ('model_forward','feature_build','reader_partial_forward'))
    # All initial runs are shared physically across the six policy trajectories.
    required_robot=128+128*6*4
    required_cal=128*6*2
    # Up to seven prefixes per trajectory: raw feature + original reader + selector;
    # extra final-only reader calls are conservatively reserved at two per arm/prefix.
    required_calls=128*6*7*3+128*6*2*2
    live_robot=128*6*4 if enable_live else 0
    live_cal=128*6*2 if enable_live else 0
    live_calls=128*6*7*4 if enable_live else 0
    if enable_live:assert reader['selected_variant'] is not None
    assert counts['robot_execution']+required_robot+live_robot<=Ledger.LIMITS['robot_execution']
    assert counts['calibration']+required_cal+live_cal<=Ledger.LIMITS['calibration']
    assert combined+required_calls+live_calls<=Ledger.LIMITS['model_and_features']
    frozen_paths=['head_selection.json','manifests/source_reader_selection.json','manifests/protocol_clarifications.json',
        'manifests/reader_selection_interpretation.json',
        'source_prefixes.parquet','continuing_value_source_per_prefix.parquet','reader_candidate_validation.csv',
        'correction_2x2_source_validation.csv','manifests/fresh_cases.json','manifests/fresh_tasks.json']
    selected=dict(task='CGX-FA-CAUSE-V2',state='sealed_before_fresh_outcomes',sealed_at=datetime.now(timezone.utc).isoformat(),
        fresh_outcomes_exist_at_seal=False,fresh_initial_acquisitions_exist_at_seal=False,
        selected_reader_variant=reader['selected_variant'],reader_available=reader['selected_variant'] is not None,
        selected_heads=heads['methods'],fixed_gate_threshold=0.,main_arms=['A0','S1','G1']+
            (['R1','RG1'] if reader['selected_variant'] is not None else []),
        all_four_cells_reported_at_updates=2000,all_four_deployment_q_inputs='FINAL',
        fresh_population='64 healthy and64 faulty new physical cases; 16 already examined held-out configurations; 3policy seeds',
        optional_live=dict(enabled=bool(enable_live),reader_variant=reader['selected_variant'],
            arms=['R-live','RG-live'] if enable_live else [],
            reason='source-qualified substantive reader and conservative resource headroom' if enable_live else
                'not enabled before opening confirmation; no post-outcome optional selection',
            gate_caveat='source head is calibrated for original reader returns; live R* deployment is an explicit input/return mismatch test'),
        optional_noise_branches=dict(enabled=False,reason='required source and sealed confirmation prioritized'),
        conservative_remaining_resources=dict(robot_executions=required_robot+live_robot,
            calibration_attempts=required_cal+live_cal,combined_model_features=required_calls+live_calls),
        ledger_at_seal=counts,frozen_files={p:file_sha256(run/p) for p in frozen_paths},
        frozen_scientific_code={name:file_sha256(EXPERIMENT/'diagnostics/false_alarm_causes_v2'/name)
            for name in ('hooks.py','reader_candidates.py','heads.py','head_data.py','rollouts.py','live.py','confirmation.py','fresh.py','ledger.py')},
        all_head_checkpoints={str(p.relative_to(run)):file_sha256(p) for p in sorted((run/'head_checkpoints').rglob('*.pt'))})
    atomic_json(destination,selected)
    print({k:selected[k] for k in ('state','selected_reader_variant','main_arms','optional_live')},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);p.add_argument('--enable-live',action='store_true')
    a=p.parse_args();run(a.run_dir,a.enable_live)
