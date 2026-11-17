"""Registered numerical choices, provenance, and independent resource accounts."""
from __future__ import annotations
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import uuid
import zipfile
from payload_change_v1.storage import canonical, digest, file_hash, write_json

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
REPO = EXP.parent.parent
PARENT = Path('/work/Users/leiyo/cgx_data_v2/payload_change_v1/20261116T180000Z')
PARENT_ZIP = PARENT.parent/'Payload_Change_Commissioning_20261116T180000Z_GPT_Pro.zip'
METHODS = {
    'b0_verify_only': 'Initial Test Only',
    'fixed_prior_plan': 'Prior-Designed Fixed Order',
    'plan_after_b0': 'Plan Once After Initial Measurements',
    'simple_parameter_rule': 'Hand-Written Parameter Rule',
    'adaptive_variance': 'Rechoose for Work-Torque Variance',
    'adaptive_work_decision': 'Rechoose for Work Decisions',
}
LIMITS = {'plant': 2_000_000., 'prediction': 1_000_000., 'cpu': 160*3600.}
REQUIREMENTS = dict(rms_m=.005, peak_m=.010, hold_m=.002, hold_s=.5,
    saturation_fraction=.01, saturation_continuous_s=.1, reference_velocity=1.5,
    reference_acceleration=3., nominal_fixture_clearance_m=.030)
REGISTERED = dict(experiment='CGX-PAYLOAD-ADAPTIVE-TESTS-V2', methods=METHODS,
    source_seed=5, validation_seed=7, test_seeds=[11,23,37], objects_per_family=8,
    contexts=['W0','W1','W2'], split_families=[12,4,8],
    source_fit_rule='object index 0..5 fit; index 6..7 held out for margins/coverage',
    identification_cap_s=60., verification_cap_s=20., additional_blocks=6,
    verification_attempts=1, audit_jobs=8, decision_seconds=10.,
    online_compute_cap_s=80., capped_serial_s=160.,
    parameter_samples=128, outcomes=16, rank_check_samples=256, rank_check_outcomes=32,
    beam_width=3, beam_depth=4, inflation_multipliers=[1,2,4],
    bootstrap_replicates=2000, bootstrap_seed=913928,
    limits=LIMITS, requirements=REQUIREMENTS,
    allowed_ssh_ports=[2072,2379], initial_workers=8,
    cpu_threads_per_worker=1, GPU_training=False,
    source_collection='B0 then all 10 base patterns in hash-permuted order, cyclic speeds 1/.75/.5; only public kinematic rejection; unsafe observed prefixes retained',
    prediction_grid_hz=100, sensitivity_refresh_hz=20, physics_hz=500,
    prediction_linear_region='no saturation or slew limiting; parameter scaled distance <= source heldout maximum',
    secondary_hypotheses=17, secondary_rule='mean plus 16 projected deterministic Sobol joint-parameter directions',
    work_variation='independent uniform +/-0.01 rad per joint/waypoint, deterministic public geometry rejection; no payload/outcome input',
    fallback='source-prior mixed diverse order; no variance substitution for decision entropy ties',
    notes='All confidence sets and screens are experimental approximations, not certified safety bounds.')

class BudgetExceeded(RuntimeError):
    pass

class Ledger:
    """Reserve before computation; unfinished jobs keep their reserved charge."""
    def __init__(self, root, account='plant'):
        self.root=Path(root)/'resources'; self.root.mkdir(parents=True,exist_ok=True)
        self.account=account
    @contextlib.contextmanager
    def locked(self):
        with (self.root/'ledger.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            path=self.root/'totals.json'
            state=json.loads(path.read_text()) if path.exists() else {
                k:dict(charged=0.,completed=0.,pending={},cap=v) for k,v in LIMITS.items()}
            yield state
            write_json(path,state)
    def event(self, row):
        with (self.root/'ledger.jsonl').open('a') as out:
            out.write(canonical(row|{'wall_timestamp':time.time(),'pid':os.getpid()})+'\n')
    def reserve(self, seconds, unit='', kind=''):
        token=uuid.uuid4().hex
        with self.locked() as state:
            a=state[self.account]
            if a['charged']+seconds>a['cap']+1e-6:
                raise BudgetExceeded(f'{self.account} ceiling: {a["charged"]}+{seconds}>{a["cap"]}')
            rec=dict(seconds=float(seconds),unit=str(unit),kind=str(kind))
            a['pending'][token]=rec; a['charged']+=seconds
            self.event(dict(event='reserve',account=self.account,token=token,**rec))
        return token
    def complete(self, token, actual_seconds, status='completed'):
        with self.locked() as state:
            a=state[self.account]; rec=a['pending'][token]
            if actual_seconds<0 or actual_seconds>rec['seconds']+1e-5:
                raise ValueError('actual exceeds reservation')
            del a['pending'][token]
            a['charged']+=actual_seconds-rec['seconds']; a['completed']+=actual_seconds
            self.event(dict(event='complete',account=self.account,token=token,
                            actual_seconds=float(actual_seconds),status=status,**rec))
    def cache_hit(self,key,seconds,scope):
        with self.locked():
            self.event(dict(event='exact_cache_hit',key=key,reused_seconds=seconds,scope=scope))

def scientific_hashes():
    files=list(HERE.glob('*.py'))+list((EXP/'payload_change_v1').glob('*.py'))
    return {str(p.relative_to(EXP)):file_hash(p) for p in files
            if p.name not in ('reporting.py','packaging.py')}

def audit(root):
    root=Path(root); root.mkdir(parents=True,exist_ok=True)
    archive=zipfile.ZipFile(PARENT_ZIP)
    rows=[]
    for p in sorted((EXP/'payload_change_v1').rglob('*')):
        if p.is_file() and '__pycache__' not in str(p):
            name='payload_change_v1/code/'+str(p.relative_to(EXP))
            archived=hashlib.sha256(archive.read(name)).hexdigest() if name in archive.namelist() else None
            rows.append(dict(path=str(p),local_sha256=file_hash(p),archive_sha256=archived,
                             match=archived==file_hash(p) if archived else None))
    core=('design.py','estimator.py','physics.py','trajectories.py','commissioning.py','cases.py')
    assert all(r['match'] for r in rows if Path(r['path']).name in core)
    for name in ('protocol_locked.yaml','CONTROLLER_DECISION.md','results/summary.csv',
                 'configs/controller.json','configs/estimator_parameters.json'):
        target=root/'parent_audit'/name; target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(archive.read('payload_change_v1/'+name))
    tracked=subprocess.check_output(['git','ls-files','-z'],cwd=REPO).decode().split('\0')
    protected={p:file_hash(REPO/p) for p in tracked if p and (REPO/p).is_file()
               and '/payload_adaptive_tests_v2/' not in p}
    write_json(root/'checks/protected_hashes.json',protected)
    write_json(root/'parent_audit/file_versions.json',rows)
    write_json(root/'registered.json',REGISTERED)
    (root/'PARENT_AUDIT.md').write_text('''# Verified parent\n
Parent code commit: 75fd606a4374a3101302a28c9231d293e1fa66e2. All six scientific
modules match the delivered archive byte for byte; complete comparison is in
parent_audit/file_versions.json. The archive itself has SHA256 '''+file_hash(PARENT_ZIP)+'''.

`design.build_design_bank` computes 17-variable information from public reference
grids. `estimator.PayloadEvidenceEstimator` jointly fits ten payload coordinates
and seven persistent biases. `design.choose_action` switches from entropy to
torque variance below .01 bit. Conditional Gaussian covariance does not depend
on observation values. `design.assess` uses .05 impact, .95 demand and a fitted
torque interval cutoff; none is a position-error prediction.

`physics.PayloadPlant` compiles the true and configured models separately. Torque
is qfrc_actuator plus persistent bias and AR noise, sampled before the new control
command. Event identity includes execution and stream identity; exact duplicates
are deduplicated by the estimator. `trajectories.original_task` retains 100 ms
transitions; `trajectories.workload` is smooth. The old initial-motion violations
are distinct from the passing final jobs. The archived controller decision and
results/summary.csv are copied alongside this audit.

The three multi-test methods each released 576/576, with 10.125 s identification,
10 s verification and 38 updates. The keep-settings reference also passed all
576 jobs. V2 replaces L0 by B0 and freezes new 5/10/2 mm requirements, three public
contexts and outcome-dependent common screening. These changes are not credited
to adaptive selection. V1 files and raw data are read-only throughout V2.
''')
    return dict(passed=True,core_archive_match=True,parent_zip_sha256=file_hash(PARENT_ZIP))

def verify_protected(root):
    expected=json.loads((Path(root)/'checks/protected_hashes.json').read_text())
    bad=[p for p,h in expected.items() if not (REPO/p).exists() or file_hash(REPO/p)!=h]
    result=dict(passed=not bad,count=len(expected),changed=bad)
    write_json(Path(root)/'checks/parent_preservation.json',result)
    if bad: raise RuntimeError('Protected files changed: '+str(bad))
    return result
