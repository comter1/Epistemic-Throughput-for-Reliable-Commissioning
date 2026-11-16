"""Engineering checks for isolation, acquisition identity and replay-first gates."""
import ast
import copy
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from cgx.config import load_config
from cgx.env.system import ExecutionResult
from cgx.evaluation.design import matrix_cells
from cgx.provenance import atomic_json, file_sha256, read_json
from .policy_training import require_training_gate
from .rollout import CachedSystem
from .selectors import ReadOnlyReplay, source_history_provenance, reader_fingerprint, ImmutableScoreLookup


class SelectorTests(unittest.TestCase):
    def test_replay_source_is_sqlite_immutable(self):
        with tempfile.TemporaryDirectory() as temp:
            source=Path(temp)/'source';run=Path(temp)/'new';(source/'training').mkdir(parents=True)
            path=source/'training/replay_train.sqlite';db=sqlite3.connect(path)
            db.executescript("CREATE TABLE meta(key TEXT,value TEXT); INSERT INTO meta VALUES('source_hash','sealed'); CREATE TABLE cases(case_id TEXT,n INTEGER,fold INTEGER,configuration TEXT); CREATE TABLE plan_support(case_id TEXT,sample_idx INTEGER,transition_idx INTEGER); INSERT INTO cases VALUES('case',1,0,'config');")
            db.commit();db.close();before=file_sha256(path);inventory=list(source.rglob('*'))
            replay=ReadOnlyReplay(source,run)
            with self.assertRaises(sqlite3.OperationalError):replay.db.execute("DELETE FROM cases")
            self.assertEqual(replay.case_rows,[('case',1,0,'config')])
            self.assertEqual(before,file_sha256(path));self.assertEqual(inventory,list(source.rglob('*')))
            self.assertFalse((source/'training/replay_train.lock').exists())

    def test_stop_preserves_identity_and_new_measurement_changes_it(self):
        common=dict(parent_case_id='opaque-case',trajectory_id='trajectory',coverage_branch=None)
        rows=[dict(common,step=0,history_hash='a',next_history_hash='b',action={'kind':'trace'}),
              dict(common,step=1,history_hash='b',next_history_hash='b',action={'kind':'stop'})]
        ids=source_history_provenance(rows)
        self.assertEqual(len(ids['a']['observation_ids']),1)
        self.assertEqual(len(ids['b']['observation_ids']),2)
        self.assertEqual(ids['a']['observation_ids'][0],ids['b']['observation_ids'][0])
        self.assertNotEqual(*ids['b']['observation_ids'])
        # Numerical payload equality is deliberately irrelevant to event IDs.
        another=copy.deepcopy(rows);another[0]['trajectory_id']='fresh';another[1]['trajectory_id']='fresh'
        fresh=source_history_provenance(another)
        self.assertNotEqual(fresh['b']['observation_ids'][1],ids['b']['observation_ids'][1])

    def test_exact_physical_cache_and_durable_attempts(self):
        with tempfile.TemporaryDirectory() as temp:
            system=CachedSystem.__new__(CachedSystem);system.cache_root=Path(temp);system.base_identity='opaque'
            system.cache_hits=system.robot_attempts=system.calibration_attempts=0
            system.physical_seconds=system.actual_physical_seconds=0.;system.last_acquisition_id=None
            calls=[]
            def physical():
                calls.append(1);return ExecutionResult({'records':[1], 'actual_cost':{'bytes':123}},b'complete-wire',{'metric':7})
            first=system._execute('acquire',{'execution_index':1},physical,1,0);first_id=system.last_acquisition_id
            second=system._execute('acquire',{'execution_index':1},physical,1,0)
            self.assertEqual(first.observation,second.observation);self.assertEqual(first.metrics,second.metrics)
            self.assertEqual(system.last_acquisition_id,first_id);self.assertEqual(len(calls),1)
            self.assertEqual(system.robot_attempts,1);self.assertEqual(system.cache_hits,1)
            system._execute('acquire',{'execution_index':2},physical,1,0)
            self.assertNotEqual(system.last_acquisition_id,first_id);self.assertEqual(len(calls),2)
            journal=[read_json(p) for p in Path(temp).rglob('*.attempts.json')]
            self.assertEqual(len(journal),2);self.assertTrue(all(j['attempts'][0]['state']=='passed' for j in journal))
            def broken():raise ValueError('physical failure')
            with self.assertRaises(ValueError):system._execute('acquire',{'execution_index':3},broken,1,0)
            failed=[read_json(p) for p in Path(temp).rglob('*.attempts.json') if read_json(p)['attempts'][0]['state']=='failed']
            self.assertEqual(len(failed),1);self.assertEqual(failed[0]['attempts'][0]['execution_attempts'],1)

    def test_reader_replay_gate_checks_completion_content(self):
        with tempfile.TemporaryDirectory() as temp:
            run=Path(temp)
            with self.assertRaises(FileNotFoundError):require_training_gate(run)
            completed=run/'evaluation/replay_complete.json';atomic_json(completed,{'state':'passed'})
            gate=dict(state='authorized_after_reader_replay',reader_replay_complete_path=str(completed),
                reader_replay_complete_sha256=file_sha256(completed))
            atomic_json(run/'manifests/policy_training_authorized.json',gate)
            self.assertEqual(require_training_gate(run),gate)
            atomic_json(completed,{'state':'passed','altered':True})
            with self.assertRaises(AssertionError):require_training_gate(run)

    def test_optimizer_update_ast_matches_registered_trainer(self):
        # Every operation inside the original optimizer/validation loop remains
        # unchanged; adapters alter only evidence and frozen model loading.
        directory=Path(__file__).parent
        def update(path):
            tree=ast.parse(path.read_text());fit=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='fit_policy')
            loop=next(n for n in fit.body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='iteration')
            return ast.dump(loop,include_attributes=False)
        self.assertEqual(update(directory/'policy_training.py'),update(directory.parent/'src/cgx/training/engine.py'))

    def test_full_registered_matrix(self):
        config=load_config();cells=matrix_cells(config)
        cases=sum(s['configurations']*s['cases_per_configuration'] for n,s in config['splits'].items() if n.startswith('test_'))
        self.assertEqual(cases,3200);self.assertEqual(cases*len(cells),147200)
        self.assertEqual(len(cells),len(set(cells)))
        self.assertEqual({cap for m,s,cap,b in cells if m=='gnn_adaptive'},{1,2,4})

    def test_frozen_parameter_fingerprint_detects_mutation(self):
        class Reader:parameters_dict={'variance':1.};checkpoint_hash='frozen-file'
        reader=Reader();before=reader_fingerprint(reader);reader.parameters_dict={'variance':2.}
        self.assertNotEqual(before,reader_fingerprint(reader))

    def test_immutable_score_cache_preserves_floats_feature_outputs_and_rng(self):
        import numpy as np
        import struct
        db=sqlite3.connect(':memory:');db.executescript('CREATE TABLE predictions(history TEXT,fold INTEGER,data TEXT); CREATE TABLE scores(id TEXT,reward REAL,next_nll REAL);')
        lp=[float(-np.log2(3.)),float(np.nextafter(-2.,-3.)),float(-0.)]
        prediction=dict(target_keys=['healthy','n0:cache_stale','n1:queue_delay'],log_probabilities_base2=lp,audit_only={'never_a_policy_feature':True})
        db.execute('INSERT INTO predictions VALUES(?,?,?)',('ref',2,json.dumps(prediction)))
        db.execute('INSERT INTO scores VALUES(?,?,?)',('transition',np.nextafter(.25,.3),lp[1]))
        rng=np.random.default_rng(23);before=copy.deepcopy(rng.bit_generator.state)
        cache=ImmutableScoreLookup(db)
        self.assertEqual(before,rng.bit_generator.state)
        original=json.loads(db.execute('SELECT data FROM predictions').fetchone()[0]);actual=cache.prediction('ref',2)
        self.assertEqual(original['target_keys'],actual['target_keys'])
        for a,b in zip(original['log_probabilities_base2'],actual['log_probabilities_base2']):self.assertEqual(struct.pack('d',a),struct.pack('d',b))
        self.assertTrue(np.array_equal(np.exp2(original['log_probabilities_base2']),np.exp2(actual['log_probabilities_base2'])))
        for a,b in zip(db.execute('SELECT reward,next_nll FROM scores').fetchone(),cache.score('transition')):self.assertEqual(struct.pack('d',a),struct.pack('d',b))
        # The preserved update AST then receives identical float arrays in the
        # original order; cache reads cannot consume training-generator draws.
        self.assertEqual(before,rng.bit_generator.state)
        with self.assertRaises(RuntimeError):cache.prediction('missing',2)

    def test_outer_failure_after_acquisition_retains_unknown_cost(self):
        from unittest.mock import patch
        from . import evaluation
        import numpy as np
        class Store:
            def graph(self,configuration):return {'public':'fixture'}
            def task(self,case):return {'nominal':'fixture'}
        case=dict(case_id='case',configuration_id='cfg',split='test_seen',fault_family='healthy')
        performed=[]
        def late_failure(*args,**kwargs):
            performed.append('physical acquisition completed')
            raise OSError('final evidence store write failed')
        context=dict(store=Store(),config={},reader=None,policy=None,configurations={'cfg':{'implementation':'A'}},source_root=Path('/source'),run_root=Path('/new'))
        with patch.object(evaluation,'_WORKER',context),patch.object(evaluation,'rollout',late_failure),patch.object(evaluation,'legal_targets',return_value=(['healthy','n0:cache_stale'],np.array([.5,.5]))):
            row,trace=evaluation._job((case,4,1.))
        self.assertEqual(performed,['physical acquisition completed'])
        self.assertTrue(row['worker_failure'] and row['abstained'] and row['resource_accounting_unavailable'])
        self.assertEqual(row['case_id'],'case');self.assertEqual(row['prediction']['target_keys'],['healthy','n0:cache_stale'])
        for key in ('bytes_used','executions_used','calibrations_used','decisions_used','selection_seconds','inference_seconds','simulation_seconds','feasible','new_robot_execution_attempts'):
            self.assertIsNone(row[key],key)


if __name__=='__main__':unittest.main()
