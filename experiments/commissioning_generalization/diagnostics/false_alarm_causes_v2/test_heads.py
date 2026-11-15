"""Small contract tests; no checkpoint fit, optimizer step, or physical rollout."""
import unittest
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd

from .heads import CELLS, matched_schedule, outcome, summarize
from .head_data import calibration_tables


class HeadContracts(unittest.TestCase):
    def test_schedule_uses_train_support_and_is_matched(self):
        rows=[]
        for config,lengths in [('c0',[1,9]),('c1',[5,2])]:
            for case,length in enumerate(lengths):
                rows.extend(dict(configuration_id=config,case_id=f'{config}_{case}',source_split='train',eligible_for_fit=True)
                            for _ in range(length))
        rows.extend(dict(configuration_id='validation',case_id='v',source_split='validation',eligible_for_fit=True) for _ in range(50))
        frame=pd.DataFrame(rows);a=matched_schedule(frame,11,updates=100,batch_size=128)
        b=matched_schedule(frame,11,updates=100,batch_size=128)
        np.testing.assert_array_equal(a,b)
        selected=frame.iloc[a.reshape(-1)]
        self.assertTrue(selected.source_split.eq('train').all())
        self.assertLess(abs(selected.configuration_id.eq('c0').mean()-.5),.025)
        self.assertLess(abs(selected.case_id.eq('c0_0').mean()-.25),.025)
        self.assertEqual(CELLS,{'T00':('oof','oof'),'T01':('final','oof'),'T10':('oof','final'),'T11':('final','final')})

    def test_false_alarm_is_legal_argmax_not_half_threshold(self):
        row=dict(case_id='c',configuration_id='g',method_key='gnn_adaptive',policy_seed=11,
                 history_hash='h',decision_index=1,true_target_key='healthy',
                 prediction=dict(target_keys=['healthy','n0:cache_stale','n1:queue_hold'],
                                 log_probabilities_base2=np.log2([.4,.35,.25]).tolist()),
                 initial_budget=dict(bytes=100,decisions=6),remaining=dict(bytes=70,decisions=5),
                 executions_used=1,calibrations_used=0)
        result=outcome(row,'S1')
        self.assertEqual(result['healthy_false_alarm'],0.)
        self.assertEqual(result['bytes_used'],30)
        self.assertEqual(result['decisions_used'],1)
        self.assertIsNone(result['physical_wall_seconds'])

    def test_summary_conditions_joint_fault_accuracy(self):
        rows=[]
        for healthy,correct in [(True,True),(False,False),(False,True)]:
            rows.append(dict(configuration_id='g',case_id=str(len(rows)),method_key='gnn_adaptive',
                arm='T11',cell='T11',step=2000,is_healthy=healthy,healthy_false_alarm=0. if healthy else None,
                fault_miss=None if healthy else 0.,correct_joint=correct,final_nll_bits=1.,
                executions_used=1,calibrations_used=0,bytes_used=10))
        result=summarize(pd.DataFrame(rows)).iloc[0]
        self.assertEqual(result.correct_joint,.5)
        self.assertEqual(result.healthy_denominator,1)
        self.assertEqual(result.faulty_denominator,2)
        self.assertEqual(result.joint_correct_faults,1)

    def test_calibration_collapses_prefixes_before_seeds(self):
        rows=[]
        for seed,length,value in [(11,1,1.),(23,3,-1.)]:
            rows.extend(dict(configuration_id='g',case_id='c',method_key='gnn_adaptive',policy_seed=seed,
                source_split='validation',fault_family='healthy',is_healthy=True,executions_used=0,
                actionable=True,eligible_for_fit=True,chosen_stop=False,resource_or_error_terminal=False,
                terminal_reason='budget_exhausted',original_q=1.,original_q_with_oof_input=.5,
                return_oof=value,return_final=value) for _ in range(length))
        with tempfile.TemporaryDirectory() as directory:
            calibration_tables(pd.DataFrame(rows),Path(directory))
            result=pd.read_csv(Path(directory)/'continuing_value_calibration.csv')
        selected=result[(result.grouping=='method_key') & ~result.positive_q_only]
        np.testing.assert_allclose(selected.realized_return,0.)
        np.testing.assert_allclose(selected.realized_return_ci_low,0.)
        np.testing.assert_allclose(selected.realized_return_ci_high,0.)


if __name__=='__main__':unittest.main()
