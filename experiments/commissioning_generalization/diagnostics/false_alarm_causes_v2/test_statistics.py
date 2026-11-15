"""Synthetic-only tests of paired statistical units, with no model execution."""
import unittest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
from .statistics import hierarchical, estimate, two_by_two


class StatisticsTests(unittest.TestCase):
    def rows(self):
        return pd.DataFrame([dict(configuration_id=f'g{g}',case_id=f'g{g}c{c}',
            method_key='gnn_adaptive',policy_seed=s,is_healthy=c<2,
            baseline=float(c<2),intervention=float(c<2)+.125)
            for g in range(3) for c in range(4) for s in (11,23,37)])

    def test_pairing_and_strata_remain_exact(self):
        point,draws,cases=hierarchical(self.rows(),['baseline','intervention'])
        np.testing.assert_array_equal(point,[.5,.625])
        np.testing.assert_array_equal(draws,np.tile([.5,.625],(2000,1)))
        self.assertEqual(len(cases),12)

    def test_repeated_seed_rows_do_not_create_cases(self):
        frame=self.rows()
        a=hierarchical(frame,['baseline','intervention'])
        b=hierarchical(pd.concat([frame,frame]),['baseline','intervention'])
        np.testing.assert_array_equal(a[1],b[1])
        self.assertEqual(len(a[2]),len(b[2]))

    def test_missing_values_fail_instead_of_dropping(self):
        frame=self.rows();frame.loc[0,'intervention']=np.nan
        with self.assertRaises(AssertionError):hierarchical(frame,['baseline','intervention'])

    def test_interval_sign_and_denominator(self):
        result=estimate(self.rows(),'final_nll_bits','test','healthy')[0]
        self.assertEqual(result['physical_cases'],6)
        self.assertEqual(result['paired_seed_runs'],18)
        self.assertEqual(result['difference'],.125)
        self.assertEqual(result['paired_ci_low'],.125)
        self.assertEqual(result['paired_ci_high'],.125)

    def test_boolean_two_by_two_and_conditional_missing_cells(self):
        # Three treatment cells are correct, baseline is incorrect. Correctness
        # must add numerically (not NumPy boolean OR), giving main effects +.5
        # and interaction -1. Healthy-only metrics remain NA for faulty rows.
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            for phase,n in [('source_validation',64),('fresh_replay',128)]:
                rows=[]
                for case in range(n):
                    healthy=case%4==0
                    for method in ('gnn_adaptive','gnn_depth2'):
                        for seed in (11,23,37):
                            for cell in ('T00','T01','T10','T11'):
                                correct=cell!='T00'
                                rows.append(dict(configuration_id=f'g{case//8}',case_id=f'c{case}',
                                    method_key=method,policy_seed=seed,is_healthy=healthy,arm=cell,step=2000,
                                    healthy_false_alarm=float(not correct) if healthy else None,
                                    fault_miss=float(not correct) if not healthy else None,correct_joint=correct,
                                    final_nll_bits=1.-float(correct),executions_used=int(correct),
                                    calibrations_used=0,decisions_used=int(correct),bytes_used=100*int(correct)))
                pd.DataFrame(rows).to_parquet(root/f'correction_2x2_{phase}_per_case.parquet',index=False)
            two_by_two(root)
            result=pd.read_csv(root/'correction_2x2_paired_effects.csv')
            joint=result[result.metric=='correct_joint']
            for effect in ('mean_FINAL_label-minus-OOF_label','mean_FINAL_q_training-minus-OOF_q_training'):
                np.testing.assert_allclose(joint[joint.contrast==effect].difference,.5)
            np.testing.assert_allclose(joint[joint.contrast=='(T11-T01)-(T10-T00)'].difference,-1.)


if __name__=='__main__':unittest.main()
