"""Small public-rule arithmetic/selection tests; no simulator or fitted model."""
from types import SimpleNamespace
import unittest
import numpy as np
import pandas as pd
import torch
from .reader_candidates import candidate_tensors,select_source_candidate


class ReaderCandidateTests(unittest.TestCase):
    def state(self):
        nodes=torch.zeros(2,32);nodes[1,31]=1;nodes[0,26]=.4
        seq=torch.tensor([[1.]*64,[3.]*64,[5.]*64,[11.]*64])
        return SimpleNamespace(local=torch.stack([seq.mean(0),torch.zeros(64)]),batch={'nodes':nodes},
            sequence_vectors=seq,node_lookup={'n0':0,'n1':1},
            sequence_metadata=[dict(port=port,record_count=n,missing=missing) for port,n,missing in
                [('n0:in',100,True),('n0:in',200,False),('n0:in',300,True),('n0:out',1000,False)]])

    def test_port_balancing_preserves_all_real_readings(self):
        state=self.state();local,nodes=candidate_tensors(state,'R-P')
        torch.testing.assert_close(local[0],torch.full((64,),7.))
        torch.testing.assert_close(state.local[0],torch.full((64,),5.))
        self.assertTrue(torch.equal(nodes,state.batch['nodes']))

    def test_count_rule_scaling_missing_fraction_and_original_formula(self):
        state=self.state();local,nodes=candidate_tensors(state,'R-C')
        np.testing.assert_allclose(nodes[0,23:26],np.log1p([2,1.2,2/3]),rtol=1e-6)
        self.assertEqual(float(nodes[0,31]),0.)
        self.assertEqual(float(nodes[1,31]),1.)
        self.assertEqual(float(nodes[0,26]),float(state.batch['nodes'][0,26]))
        self.assertTrue(torch.equal(local,state.local))
        state.sequence_metadata=state.sequence_metadata[:1];state.sequence_vectors=state.sequence_vectors[:1]
        _,nodes=candidate_tensors(state,'R-C')
        self.assertAlmostEqual(float(nodes[0,31]),1-np.log(2),places=7)

    def frame(self):
        return pd.DataFrame([dict(source_purpose='source_validation',configuration_id=f'c{cfg}',
            case_id=f'c{cfg}_{healthy}',is_healthy=healthy,variant=variant,
            true_target_nll_bits=loss,false_alarm=0,missed_fault=0,joint_fault_correct=int(not healthy),
            processing_seconds=cost)
            for cfg in (0,1) for healthy in (False,True)
            for variant,loss,cost in [('original',1.,0),('R-P',.8,.1),('R-C',.9,.05),('R-PC',.797,.02)]])

    def test_single_rule_preferred_within_declared_nll_tie(self):
        selected,_=select_source_candidate(self.frame())
        self.assertEqual(selected,'R-P')

    def test_guardrail_rejects_more_missed_faults(self):
        frame=self.frame();frame.loc[frame.variant.isin(['R-P','R-PC'])&~frame.is_healthy,'missed_fault']=1
        selected,_=select_source_candidate(frame)
        self.assertEqual(selected,'R-C')

    def test_v1_or_fresh_cannot_be_selection_material(self):
        frame=self.frame();frame.loc[0,'source_purpose']='V1_confirmation'
        with self.assertRaises(ValueError):select_source_candidate(frame)

    def test_nonimproving_reader_not_forced(self):
        frame=self.frame();frame.loc[frame.variant!='original','true_target_nll_bits']=1.1
        selected,_=select_source_candidate(frame)
        self.assertIsNone(selected)


if __name__=='__main__':unittest.main()
