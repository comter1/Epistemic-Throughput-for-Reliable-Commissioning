"""由用户自行运行：python -m unittest -v test_benchmark.py。"""
import itertools
import unittest
import numpy as np
from benchmark import (Config, Model, Case, MEASURES, evaluate_policy,
                       dp_optimal_gain, exact_policy_gain)


class ModelTests(unittest.TestCase):
    def test_prior_and_joint_likelihood(self):
        action = ('Os', 'Oq', 'Ds', 'Dq')
        for mode in ('independent', 'shared_round'):
            model = Model(Config(noise_mode=mode))
            self.assertEqual(len(model.states), 225)
            np.testing.assert_allclose(model.y_posterior(model.prior), [.25]*4)
            outcomes, likelihood = model.bundle_likelihood(action)
            np.testing.assert_allclose(likelihood.sum(axis=0), 1)
            # 独立构造一个固定 theta 的噪声结果字典，复核整个联合分布。
            idx = 100
            expected = {}
            noises = itertools.product(range(3), repeat=4) if mode == 'independent' else [(i,)*4 for i in range(3)]
            for noise_indices in noises:
                obs = tuple(int(model.base[a][idx]+model.noise_values[j]) for a, j in zip(action, noise_indices))
                prob = (float(np.prod([model.noise_probs[j] for j in noise_indices]))
                        if mode == 'independent' else float(model.noise_probs[noise_indices[0]]))
                expected[obs] = expected.get(obs, 0) + prob
            actual = {tuple(o): p for o, p in zip(outcomes, likelihood[:, idx]) if p > 0}
            self.assertEqual(set(expected), set(actual))
            for obs in expected:
                self.assertAlmostEqual(expected[obs], actual[obs])

    def test_shared_noise_cancellation_and_fresh_rounds(self):
        model = Model(Config(noise_mode='shared_round'))
        case = Case(100, int(model.labels[100]), 0, 1)
        rng = np.random.default_rng(7)
        samples = [model.observe(case, ('Os', 'Ds'), rng) for _ in range(100)]
        self.assertTrue(all(os-ds == model.states[100, 0] for os, ds in samples))
        self.assertGreater(len(set(ds for _, ds in samples)), 1)
        # 完整 bundle 同轮消除两个偏移和共同噪声，类别可完全恢复。
        action = ('Os', 'Oq', 'Ds', 'Dq')
        for x0 in range(-1, 6):
            posterior = model.initial_posterior(x0)
            self.assertAlmostEqual(model.expected_gain(posterior, action), model.entropy_y(posterior))

    def test_costs_and_budget(self):
        for cost in (1, 2, 4):
            model = Model(Config(calibration_cost=cost))
            action = ('Os', 'Oq', 'Ds', 'Dq')
            self.assertEqual(model.action_cost(action), 2+2*cost)
            rng = np.random.default_rng(9)
            cases = [model.draw_case(rng) for _ in range(12)]
            rows = []
            result = evaluate_policy(model, cases, 'full_bundle', 2+2*cost, [action], rows)
            self.assertEqual(result['mean_cost'], 2+2*cost)
            self.assertTrue(all(r['rounds'] == 1 for r in rows))
            stop = evaluate_policy(model, cases, 'stop', 0, model.action_library())
            self.assertEqual(stop['information_gain_bits'], 0)
            for budget in range(3):
                result = evaluate_policy(model, cases, 'greedy', budget, model.action_library(True, 1))
                self.assertLessEqual(result['mean_cost'], budget)

    def test_dp_same_library(self):
        model = Model(Config())
        actions = [(m,) for m in MEASURES]
        p = model.initial_posterior(2)
        for rounds in (1, 2):
            optimal, _ = dp_optimal_gain(model, p, 2, rounds, actions)
            greedy = exact_policy_gain(model, p, 2, rounds, actions)
            self.assertGreaterEqual(optimal+1e-10, greedy)
            if rounds == 1:
                self.assertAlmostEqual(optimal, greedy)

    def test_calibration_alone_initially_uninformative(self):
        model = Model(Config())
        for x0 in range(-1, 6):
            p = model.initial_posterior(x0)
            self.assertAlmostEqual(model.expected_gain(p, ('Ds',)), 0)
            self.assertAlmostEqual(model.expected_gain(p, ('Dq',)), 0)


if __name__ == '__main__':
    unittest.main()
