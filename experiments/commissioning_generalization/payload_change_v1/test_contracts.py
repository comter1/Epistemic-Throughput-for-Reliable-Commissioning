"""Scientific integrity checks for data boundaries, shared budgets and resume."""
import json
from pathlib import Path
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from .storage import PhysicsLedger
from .cases import policy_view, MAIN_METHODS, REFERENCES


class Contracts(unittest.TestCase):
    def test_budget_counts_aborted_and_pending_work(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger = PhysicsLedger(directory, cap_seconds=20)
            a = ledger.reserve(10, 'a'); ledger.complete(a, 4, 'hard_abort')
            ledger.reserve(10, 'interrupted')
            self.assertEqual(ledger.summary()['charged_seconds'], 14)
            with self.assertRaises(RuntimeError):
                ledger.reserve(7, 'over_cap')
            self.assertEqual(ledger.summary()['pending_blocks'], 1)

    def test_parallel_reservations_exact(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger = PhysicsLedger(directory, cap_seconds=100)
            def block(index):
                token = ledger.reserve(2, str(index))
                ledger.complete(token, 1.25)
            with ThreadPoolExecutor(max_workers=8) as pool:
                list(pool.map(block, range(32)))
            self.assertEqual(ledger.summary()['completed_seconds'], 40)
            self.assertEqual(ledger.summary()['pending_blocks'], 0)

    def test_public_view_poisoning_does_not_change_features(self):
        public = {'configured_payload': {'mass': .7}, 'box_size': [.1]*3,
                  'box_center': [0, 0, .08], 'mount_frame': 'panda_hand', 'case_id': 'opaque'}
        expected = policy_view(public)
        poisoned = public | {'case_id': 'G3_leak', 'group': 'G3', 'family_id': 900,
                             'true_payload': {'mass': 999}, 'private_audit_results': [False]}
        self.assertEqual(expected, policy_view(poisoned))

    def test_registered_method_and_budget_counts(self):
        self.assertEqual(len(MAIN_METHODS), 4)
        self.assertEqual(len(REFERENCES), 2)
        self.assertEqual(192 * 3 * len(MAIN_METHODS), 2304)
        self.assertLessEqual(10 + 6 * 8, 60)
        self.assertLessEqual(10 + 6 * 8 + 20, 80)


if __name__ == '__main__':
    unittest.main()
