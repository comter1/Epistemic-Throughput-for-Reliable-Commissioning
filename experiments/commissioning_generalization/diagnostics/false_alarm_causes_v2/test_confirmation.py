"""Scientific boundary checks for exact-prefix stopping; no model/simulator use."""
import unittest
from pathlib import Path
import tempfile
import json

from .confirmation import first_execution_prefix, gated_prefix


def prefix(ordinal, executions, kind='single_port', actionable=True):
    return dict(prefix_ordinal=ordinal, executions_used=executions, actionable=actionable,
                history_hash=f'actual_history_{ordinal}',
                chosen_action=dict(kind=kind) if actionable else None)


class PrefixStoppingTests(unittest.TestCase):
    def test_calibration_does_not_trigger_first_execution_stop(self):
        trajectory = [prefix(0, 0, 'calibration_only'), prefix(1, 0), prefix(2, 1), prefix(3, 2)]
        self.assertIs(first_execution_prefix(trajectory), trajectory[2])

    def test_original_stop_before_first_execution_is_retained(self):
        trajectory = [prefix(0, 0, 'calibration_only'), prefix(1, 0, 'stop')]
        self.assertIs(first_execution_prefix(trajectory), trajectory[-1])
        self.assertIs(gated_prefix(trajectory, {0: 1.}), trajectory[-1])

    def test_gate_stops_at_first_nonpositive_and_never_reads_future(self):
        trajectory = [prefix(0, 0), prefix(1, 1), prefix(2, 2), prefix(3, 3)]
        self.assertIs(gated_prefix(trajectory, {0: .2, 1: 0.}), trajectory[1])
        self.assertIs(gated_prefix(trajectory, {0: -.1}), trajectory[0])

    def test_final_original_terminal_retained_when_all_continue(self):
        trajectory = [prefix(0, 0), prefix(1, 1), prefix(2, 2, actionable=False)]
        self.assertIs(gated_prefix(trajectory, {0: .2, 1: .1}), trajectory[2])

    def test_unavailable_truth_cannot_choose_a_best_prefix(self):
        class PublicPrefix(dict):
            def __getitem__(self, name):
                if name in ('truth', 'true_target_key', 'correct_joint', 'final_nll_bits'):
                    raise AssertionError('An evaluator label entered stopping')
                return super().__getitem__(name)
        trajectory = [PublicPrefix(prefix(0, 0)), PublicPrefix(prefix(1, 1)), PublicPrefix(prefix(2, 2))]
        self.assertIs(gated_prefix(trajectory, {0: .2, 1: -.01}), trajectory[1])
        self.assertIs(first_execution_prefix(trajectory), trajectory[1])

    def test_invalid_prediction_is_not_treated_as_continue(self):
        with self.assertRaises(FloatingPointError):
            gated_prefix([prefix(0, 0)], {0: float('nan')})

    def test_live_gate_uses_actual_changed_reader_route(self):
        from .live import gate_prefix
        trajectory = [prefix(0, 0), prefix(1, 1), prefix(2, 2)]
        trajectory[0]['selected_gate_score'] = .1
        trajectory[1]['selected_gate_score'] = -.2
        # A future item has no score: an accidental post-stop read would fail.
        self.assertIs(gate_prefix(trajectory), trajectory[1])

    def test_fresh_cannot_start_without_source_seal(self):
        from .fresh import run
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                run('/unused-root', directory)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_optional_live_cannot_start_when_not_predeclared(self):
        from .live import run
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / 'manifests/source_variant_selection.json'
            manifest.parent.mkdir()
            manifest.write_text(json.dumps(dict(state='sealed_before_fresh_outcomes', optional_live={'enabled': False})))
            with self.assertRaises(AssertionError):
                run('/unused-root', directory)
            self.assertFalse((Path(directory) / 'cache').exists())


if __name__ == '__main__':
    unittest.main()
