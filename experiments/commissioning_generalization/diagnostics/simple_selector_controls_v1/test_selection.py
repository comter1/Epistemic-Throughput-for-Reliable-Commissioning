"""Pure selection fixtures: no simulator, reader, or scientific outcomes."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from cgx.provenance import file_sha256, hash_json, atomic_json, now
from . import selection as s


def fixtures():
    cases = [dict(case_id=f'{i:064x}', configuration_id=f'fixture_{i // 64}',
                  fault_family=s.FAMILIES[(i % 64) // 16], split='validation') for i in range(512)]
    splits = dict(validation=[c['case_id'] for c in cases], test_connections=['unopened_test_case_id'])
    rows = []
    for case in cases:
        for cap in s.CAPS:
            for candidate in s.CANDIDATES:
                template = candidate in s.TEMPLATES
                key = hash_json([case['case_id'], cap, candidate])
                rows.append(dict(case_id=case['case_id'], configuration_id=case['configuration_id'],
                    fault_family=case['fault_family'], suite='validation', execution_cap=cap,
                    candidate_id=candidate, method='fixed_template_validation' if template else 'interface_rule_se',
                    template_id=candidate if template else None,
                    stop_threshold=None if template else s.STOPS[candidate], policy_seed=None,
                    cell_key=key, reader_hash='r' * 64, protocol_hash='c' * 64, final_nll_bits=2.,
                    bytes_used=100, executions_used=1, calibrations_used=0,
                    correct_joint=False, healthy_false_alarm=False, fault_miss=False, timeout=False,
                    abstained=False, worker_failure=False, resource_accounting_unavailable=False,
                    result_mode='online', rule_hash=hash_json(candidate), evidence_trace_hash=key,
                    event_provenance_hash=key, execution_source_key=key))
    online = pd.DataFrame(rows); completed = online.copy()
    completed['result_mode'] = 'completed'; completed['completion_status'] = 'reused_verified_online'
    return online, completed, cases, splits


class SelectionControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.online, cls.completed, cls.cases, cls.splits = fixtures()
        cls.summary, cls.configuration_summary = s.summarize_candidates(cls.online, cls.completed)

    def validate(self, online=None, completed=None, splits=None):
        return s.validate_matrix(self.online if online is None else online,
                                 self.completed if completed is None else completed,
                                 self.cases, self.splits if splits is None else splits, 'r' * 64, 'c' * 64)

    def test_complete_denominators_and_corruption(self):
        self.assertEqual(self.validate()['rows_per_mode'], 16896)
        self.assertEqual(len(self.summary), 33)
        self.assertEqual(len(self.configuration_summary), 264)
        for bad in (self.online.iloc[:-1], pd.concat([self.online.iloc[:-1], self.online.iloc[:1]])):
            with self.assertRaises(ValueError):
                self.validate(online=bad)
        bad = self.online.copy(); bad.loc[0, 'case_id'] = 'foreign_case'
        with self.assertRaises(ValueError):
            self.validate(online=bad)
        bad = self.completed.copy(); bad.loc[0, 'evidence_trace_hash'] = 'other_evidence'
        with self.assertRaises(ValueError):
            self.validate(completed=bad)
        bad = self.online.copy(); bad.loc[0, 'policy_seed'] = 11
        with self.assertRaises(ValueError):
            self.validate(online=bad)
        bad = deepcopy(self.splits); bad['test_connections'].append(self.cases[0]['case_id'])
        with self.assertRaises(ValueError):
            self.validate(splits=bad)

    def test_exact_lexicographic_order_and_completed_independence(self):
        settings, _ = s.choose_settings(self.summary)
        self.assertEqual(settings['selected_template_by_cap'], {'1': 'F0', '2': 'F0', '4': 'F0'})
        self.assertEqual(settings['selected_stop_threshold_by_cap'], {'1': None, '2': None, '4': None})
        for index, column in enumerate(s.RANK_COLUMNS):
            summary = self.summary.copy()
            for field in s.RANK_COLUMNS:
                summary[field] = 10.
            target = summary.candidate_id.eq('F7')
            summary.loc[target, column] = 1.
            for field in s.RANK_COLUMNS[index + 1:]:
                summary.loc[target, field] = 10000.
            # Completed-record winner deliberately disagrees with deployed winner.
            summary['mean_completed_final_nll_bits'] = 0.
            summary.loc[target, 'mean_completed_final_nll_bits'] = 999.
            chosen, _ = s.choose_settings(summary)
            self.assertEqual(set(chosen['selected_template_by_cap'].values()), {'F7'})
        summary = self.summary.copy()
        summary.loc[summary.candidate_id.eq('rule_0.999'), 'mean_deployed_final_nll_bits'] = 1.
        chosen, _ = s.choose_settings(summary)
        self.assertEqual(set(chosen['selected_stop_threshold_by_cap'].values()), {.999})

    def test_unknown_resources_keep_failed_rows_and_rank_explicitly(self):
        online, completed = self.online.copy(), self.completed.copy()
        for frame in (online, completed):
            frame.loc[0, ['bytes_used', 'executions_used', 'calibrations_used']] = np.nan
            frame.loc[0, ['worker_failure', 'resource_accounting_unavailable', 'timeout', 'abstained']] = True
        checks = self.validate(online, completed)
        self.assertTrue(checks['all_failed_timeout_abstained_rows_retained'])
        summary, _ = s.summarize_candidates(online, completed)
        chosen, _ = s.choose_settings(summary)
        self.assertEqual(chosen['selected_template_by_cap']['1'], 'F1')
        summary.loc[(summary.candidate_id == 'F0') & (summary.execution_cap == 1), 'mean_deployed_final_nll_bits'] = 1.
        chosen, _ = s.choose_settings(summary)
        self.assertEqual(chosen['selected_template_by_cap']['1'], 'F0')
        winner = next(r for r in chosen['selected_candidate_summaries'] if r['candidate_id'] == 'F0' and r['execution_cap'] == 1)
        self.assertIsNone(winner['mean_bytes_used'])
        json.dumps(chosen, allow_nan=False)
        online.loc[0, 'final_nll_bits'] = np.nan
        with self.assertRaises(ValueError):
            self.validate(online, completed)

    def test_configuration_weighting(self):
        frame = pd.DataFrame(dict(configuration_id=['a'] * 9 + ['b'], value=[0.] * 9 + [10.]))
        self.assertEqual(s.configuration_balanced_mean(frame, 'value'), 5.)
        self.assertNotEqual(s.configuration_balanced_mean(frame, 'value'), frame.value.mean())

    def test_seal_files_and_traceability_without_outcome_access(self):
        # Fake tables exercise artifact sealing only; no fixture is a research result.
        with tempfile.TemporaryDirectory(prefix='simple-selection-control-') as temporary:
            run = Path(temporary); directory = run / 'validation'; directory.mkdir()
            (run / 'provenance').mkdir()
            self.online.to_parquet(directory / s.SOURCE_OUTPUTS[0], index=False)
            self.completed.to_parquet(directory / s.SOURCE_OUTPUTS[1], index=False)
            index = pd.DataFrame(dict(cell_key=self.online.cell_key,
                cell_file_sha256=['a' * 64] * len(self.online),
                cell_path=[str(directory / 'cases' / (key + '.json')) for key in self.online.cell_key]))
            index.to_parquet(directory / s.SOURCE_OUTPUTS[2], index=False)
            (directory / s.SOURCE_OUTPUTS[3]).write_bytes(b'engineering-fixture-no-acquisitions')
            protocol = dict(source_root='fixture_only', reader_hash='r' * 64, config_hash='c' * 64)
            marker = dict(runtime_identity_hash='i' * 64,
                outputs_sha256={name: file_sha256(directory / name) for name in s.SOURCE_OUTPUTS})
            atomic_json(run / 'protocol.json', protocol); atomic_json(run / 'source_inventory.json', {})
            atomic_json(run / 'provenance/validation_design_seal.json', {})
            atomic_json(run / 'provenance/split_case_ids.json', self.splits)
            atomic_json(directory / 'phase_validation_complete.json', marker)
            with patch.object(s, '_prerequisites', return_value=(protocol, {}, {}, marker)), \
                 patch.object(s.Store, 'cases', return_value=iter(self.cases)):
                manifest = s.seal_selection(run)
            self.assertEqual(manifest['state'], 'sealed')
            self.assertEqual(set(manifest['code_hashes']), set(s.SCIENCE_FILES))
            self.assertFalse(manifest['completed_record_nll_used_for_selection'])
            for name, digest in manifest['selection_output_hashes'].items():
                self.assertEqual(file_sha256(run / name), digest)
            self.assertEqual(len(pd.read_csv(directory / 'candidate_summary.csv')), 33)
            self.assertEqual(len(pd.read_csv(directory / 'candidate_configuration_summary.csv')), 264)
            with self.assertRaises(ValueError):
                s.seal_selection(run)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--output', required=True)
    args = parser.parse_args(); started = time.monotonic()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(SelectionControls)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = dict(state='passed' if result.wasSuccessful() else 'failed', created_at=now(),
        tests_run=result.testsRun, failures=len(result.failures), errors=len(result.errors),
        seconds=time.monotonic() - started, physical_executions=0, new_scientific_outcomes_read=0,
        fixture_validation_rows=16896, source_file_sha256=file_sha256(Path(s.__file__)),
        test_file_sha256=file_sha256(Path(__file__)),
        unknown_resource_convention='retained with unknown mean; positive infinity at the affected tie-break',
        checks=['exact complete matrix', 'missing/duplicate/foreign/seed/pair corruption',
                'all lexicographic priorities', 'completed NLL cannot select', 'configuration balance',
                'failed unknown resources retain prior NLL and all denominators', 'sealed artifact hashes'])
    atomic_json(args.output, report); print(json.dumps(report, sort_keys=True), flush=True)
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == '__main__':
    main()
