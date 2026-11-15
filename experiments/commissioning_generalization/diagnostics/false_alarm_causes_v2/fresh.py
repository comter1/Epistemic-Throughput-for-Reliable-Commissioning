"""Run sealed fresh A0 only after every source choice and checkpoint is frozen."""
import argparse
from pathlib import Path

from cgx.provenance import read_json, file_sha256
from .rollouts import run as rollout


def run(root, run_dir, device='cuda:1', workers=32, window=96, batch_size=16):
    run_dir = Path(run_dir)
    seal = read_json(run_dir / 'manifests/source_variant_selection.json')
    assert seal['state'] == 'sealed_before_fresh_outcomes'
    assert not seal['fresh_outcomes_exist_at_seal']
    assert not seal['fresh_initial_acquisitions_exist_at_seal']
    assert seal['fixed_gate_threshold'] == 0.
    assert read_json(run_dir / 'trajectories/source/complete.json')['state'] == 'passed'
    assert read_json(run_dir / 'head_replay_source_complete.json')['state'] == 'passed'
    for name, digest in seal['frozen_scientific_code'].items():
        assert file_sha256(Path(__file__).with_name(name)) == digest, name
    for relative, digest in seal['frozen_files'].items():
        assert file_sha256(run_dir / relative) == digest, relative
    for relative, digest in seal['all_head_checkpoints'].items():
        assert file_sha256(run_dir / relative) == digest, relative
    rollout(root, run_dir, run_dir / 'manifests/fresh_cases.json', 'confirmation', device,
            workers, window, batch_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--workers', type=int, default=32)
    parser.add_argument('--window', type=int, default=96)
    parser.add_argument('--batch-size', type=int, default=16)
    args = parser.parse_args()
    run(args.root, args.run_dir, args.device, args.workers, args.window, args.batch_size)
