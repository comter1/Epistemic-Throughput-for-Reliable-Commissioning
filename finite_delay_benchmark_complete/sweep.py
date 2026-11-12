"""逐因素实验；仅用户执行此脚本时才运行。每个条件共享同一批案例比较策略。"""
import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n', type=int, default=500)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42])
    parser.add_argument('--out', default='results/sweeps')
    parser.add_argument('--bundle-size', type=int, default=4)
    parser.add_argument('--rounds', type=int, nargs='+', default=[2])
    parser.add_argument('--dp', action='store_true', help='仅基础条件、最小 seed 和最小轮数运行 DP')
    parser.add_argument('--dp-library', choices=['single', 'matched'], default='single')
    parser.add_argument('--dp-max-budget', type=int, default=2)
    args = parser.parse_args()
    experiments = [
        ('base', []),
        ('shared_noise', ['--noise-mode', 'shared_round']),
        ('noise_low', ['--noise', '0.1']),
        ('noise_high', ['--noise', '0.3']),
        ('offset_small', ['--offset', '1']),
        ('offset_large', ['--offset', '3']),
        ('prior_nonuniform', ['--prior', '0.7', '0.1', '0.1', '0.1']),
        ('cal_cost_1', ['--calibration-cost', '1']),
        ('cal_cost_4', ['--calibration-cost', '4']),
    ]
    for rounds in sorted(set(args.rounds)):
        for seed in sorted(set(args.seeds)):
            for label, extra in experiments:
                destination = Path(args.out) / f'rounds_{rounds}' / f'seed_{seed}' / label
                cmd = [sys.executable, str(Path(__file__).with_name('benchmark.py')),
                       '--n', str(args.n), '--seed', str(seed), '--rounds', str(rounds),
                       '--bundle-size', str(args.bundle_size), '--out', str(destination), *extra]
                if args.dp and label == 'base' and seed == min(args.seeds) and rounds == min(args.rounds):
                    cmd += ['--dp', '--dp-library', args.dp_library,
                            '--dp-max-budget', str(args.dp_max_budget)]
                print('Running:', ' '.join(cmd), flush=True)
                subprocess.run(cmd, check=True)


if __name__ == '__main__':
    main()
