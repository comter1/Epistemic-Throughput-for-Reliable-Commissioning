# Implementation and actual execution

`STATUS.md` and the external command/artifact records determine actual completion.
Formal research results are unavailable until the main driver completes P0–P6.

- Branch `feat/commissioning-generalization-main` starts at
  `8b6681efaf6a11d0af886f55b23ecc9590a67acd`. New source and small reports are confined to this
  directory. Protected legacy hashes and the original five tests are checked unchanged.
- Python 3.12.3, MuJoCo 4.0.1 and PyTorch 2.14.1+cu130 run in an isolated environment. Both
  B200s passed actual BF16 forward/backward and graph aggregation. No driver/global changes
  were made. Panda asset commit: `370c43965323c0944b433b202aac5058b5f907a9`.
- Candidate manifests contain 252 configurations and 48,640 cases. Exact isomorphism,
  matching, family balance, topology folds, private streams and repair subsets are checked.
- Every ten-second execution takes 5000 MuJoCo steps. Cache, queue and adapter faults
  alter actual controller input, release times or payload. Ordinary wrong-node repairs
  leave the true fault intact. Source-only settling failures remain recorded. A uniform
  0.1-second smooth reference transition fixes timing before sealing; gains, thresholds,
  task samples and fault cases remain unchanged. Full healthy catalogs are rerun.
- Wire events really occupy 128 bytes and calibration bursts 160 bytes. Raw bytes are
  charged before encoding. No true time, hidden offsets, ticks or numeric message IDs
  become model inputs. Tests cover independent noise, calibration algebra and strict caps.
- Variable graph/target/action heads share temporal evidence and public topology. The
  Transformer receives relations too. Crossfit readers produce source rewards; the common
  final reader remains frozen for every method. All five seeds and formal update caps
  remain registered. Checkpoints save optimizer, target weights, RNG and hash identities.
- The phase driver implements every required CLI stage with prerequisite checks, atomic
  completion evidence, independent per-GPU jobs and resumability. Actual invocations record
  command, timestamps, exit code, configuration/source hashes, Git state and output paths.

Initial setup preceded the command recorder. Its executed commands and retained install
logs are reconstructed in `reports/setup_evidence.json`; exact initial command start times
are not asserted. One failed relative-path editable-install attempt was corrected using an
absolute path, but its original raw log was overwritten. Formal stages use the full recorder.

The 1000-run and 1000-update measurements are engineering capacity tests, not scientific
estimates. Forecasts distinguish optimizer time from preprocessing, source generation,
full validation rollouts, target acquisition and audit. The 320 GPU-hour soft cap triggers
a report without pruning methods or seeds. All raw public JSON observations are retained
losslessly; source binary wire retention uses a deterministic 2% case subset.

Scientific tables/figures require complete frozen evaluations and contain all registered
systems, paired bootstrap intervals, seed variability, secondary Holm corrections,
independent-reader scores and repair risk/coverage/yield. Outputs not yet run stay explicitly
unexecuted. No positive result is assumed or inferred from engineering checks.
