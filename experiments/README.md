# Experiment scripts

The exact commands behind the numbers in `docs/contour_selector_results.md`.

These used to live in a session scratchpad and were thrown away. That cost a day: the
script that produced the zero-training control was gone, so reproducing it meant
comparing mask files bit by bit to work out which feature it had used, and the budget
rule behind it was never recovered at all. A number nobody can regenerate is a number
nobody can check.

So they live here now, unedited, including the ones whose results were negative.

| script | what it measured | where the result is |
|---|---|---|
| `sn_sweep.sh` | ScanNet retention 2–20% | §4.8 |
| `thr_sweep.sh` | every hand-crafted feature on the full split | §3 |
| `lotus_ctrl.sh` | the training-free controls on Lotus v1 | §3 |
| `lotus_seeds.sh` | four seeds on Lotus v1 | §4.8 |
| `mix2.sh` | mixed training with per-source validation | §4.9 |
| `grid.sh` | the selector at SAM point grids 16 and 32 | §4.11 |
| `sn_indomain.sh` | training on ScanNet, testing on held-out ScanNet scenes | §4.12 |
| `sn_seeds.sh` | three more seeds for the above | §4.12 |
| `hs_sweep.sh` | Hypersim retention 2–40%, both directions | §4.12 |
| `nyu393.sh` | NYUv2 cut to ScanNet's frame count | §4.12 |
| `see_full.sh` | Soft Edge Error on both depth models | — |
| `label_t_sweep.sh` | the label threshold at 15, 20 and 25%, scored on test | §5 |
| `label_t_on_val.sh` | the same four thresholds, scored on a validation split | §5 |
| `hs10k_in_domain.sh` | the Hypersim-trained selector in its own domain | §5 |

## Paths

These were written before `utils/dataset_paths.py` existed, so they spell every path
out. New scripts should pass `--dataset` and `--model` instead and let the registry
resolve them; that is what stops a sweep being launched against the wrong directory at
the wrong resolution, which happened here with Hypersim.

They are left as they were run rather than updated, because they are the record of how
each number was produced.
