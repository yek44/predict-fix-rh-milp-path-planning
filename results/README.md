# Results

Per-instance outputs of the runs reported in the paper. Times are wall-clock
seconds and include model construction; `*_objective` columns are MILP costs.

| File | Written by | Contents |
|---|---|---|
| `oracle.csv` | `oracle.py` | full MILP vs. all side groups fixed from its optimum, validation set |
| `validation.csv` | `validate.py` | one row per instance and arm (`full_milp` or `pf_rho_<ρ>`); `deploy_s` adds the full-MILP time when the residual problem has no plan |
| `test_run1.csv` … `test_run3.csv` | `test.py` | three timing runs of the held-out test; the paper takes the per-instance median. `staged_total_s` is the deployed method (ρ = 0.2 → 0.3 → full MILP), `direct_total_s` the variant that skips ρ = 0.3 |
| `test_scip_default.csv` | `test.py --profile default` | the same test with SCIP's default settings |
| `ranking.csv` | `ranking.py` | confidence ranking (`repeat` 0) vs. five random selections per instance |
| `closed_loop.csv` | `closed_loop.py` | one row per replanning step; `pf_s` is the deployed time, `full_s` the reference solved at the same state |
| `transfer.csv` | `transfer.py` | unseen maps, 6 s horizon, denser maps, two and three vehicles |

`oracle.csv` was rerun with the released code in a later session. It solves
the same problems with the same node counts and objectives as
`validation.csv`, but the machine was faster that day: 1.02 s and 49 ms on
average, against 1.64 s and 87 ms in the paper. Compare times only within
one file.

Apart from `oracle.csv`, these files come from the research version of the
scripts, which built the same MILPs (checked constraint by constraint) but
logged a few extra columns. In `validation.csv` the baseline arm is labelled
`scip_strict_tuned`, and `test_scip_default.csv` has its own column set.
`summarize.py` reads both the old and the new layouts.
