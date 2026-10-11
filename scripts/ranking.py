"""Confidence ranking against random selection on the test set.

Both arms use the same predictions and fix the same number of groups; only
the choice of which groups to fix differs.  The random arm is repeated five
times per instance with the same recovery schedule.  The confidence arm and
the full-MILP times are taken from the three test runs, so run
``scripts/test.py`` first.

    python scripts/ranking.py
"""

import argparse

import numpy as np

import predictfix as pf
from predictfix.scenarios import test_scenarios
from common import CHECKPOINT, RESULTS, read_csv, write_csv

RHO0, RHO1 = 0.2, 0.3
REPEATS = 5


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--runs", nargs="+",
                    default=[RESULTS / f"test_run{r}.csv" for r in (1, 2, 3)])
    ap.add_argument("--out", default=RESULTS / "ranking.csv")
    args = ap.parse_args()

    runs = [read_csv(path) for path in args.runs]
    predictor = pf.Predictor.load(CHECKPOINT)
    rows = []
    for i, scenario in enumerate(test_scenarios(args.n)):
        measured = [run[i] for run in runs]
        ref = float(measured[0]["full_objective"])
        full_s = float(np.median([float(m["full_s"]) for m in measured]))
        conf_s = float(np.median([float(m["staged_total_s"]) for m in measured]))
        first = measured[0]["rho02_objective"]
        rows.append(dict(
            instance=i, policy="confidence", repeat=0, rho=RHO0, baseline_s=full_s,
            method_s=conf_s, deploy_s=conf_s,
            feasible=int(measured[0]["rho02_plan"] == "True"),
            reference_match=int(first != "" and pf.milp.same_objective(ref, float(first))),
            final_reference_match=int(measured[0]["staged_reference_match"] == "True"),
            objective=first, reference_objective=ref))

        lay = pf.layout(scenario)
        pred = pf.predict(lay, predictor)
        outcome = []
        for repeat in range(REPEATS):
            score = np.random.RandomState(70_000 + 1009 * i + repeat).random(len(pred.groups))
            sol = pf.solve_residual(lay, pred, RHO0, score=score)
            rec = None if sol.found else pf.solve_residual(lay, pred, RHO1, score=score)
            to_full = rec is not None and not rec.found
            method = pred.seconds + sol.seconds + (rec.seconds if rec else 0.0)
            final = (sol.objective if sol.found
                     else rec.objective if not to_full else ref)
            rows.append(dict(
                instance=i, policy="random", repeat=repeat, rho=RHO0, baseline_s=full_s,
                method_s=method, deploy_s=method + (full_s if to_full else 0.0),
                feasible=int(sol.found),
                reference_match=int(pf.milp.same_objective(ref, sol.objective)),
                final_reference_match=int(pf.milp.same_objective(ref, final)),
                objective=sol.objective if sol.found else "", reference_objective=ref,
                status=sol.status, recovery_status=rec.status if rec else "not_run",
                recovery_s=rec.seconds if rec else 0.0, full_fallback=int(to_full),
                nodes=sol.nodes, free_groups=int(round(RHO0 * len(pred.groups)))))
            outcome.append("ok" if sol.found else "fail")
        print(f"[{i + 1:3d}/{args.n}] random: {' '.join(outcome)}", flush=True)
    write_csv(args.out, rows)


if __name__ == "__main__":
    main()
