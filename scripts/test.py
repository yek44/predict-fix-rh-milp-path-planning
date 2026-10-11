"""Held-out test with the frozen model and recovery schedule.

On each instance the method fixes all but the 20% least confident groups.
If that returns no plan it frees 30%, and if that also fails it solves the
full MILP.  The full MILP is solved on every instance as the reference.
The paper reports the per-instance median over three runs of this script.

    python scripts/test.py --out results/test_run1.csv
    python scripts/test.py --profile default --out results/test_scip_default.csv
"""

import argparse

import numpy as np

import predictfix as pf
from predictfix.scenarios import test_scenarios
from common import CHECKPOINT, RESULTS, write_csv

RHO0, RHO1 = 0.2, 0.3


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--profile", choices=("strict", "default"), default="strict")
    ap.add_argument("--out", default=RESULTS / "test_run1.csv")
    args = ap.parse_args()

    predictor = pf.Predictor.load(CHECKPOINT)
    rows = []
    for i, scenario in enumerate(test_scenarios(args.n)):
        lay = pf.layout(scenario)
        pred = pf.predict(lay, predictor)
        first = pf.solve_residual(lay, pred, RHO0, profile=args.profile)
        second = None
        if not first.found:
            second = pf.solve_residual(lay, pred, RHO1, profile=args.profile)
        full = pf.solve(lay, profile=args.profile)
        if not full.found:
            raise RuntimeError(f"instance {i}: full MILP returned no plan ({full.status})")

        rescued = second is not None and second.found
        direct = pred.seconds + first.seconds + (0.0 if first.found else full.seconds)
        staged = (pred.seconds + first.seconds
                  + (second.seconds if second else 0.0)
                  + (full.seconds if second and not rescued else 0.0))
        direct_obj = first.objective if first.found else full.objective
        staged_obj = (first.objective if first.found
                      else second.objective if rescued else full.objective)
        rows.append(dict(
            instance=i, feature_s=pred.feature_s, prediction_s=pred.inference_s,
            prep_s=pred.seconds, n_groups=len(pred.groups),
            rho02_s=first.seconds, rho02_status=first.status, rho02_nodes=first.nodes,
            rho02_plan=first.found, rho02_objective=first.objective,
            rho03_executed=second is not None,
            rho03_s=second.seconds if second else 0.0,
            rho03_status=second.status if second else "not_run",
            rho03_nodes=second.nodes if second else 0,
            rho03_plan=rescued, rho03_objective=second.objective if second else None,
            full_s=full.seconds, full_status=full.status, full_nodes=full.nodes,
            full_objective=full.objective,
            direct_total_s=direct, staged_total_s=staged,
            direct_final_plan=True, staged_final_plan=True,
            direct_reference_match=pf.milp.same_objective(full.objective, direct_obj),
            staged_reference_match=pf.milp.same_objective(full.objective, staged_obj),
        ))
        print(f"[{i + 1:3d}/{args.n}] full {full.seconds:.2f}s  P&F {staged:.2f}s"
              f"{'' if first.found else '  (recovery: ' + ('rho=0.3' if rescued else 'full') + ')'}",
              flush=True)
    write_csv(args.out, rows)

    full_t = np.array([r["full_s"] for r in rows])
    pf_t = np.array([r["staged_total_s"] for r in rows])
    print(f"full MILP {full_t.mean():.2f}s, P&F {pf_t.mean():.2f}s, "
          f"reduction {100 * (1 - pf_t.sum() / full_t.sum()):.1f}%, "
          f"matches {sum(r['staged_reference_match'] for r in rows)}/{len(rows)}")


if __name__ == "__main__":
    main()
