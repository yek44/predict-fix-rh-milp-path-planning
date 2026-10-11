"""Choose the free fraction rho on the validation set.

Every instance is solved once as the full MILP and once per candidate rho.
A residual problem that returns no plan is charged the full-MILP time on top,
so an unreliable rho cannot look fast.  The rho with the lowest mean time is
the one used for the first attempt in every later experiment.

    python scripts/validate.py [--n 100]
"""

import argparse

import numpy as np

import predictfix as pf
from predictfix.scenarios import validation_scenarios
from common import CHECKPOINT, RESULTS, write_csv

RHOS = (0.0, 0.1, 0.2, 0.3, 0.4)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", default=RESULTS / "validation.csv")
    args = ap.parse_args()

    predictor = pf.Predictor.load(CHECKPOINT)
    rows = []
    for i, scenario in enumerate(validation_scenarios(args.n)):
        lay = pf.layout(scenario)
        full = pf.solve(lay)
        if not full.found:
            raise RuntimeError(f"instance {i}: full MILP returned no plan ({full.status})")
        ref = full.objective
        rows.append(dict(instance=i, arm="full_milp", rho="", time_s=full.seconds,
                         deploy_s=full.seconds, status=full.status, nodes=full.nodes,
                         feasible=1, reference_match=1, final_reference_match=1,
                         objective=ref, reference_objective=ref))
        pred = pf.predict(lay, predictor)
        line = []
        for rho in RHOS:
            sol = pf.solve_residual(lay, pred, rho)
            method = pred.seconds + sol.seconds
            fallback = 0.0 if sol.found else full.seconds
            match = pf.milp.same_objective(ref, sol.objective)
            rows.append(dict(instance=i, arm=f"pf_rho_{rho:g}", rho=rho, time_s=method,
                             deploy_s=method + fallback, status=sol.status, nodes=sol.nodes,
                             feasible=int(sol.found), reference_match=int(match),
                             final_reference_match=int(match or not sol.found),
                             objective=sol.objective if sol.found else "",
                             reference_objective=ref,
                             free_groups=int(round(rho * len(pred.groups))),
                             n_groups=len(pred.groups), prediction_s=pred.inference_s,
                             feature_s=pred.feature_s, fallback_s=fallback))
            line.append(f"{rho:g}:{method:.2f}s{'' if sol.found else '+full'}")
        print(f"[{i + 1:3d}/{args.n}] full {full.seconds:.2f}s  " + "  ".join(line), flush=True)
    write_csv(args.out, rows)

    full_t = np.array([r["time_s"] for r in rows if r["arm"] == "full_milp"])
    means = {rho: np.mean([r["deploy_s"] for r in rows if r["rho"] == rho]) for rho in RHOS}
    best = min(RHOS, key=lambda rho: (means[rho], rho))
    print(f"full MILP mean {full_t.mean():.2f}s; "
          + ", ".join(f"rho={r:g}: {m:.2f}s" for r, m in means.items()))
    print(f"selected rho = {best:g}")


if __name__ == "__main__":
    main()
