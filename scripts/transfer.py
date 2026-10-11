"""Transfer to settings the model never saw, with no retraining.

Unseen maps, a 6 s horizon, denser maps, and two- and three-vehicle
crossings.  With several vehicles SideGRU still predicts only the
vehicle-obstacle groups; every vehicle-vehicle separation decision stays
with SCIP.  A full-MILP reference that hits the time limit is kept in the
timing but not used for the optimum-match count.

    python scripts/transfer.py
"""

import argparse

import numpy as np

import predictfix as pf
from predictfix import scenarios as S
from common import CHECKPOINT, RESULTS, write_csv

# name, horizon [s], obstacles, size regime, stream seed
SINGLE_VEHICLE = (
    ("unseen map", 4.0, 13, S.SPARSE_MAPS, 2091),
    ("longer horizon", 6.0, 13, S.SPARSE_MAPS, 5091),
    ("denser clutter", 4.0, 20, S.DENSE_MAPS, 7091),
)
CROSSINGS = (("2 vehicles", 2, 188), ("3 vehicles", 3, 288))


def settings(n_single, n_two, n_three):
    for name, horizon, n_obs, sizes, seed in SINGLE_VEHICLE:
        rng = np.random.RandomState(seed)
        for i in range(n_single):
            yield name, i, S.unseen_map_scenario(rng, horizon, n_obs, sizes)
    for (name, k, seed), count in zip(CROSSINGS, (n_two, n_three)):
        rng = np.random.RandomState(seed)
        for i in range(count):
            yield name, i, S.crossing_scenario(rng, k)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-single", type=int, default=20)
    ap.add_argument("--n-two", type=int, default=30)
    ap.add_argument("--n-three", type=int, default=20)
    ap.add_argument("--out", default=RESULTS / "transfer.csv")
    args = ap.parse_args()

    predictor = pf.Predictor.load(CHECKPOINT)
    rows = []
    for name, i, scenario in settings(args.n_single, args.n_two, args.n_three):
        lay = pf.layout(scenario)
        full = pf.solve(lay)
        plan = pf.plan(lay, predictor, full=full)
        (_, first), *rest = plan.attempts
        second = rest[0][1] if rest else None
        certified = full.status == "optimal"
        sol = plan.solution
        rows.append(dict(
            setting=name, instance=i, vehicles=scenario.n_vehicles,
            horizon_steps=scenario.horizon_steps, obstacles=len(scenario.obstacles),
            obstacle_groups=lay.sides[..., 0].size, pair_groups=lay.pair_sides[..., 0].size,
            full_s=full.seconds, full_status=full.status, full_nodes=full.nodes,
            full_objective=full.objective, deploy_s=plan.seconds,
            initial_residual_s=first.seconds, recovery_s=second.seconds if second else 0.0,
            residual_status=first.status, residual_plan=int(first.found),
            recovery_attempted=int(not first.found),
            recovered=int(second is not None and second.found),
            fallback=int(plan.rho == 1.0 and full.found), final_plan=int(sol.found),
            certified_reference=int(certified),
            reference_match=int(pf.milp.same_objective(full.objective, sol.objective))
            if certified else ""))
        print(f"[{name:14s} {i + 1:2d}] full {full.seconds:6.2f}s {full.status:9s} "
              f"P&F {plan.seconds:5.2f}s (rho={plan.rho:g})", flush=True)
    write_csv(args.out, rows)


if __name__ == "__main__":
    main()
