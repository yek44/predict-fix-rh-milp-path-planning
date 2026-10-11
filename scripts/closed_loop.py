"""Closed-loop replanning: 20 simulations of 30 replanning steps.

At every step the predict-and-fix plan is applied and the vehicle advances by
one sample.  The full MILP is solved at the same state as the reference, so
both methods are compared on exactly the states the controller visits.

    python scripts/closed_loop.py [--rollouts 20 --steps 30 --seed 41]
"""

import argparse

import numpy as np

import predictfix as pf
from predictfix.scenarios import canonical_scenario
from common import CHECKPOINT, RESULTS, write_csv


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rollouts", type=int, default=20)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=41)
    ap.add_argument("--out", default=RESULTS / "closed_loop.csv")
    args = ap.parse_args()

    predictor = pf.Predictor.load(CHECKPOINT)
    rng = np.random.RandomState(args.seed)
    initial = [canonical_scenario(rng) for _ in range(args.rollouts)]
    rows = []
    for r, scenario in enumerate(initial):
        fuel = 0.0
        for step in range(args.steps):
            lay = pf.layout(scenario)
            full = pf.solve(lay, keep_values=True)
            if not full.found:
                rows.append(dict(rollout=r, step=step, full_status=full.status,
                                 failure="full MILP returned no plan"))
                break
            plan = pf.plan(lay, predictor, full=full, keep_values=True)
            (_, first), *rest = plan.attempts
            second = rest[0][1] if rest else None
            sol = plan.solution

            du = np.linalg.norm(sol.first_input(lay) - full.first_input(lay))
            dx = np.linalg.norm(sol.next_state(lay) - full.next_state(lay))
            fuel += np.abs(sol.first_input(lay)).sum()
            rows.append(dict(
                rollout=r, step=step, rho0=0.2, rho1=0.3,
                full_s=full.seconds, pf_s=plan.seconds,
                residual_s=first.seconds, recovery_s=second.seconds if second else 0.0,
                recovery_attempted=int(not first.found),
                recovered=int(second is not None and second.found),
                fallback=int(plan.rho == 1.0),
                residual_status=first.status,
                recovery_status=second.status if second else "not_run",
                full_status=full.status, full_nodes=full.nodes,
                residual_nodes=first.nodes, recovery_nodes=second.nodes if second else 0,
                reference_match=int(pf.milp.same_objective(full.objective, sol.objective)),
                relative_gap=(sol.objective - full.objective) / (abs(full.objective) + 1.0),
                control_l2=du, next_state_l2=dx, cumulative_fuel=fuel))
            scenario = scenario.advanced(sol.next_state(lay))
        steps = [row for row in rows if row["rollout"] == r and "pf_s" in row]
        print(f"rollout {r + 1:2d}/{args.rollouts}: {len(steps)} steps, "
              f"full {np.mean([s['full_s'] for s in steps]):.2f}s, "
              f"P&F {np.mean([s['pf_s'] for s in steps]):.2f}s, "
              f"fallbacks {sum(s['fallback'] for s in steps)}", flush=True)
    write_csv(args.out, rows)


if __name__ == "__main__":
    main()
