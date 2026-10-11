"""How much of the solve time is spent on the obstacle-side decisions?

Solve each validation instance as the full MILP, then fix every
obstacle-side group to the face its optimal trajectory clears and solve
again.  With all groups fixed only an LP remains.  This is an oracle, not a
method: it needs the optimal solution first.

    python scripts/oracle.py [--n 100]
"""

import argparse

import numpy as np

import predictfix as pf
from predictfix.features import side_labels
from predictfix.scenarios import validation_scenarios
from common import RESULTS, write_csv


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", default=RESULTS / "oracle.csv")
    args = ap.parse_args()

    rows = []
    for i, scenario in enumerate(validation_scenarios(args.n)):
        lay = pf.layout(scenario)
        full = pf.solve(lay, keep_values=True)
        labels, _ = side_labels(lay, full.x)
        oracle = pf.solve(lay, (lay.sides.reshape(-1, 4), labels.reshape(-1)))
        rows.append(dict(instance=i, full_s=full.seconds, full_nodes=full.nodes,
                         full_objective=full.objective, oracle_s=oracle.seconds,
                         oracle_status=oracle.status, oracle_objective=oracle.objective,
                         match=int(pf.milp.same_objective(full.objective, oracle.objective))))
        print(f"[{i + 1:3d}/{args.n}] full {full.seconds:.2f}s  "
              f"all sides fixed {1000 * oracle.seconds:.0f} ms", flush=True)
    write_csv(args.out, rows)
    print(f"full MILP {np.mean([r['full_s'] for r in rows]):.2f}s, "
          f"oracle {1000 * np.mean([r['oracle_s'] for r in rows]):.0f} ms, "
          f"same objective {sum(r['match'] for r in rows)}/{len(rows)}")


if __name__ == "__main__":
    main()
