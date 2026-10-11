"""Build the SideGRU training set by solving MILPs to optimality.

The released model was trained on about 114k labelled instances:

  canonical   the first 1000 scenarios of seed 13
  random      99k further start/goal draws on the training map, in shards
              with their own seeds
  rollouts    1400 closed-loop rollouts of 10 steps under the full MILP, so
              the data also contain moving starts (non-zero velocity)

Each shard is written to ``data/train/<name>.npz`` with the feature
sequences and labels of every vehicle-obstacle pair; existing shards are
skipped, so an interrupted run can be resumed.  Labelling is CPU-bound and
takes about a day with 8 workers; use ``--only`` to build a subset.

    python scripts/collect_data.py --workers 8
    python scripts/collect_data.py --only canonical
"""

import argparse
from multiprocessing import Pool

import numpy as np

import predictfix as pf
from predictfix.features import side_features, side_labels
from predictfix.scenarios import canonical_scenario, training_scenarios
from common import ROOT

# (shard name, kind, seed, count)
SHARDS = (
    [("canonical", "canonical", 13, 1000)]
    + [(f"random_a{w:02d}", "random", 1300 + w, 1125) for w in range(8)]
    + [(f"random_b{w:02d}", "random", 5300 + w, 1000) for w in range(20)]
    + [(f"random_c{w:02d}", "random", 9300 + w, 1000) for w in range(70)]
    + [(f"rollout_{w:02d}", "rollout", 20000 + w, 50) for w in range(28)]
)
ROLLOUT_STEPS = 10


def label(scenario):
    """Features and labels of one instance, or None if it was not solved."""
    lay = pf.layout(scenario)
    sol = pf.solve(lay, profile="labels", keep_values=True)
    if sol.status != "optimal":
        return None, None
    sequences, _ = side_features(lay)
    labels, _ = side_labels(lay, sol.x)
    return (np.stack(sequences), labels), sol.next_state(lay)


def build_shard(spec):
    name, kind, seed, count = spec
    path = ROOT / "data" / "train" / f"{name}.npz"
    if path.exists():
        return name, "exists"
    rng = np.random.RandomState(seed)
    items = []
    if kind == "canonical":
        items = [label(s)[0] for s in training_scenarios(count)]
    elif kind == "random":
        items = [label(canonical_scenario(rng))[0] for _ in range(count)]
    else:
        for _ in range(count):
            scenario = canonical_scenario(rng)
            for _ in range(ROLLOUT_STEPS):
                item, nxt = label(scenario)
                if item is None:
                    break
                items.append(item)
                scenario = scenario.advanced(nxt)
    items = [item for item in items if item is not None]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path,
                        features=np.concatenate([f for f, _ in items]),
                        labels=np.concatenate([y for _, y in items]).astype(np.int8))
    return name, f"{len(items)} instances"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--only", nargs="*", help="shard name prefixes to build")
    args = ap.parse_args()
    shards = [s for s in SHARDS
              if not args.only or any(s[0].startswith(p) for p in args.only)]
    with Pool(args.workers) as pool:
        for name, info in pool.imap_unordered(build_shard, shards):
            print(f"{name}: {info}", flush=True)


if __name__ == "__main__":
    main()
