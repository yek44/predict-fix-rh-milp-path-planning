# Learning to Accelerate Receding-Horizon MILP Path Planning

Code, trained model and results for the paper *Learning to Accelerate
Receding-Horizon MILP Path Planning* by Yunus Emre Kara and Murad Abu-Khalaf,
submitted to the 2027 European Control Conference.

A receding-horizon planner in the style of Schouwenaars et al. (ECC 2001)
solves a mixed-integer linear program at every control step. Most of the
solve time goes into the binary variables that decide on which side of each
obstacle the vehicle passes at each step of the horizon. We train a small
recurrent network, SideGRU, to predict these sides from the problem data,
fix the predictions it is most confident about, and let SCIP solve what is
left. On held-out problems this cuts the mean planning time by 79%, and 87 of
100 plans keep the optimal objective of the full MILP.

## Method

With one vehicle, 13 obstacles and a 4 s horizon at 0.1 s, the MILP has 520
*obstacle-side groups*: for every obstacle and step, four binaries that pick
the left, right, lower or upper face of the obstacle. Fixing all of them to
the sides of an optimal trajectory leaves a linear program that SCIP solves
about 20 times faster than the full MILP (87 ms against 1.64 s in the paper).
Learning these sides is therefore the target.

1. **Predict.** For every vehicle-obstacle pair, SideGRU reads a sequence of
   28 features over the horizon: start, goal and obstacle geometry, plus a
   straight-line and a constant-velocity guess of the vehicle position. No
   solver call is needed. A bidirectional GRU outputs a probability for each
   of the four sides at every step. Training labels come from optimal MILP
   solutions; the label of a group is the side the optimal trajectory clears
   by the largest margin.
2. **Fix.** Rank the groups by the probability of the predicted side and fix
   all but the least confident fraction ρ. Fixing only tightens variable
   bounds, so any plan the reduced problem returns is feasible for the
   original MILP.
3. **Recover.** If the reduced problem has no feasible plan, free more groups
   (ρ = 0.2, then 0.3) and, as a last resort, solve the full MILP.

ρ = 0.2 and the recovery step to 0.3 were chosen on a validation set and kept
fixed for every experiment afterwards.

## Results

All times include building the MILP, and for predict-and-fix (P&F) also the
features, the network and every recovery attempt. SCIP 10 runs with presolve
and cut separation off for both methods. A plan *matches* when its objective is
within 1e-4 (relative) of the full-MILP optimum.

| Experiment | Problems | Full MILP | P&F | Feasible | Match |
|---|---:|---:|---:|---:|---:|
| Held-out test | 100 | 1.49 s | 0.31 s | 100/100 | 87/100 |
| Held-out test, SCIP default settings | 100 | 8.45 s | 1.47 s | 100/100 | 87/100 |
| Closed loop (20 runs × 30 replans) | 600 | 1.01 s | 0.22 s | 600/600 | 540/600 |
| Unseen maps | 20 | 0.65 s | 0.19 s | 20/20 | 11/20 |
| 6 s horizon, unseen maps | 20 | 4.99 s | 0.53 s | 20/20 | 14/20 |
| Denser unseen maps (17–20 obstacles) | 20 | 2.69 s | 0.63 s | 20/20 | 13/20 |
| Two vehicles | 30 | 0.97 s | 0.30 s | 30/30 | 25/30 |
| Three vehicles | 20 | 15.14 s | 5.96 s | 20/20 | 17/19 |

On the test set the reduction is 79.1% (95% bootstrap interval 72.2–84.0%),
and P&F is faster on 87 of the 100 problems. The first attempt returns a plan
on 97; of the other three, freeing 30% recovers two and one falls back to the
full MILP. Choosing the fixed groups at random instead of by confidence,
with the same predictions and the same number of fixes, drops first-attempt feasibility from 97% to 69%
and raises the mean time to 0.84 s. With several vehicles the model, trained
on one vehicle, predicts only the vehicle-obstacle groups; SCIP decides all
vehicle-vehicle separations. One three-vehicle reference hit the 120 s limit
and is left out of the match count.

`python scripts/summarize.py` prints every table from the CSV files in
[`results/`](results).

## Installation

Python 3.10 or newer. The SCIP solver comes with the PySCIPOpt wheel.

```bash
git clone https://github.com/yek44/predict-fix-rh-milp-path-planning.git
cd predict-fix-rh-milp-path-planning
pip install -e .
```

The results were measured with Python 3.14, SCIP 10.0, PySCIPOpt 6.2.1 and
PyTorch 2.13 (CPU) under WSL Ubuntu 24.04 on an AMD Ryzen 7 6800H. Solver
outcomes are deterministic; absolute times depend on the machine.

## Usage

```python
import predictfix as pf
from predictfix.scenarios import test_scenarios

predictor = pf.Predictor.load("checkpoints/sidegru.pt")
scenario = test_scenarios(5)[4]   # one vehicle, 13 obstacles, 4 s horizon
milp = pf.layout(scenario)

full = pf.solve(milp)             # SCIP decides every binary
plan = pf.plan(milp, predictor)   # predict, fix, recover if needed

print(f"full MILP {full.seconds:.2f} s, objective {full.objective:.3f}")
print(f"P&F       {plan.seconds:.2f} s, objective {plan.solution.objective:.3f}")
```

To plan from your own state, build a `pf.scenarios.Scenario` with the start
and goal states (`[px, py, vx, vy]` per vehicle) and the obstacle rectangles
(`[xmin, xmax, ymin, ymax]`). `scenario.advanced(state)` re-poses the same
problem from a new measured state for the next replanning step.

## Reproducing the paper

Each script writes one CSV to `results/`; the files in the repository are the
ones behind the paper.

| Script | Paper | Runtime |
|---|---|---|
| `scripts/oracle.py` | cost of the side decisions (motivation) | ~4 min |
| `scripts/validate.py` | choice of ρ on 100 validation problems | ~10 min |
| `scripts/test.py --out results/test_runK.csv` | held-out test, run three times | ~4 min each |
| `scripts/test.py --profile default --out results/test_scip_default.csv` | SCIP default settings | ~20 min |
| `scripts/ranking.py` | confidence vs random selection | ~10 min |
| `scripts/closed_loop.py` | closed-loop replanning | ~15 min |
| `scripts/transfer.py` | unseen maps, longer horizon, more vehicles | ~15 min |
| `scripts/summarize.py` | prints all tables | seconds |

The released model was trained on about 114k instances solved to optimality:
1,000 canonical problems, 99,000 further start/goal draws on the same map,
and 1,400 closed-loop rollouts of 10 steps so that the data include moving
starts. To rebuild it:

```bash
python scripts/collect_data.py --workers 8   # writes data/train/*.npz, about a day
python scripts/train.py                      # ~1 h on a GPU
```

The validation and test problems never enter training. The training,
validation and test streams are fixed by their random seeds in
[`predictfix/scenarios.py`](predictfix/scenarios.py).

## Repository layout

```
predictfix/          the method
  scenarios.py       maps, start/goal sampling, fixed problem streams
  milp.py            the MILP, solver settings, solve with fixed sides
  features.py        SideGRU inputs and training labels
  model.py           SideGRU and checkpoint loading
  predict_fix.py     confidence ranking, fixing, recovery
scripts/             one script per experiment, data collection, training
checkpoints/         the trained SideGRU (1.1 MB)
results/             per-instance results behind every table
```

## Citation

```bibtex
@unpublished{kara2027accelerate,
  title  = {Learning to Accelerate Receding-Horizon {MILP} Path Planning},
  author = {Kara, Yunus Emre and Abu-Khalaf, Murad},
  note   = {Submitted to the 2027 European Control Conference},
  year   = {2026}
}
```

This work grew out of the first author's graduation project at Istanbul
Technical University.

## License

MIT, see [LICENSE](LICENSE).
