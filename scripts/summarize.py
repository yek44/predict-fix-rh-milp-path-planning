"""Print the paper's result tables from the CSV files in results/.

    python scripts/summarize.py
"""

from collections import defaultdict

import numpy as np

from common import RESULTS, bootstrap_reduction, read_csv, reduction


def f(rows, key):
    return np.array([float(r[key]) for r in rows])


def true(value):
    return value in ("True", "1")


def oracle():
    path = RESULTS / "oracle.csv"
    if not path.exists():
        return
    rows = read_csv(path)
    print(f"\nOracle ({len(rows)} validation instances)")
    print(f"  full MILP {f(rows, 'full_s').mean():.2f} s, all sides fixed from the optimum "
          f"{1000 * f(rows, 'oracle_s').mean():.0f} ms, "
          f"same objective {sum(int(r['match']) for r in rows)}/{len(rows)}")


def validation():
    rows = read_csv(RESULTS / "validation.csv")
    full = [r for r in rows if not r["arm"].startswith("pf_")]
    n = len(full)
    print(f"\nValidation, {n} instances (full MILP mean {f(full, 'time_s').mean():.2f} s)")
    print(f"  {'rho':>4}  {'P&F [s]':>8}  {'reduction':>9}  {'initial plan':>12}  {'final match':>11}")
    by_arm = defaultdict(list)
    for r in rows:
        if r["arm"].startswith("pf_"):
            by_arm[float(r["rho"])].append(r)
    for rho, group in sorted(by_arm.items()):
        t = f(group, "deploy_s")
        print(f"  {rho:4.1f}  {t.mean():8.2f}  {reduction(f(full, 'time_s'), t):8.1f}%  "
              f"{sum(int(r['feasible']) for r in group):>8}/{n}  "
              f"{sum(int(r['final_reference_match']) for r in group):>7}/{n}")


def test():
    runs = [read_csv(RESULTS / f"test_run{k}.csv") for k in (1, 2, 3)]
    full = np.median([f(run, "full_s") for run in runs], axis=0)
    method = np.median([f(run, "staged_total_s") for run in runs], axis=0)
    first = runs[0]
    n = len(first)
    lo, hi = bootstrap_reduction(full, method)
    failed = [r for r in first if not true(r["rho02_plan"])]
    rescued = sum(true(r["rho03_plan"]) for r in failed)
    print(f"\nTest, {n} instances, median of {len(runs)} timing runs")
    print(f"  full MILP mean               {full.mean():.2f} s")
    print(f"  P&F mean                     {method.mean():.2f} s")
    print(f"  time reduction               {reduction(full, method):.1f}% "
          f"(95% CI {lo:.1f}-{hi:.1f}%)")
    print(f"  faster with P&F              {int((method < full).sum())}/{n}")
    print(f"  plan on first attempt        {n - len(failed)}/{n} "
          f"(rho=0.3 rescued {rescued}, full MILP {len(failed) - rescued})")
    print(f"  final feasible plans         {sum(true(r['staged_final_plan']) for r in first)}/{n}")
    print(f"  final matches with J*        {sum(true(r['staged_reference_match']) for r in first)}/{n}")

    path = RESULTS / "test_scip_default.csv"
    if path.exists():
        rows = read_csv(path)
        key = "method_s" if "method_s" in rows[0] else "staged_total_s"
        print(f"  with SCIP default settings   {f(rows, 'full_s').mean():.2f} s -> "
              f"{f(rows, key).mean():.2f} s")


def ranking():
    rows = read_csv(RESULTS / "ranking.csv")
    print("\nConfidence vs random selection (first attempt; time includes recovery)")
    print(f"  {'selection':<11} {'trials':>6}  {'initial plan':>12}  {'initial match':>13}  {'P&F [s]':>7}")
    for policy in ("confidence", "random"):
        group = [r for r in rows if r["policy"] == policy]
        n = len(group)
        print(f"  {policy:<11} {n:>6}  {100 * sum(int(r['feasible']) for r in group) / n:11.1f}%  "
              f"{100 * sum(int(r['reference_match']) for r in group) / n:12.1f}%  "
              f"{f(group, 'deploy_s').mean():7.2f}")


def closed_loop():
    rows = [r for r in read_csv(RESULTS / "closed_loop.csv") if r.get("pf_s")]
    full, method = f(rows, "full_s"), f(rows, "pf_s")
    n = len(rows)
    print(f"\nClosed loop, {len({r['rollout'] for r in rows})} simulations, {n} replanning steps")
    print(f"  mean time                    {full.mean():.2f} s -> {method.mean():.2f} s "
          f"({reduction(full, method):.1f}% reduction)")
    print(f"  95th percentile              {np.percentile(full, 95):.2f} s -> "
          f"{np.percentile(method, 95):.2f} s")
    print(f"  rho=0.3 rescues / full MILP  {sum(int(r['recovered']) for r in rows)} / "
          f"{sum(int(r['fallback']) for r in rows)}")
    print(f"  final matches with J*        {sum(int(r['reference_match']) for r in rows)}/{n}")


def transfer():
    rows = read_csv(RESULTS / "transfer.csv")
    print("\nTransfer, same model and recovery schedule")
    print(f"  {'setting':<15} {'n':>3}  {'SCIP [s]':>8}  {'P&F [s]':>7}  {'initial plan':>12}  {'final match':>11}")
    for setting in dict.fromkeys(r["setting"] for r in rows):
        g = [r for r in rows if r["setting"] == setting]
        cert = [r for r in g if int(r["certified_reference"])]
        print(f"  {setting:<15} {len(g):>3}  {f(g, 'full_s').mean():8.2f}  {f(g, 'deploy_s').mean():7.2f}  "
              f"{sum(int(r['residual_plan']) for r in g):>8}/{len(g):<3}  "
              f"{sum(int(r['reference_match']) for r in cert):>7}/{len(cert)}")


if __name__ == "__main__":
    oracle()
    validation()
    test()
    ranking()
    closed_loop()
    transfer()
