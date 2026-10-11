"""The receding-horizon path-planning MILP of Schouwenaars et al. (ECC 2001).

Each vehicle follows a discretised double integrator.  The cost is the 1-norm
of the inputs plus a weighted 1-norm of the terminal error, both written with
slack variables.  Collision avoidance uses big-M disjunctions: for vehicle k,
obstacle o and step n there are four binaries ``b[k, o, n, side]``, one per
face (left, right, below, above), and ``b = 1`` relaxes that face.  At least
one face must hold, so ``sum(b) <= 3``.  In the paper's notation
``alpha = 1 - b``.  Vehicle pairs get the same construction on their relative
position.

Every solve, full or partially fixed, builds a fresh model, so the reported
times include model construction.
"""

import time
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import pyscipopt as scip

TIME_LIMIT = 120.0
FACE_EPS = 1e-4
SIDES = ("left", "right", "below", "above")


@dataclass
class Layout:
    """Column index of every MILP variable, plus the scenario it came from.

    ``state[k, i, n]`` is px, py, vx, vy of vehicle k at step n = 0..N,
    ``inputs[k, i, n]`` is ux, uy at n = 0..N-1 and ``input_abs`` their
    1-norm slacks, ``terminal[k]`` the terminal-error slacks, and
    ``sides[k, o, n - 1, :]`` the four face binaries of obstacle o at step n.
    ``pair_sides[p, n - 1, :]`` does the same for vehicle pair ``pairs[p]``.
    """

    scenario: object
    state: np.ndarray
    inputs: np.ndarray
    input_abs: np.ndarray
    terminal: np.ndarray
    sides: np.ndarray
    pairs: list
    pair_sides: np.ndarray
    n_vars: int
    lb: np.ndarray = field(repr=False)
    ub: np.ndarray = field(repr=False)
    obj: np.ndarray = field(repr=False)
    binary: np.ndarray = field(repr=False)

    @property
    def horizon_steps(self) -> int:
        return self.state.shape[2] - 1


@dataclass
class Solution:
    status: str
    objective: float | None
    seconds: float
    nodes: int
    x: np.ndarray | None = None  # primal values by column index

    @property
    def found(self) -> bool:
        return self.objective is not None

    def next_state(self, layout: Layout) -> np.ndarray:
        """State of every vehicle after the first control input."""
        return self.x[layout.state[:, :, 1]]

    def first_input(self, layout: Layout) -> np.ndarray:
        return self.x[layout.inputs[:, :, 0]]


def vehicle_pairs(scenario):
    """Pairs whose start-goal boxes, padded by ``pair_margin``, overlap."""
    if scenario.n_vehicles < 2 or scenario.safe_separation <= 0:
        return []
    boxes = []
    for s, g in zip(scenario.start, scenario.goal):
        pts = np.vstack([s[:2], g[:2]])
        boxes.append((pts.min(axis=0) - scenario.pair_margin,
                      pts.max(axis=0) + scenario.pair_margin))
    pairs = []
    for i, j in combinations(range(scenario.n_vehicles), 2):
        (lo_i, hi_i), (lo_j, hi_j) = boxes[i], boxes[j]
        if (hi_i[0] >= lo_j[0] and hi_j[0] >= lo_i[0]
                and hi_i[1] >= lo_j[1] and hi_j[1] >= lo_i[1]):
            pairs.append((i, j))
    return pairs


def layout(scenario) -> Layout:
    """Assign columns, bounds and objective coefficients."""
    K = scenario.n_vehicles
    L = len(scenario.obstacles)
    N = scenario.horizon_steps
    pairs = vehicle_pairs(scenario)

    counter = 0

    def block(*shape):
        nonlocal counter
        ids = np.arange(counter, counter + int(np.prod(shape))).reshape(shape)
        counter += ids.size
        return ids

    state = block(K, 4, N + 1)
    inputs = block(K, 2, N)
    input_abs = block(K, 2, N)
    terminal = block(K, 4)
    sides = block(K, L, N, 4)
    pair_sides = block(len(pairs), 4, N).transpose(0, 2, 1)
    n = counter

    lb = np.full(n, -np.inf)
    ub = np.full(n, np.inf)
    for i in range(2):
        lb[state[:, i]], ub[state[:, i]] = scenario.pos_bounds[i]
        lb[state[:, i + 2]], ub[state[:, i + 2]] = scenario.vel_bounds[i]
        lb[inputs[:, i]], ub[inputs[:, i]] = scenario.input_bounds[i]
    lb[input_abs] = 0.0
    lb[terminal] = 0.0
    binary = np.zeros(n, dtype=bool)
    for ids in (sides, pair_sides):
        binary[ids] = True
        lb[ids], ub[ids] = 0.0, 1.0

    obj = np.zeros(n)
    obj[input_abs] = 1.0
    w_pos, w_vel = scenario.terminal_weights
    obj[terminal[:, :2]] = w_pos
    obj[terminal[:, 2:]] = w_vel

    return Layout(scenario, state, inputs, input_abs, terminal, sides, pairs,
                  pair_sides, n, lb, ub, obj, binary)


def build(lay: Layout, time_limit=TIME_LIMIT):
    """Create the SCIP model.  Returns the model and its variables by column."""
    sc = lay.scenario
    K, N = sc.n_vehicles, lay.horizon_steps
    dt = sc.dt
    model = scip.Model("rh_path_planning")
    model.setParam("display/verblevel", 0)
    model.setParam("separating/maxrounds", 0)
    model.setParam("presolving/maxrestarts", 0)
    model.setParam("timing/clocktype", 2)
    if time_limit:
        model.setParam("limits/time", float(time_limit))

    x = []
    for i in range(lay.n_vars):
        lo = lay.lb[i] if np.isfinite(lay.lb[i]) else -model.infinity()
        hi = lay.ub[i] if np.isfinite(lay.ub[i]) else model.infinity()
        x.append(model.addVar(name=f"x{i}", vtype="B" if lay.binary[i] else "C",
                              lb=lo, ub=hi))
    model.setObjective(scip.quicksum(float(lay.obj[i]) * x[i] for i in range(lay.n_vars)),
                       "minimize")

    S, U, A = lay.state, lay.inputs, lay.input_abs
    for k in range(K):
        for n in range(N):
            for i in range(2):
                model.addCons(x[U[k, i, n]] <= x[A[k, i, n]])
                model.addCons(-x[U[k, i, n]] <= x[A[k, i, n]])

    # Double integrator, zero-order hold.
    for k in range(K):
        for n in range(N):
            for i in range(2):
                model.addCons(x[S[k, i, n + 1]] == 1.0 * x[S[k, i, n]]
                              + dt * x[S[k, i + 2, n]]
                              + 0.5 * dt**2 * x[U[k, i, n]])
            for i in range(2, 4):
                model.addCons(x[S[k, i, n + 1]] == 1.0 * x[S[k, i, n]]
                              + dt * x[U[k, i - 2, n]])

    for k in range(K):
        for i in range(4):
            model.addCons(x[S[k, i, 0]] == sc.start[k, i])

    # Soft terminal constraint: |x[N] - goal| <= terminal slack.
    for k in range(K):
        for i in range(4):
            t = x[lay.terminal[k, i]]
            model.addCons(x[S[k, i, N]] - sc.goal[k, i] <= t)
            model.addCons(sc.goal[k, i] - x[S[k, i, N]] <= t)

    M = float(sc.obstacle_big_m)
    for k in range(K):
        for n in range(1, N + 1):
            px, py = x[S[k, 0, n]], x[S[k, 1, n]]
            for o, rect in enumerate(sc.obstacles):
                xmin = rect[0] - sc.obstacle_buffer
                xmax = rect[1] + sc.obstacle_buffer
                ymin = rect[2] - sc.obstacle_buffer
                ymax = rect[3] + sc.obstacle_buffer
                b = [x[j] for j in lay.sides[k, o, n - 1]]
                model.addCons(px <= xmin - FACE_EPS + M * b[0])
                model.addCons(px >= xmax + FACE_EPS - M * b[1])
                model.addCons(py <= ymin - FACE_EPS + M * b[2])
                model.addCons(py >= ymax + FACE_EPS - M * b[3])
                model.addCons(b[0] + b[1] + b[2] + b[3] <= 3)

    d = float(sc.safe_separation)
    M = float(sc.separation_big_m)
    for p, (i, j) in enumerate(lay.pairs):
        for n in range(1, N + 1):
            xi, yi = x[S[i, 0, n]], x[S[i, 1, n]]
            xj, yj = x[S[j, 0, n]], x[S[j, 1, n]]
            b = [x[c] for c in lay.pair_sides[p, n - 1]]
            model.addCons(xi <= xj - d + M * b[0])
            model.addCons(xi >= xj + d - M * b[1])
            model.addCons(yi <= yj - d + M * b[2])
            model.addCons(yi >= yj + d - M * b[3])
            model.addCons(b[0] + b[1] + b[2] + b[3] <= 3)

    return model, x


def apply_profile(model, profile):
    """Solver settings.

    ``strict``   the paper's setting: presolve and cut separation off, tight
                 feasibility tolerances so near-integral binaries cannot be
                 amplified by big-M into infeasible trajectories.
    ``default``  SCIP defaults, used only for the sensitivity check.
    ``labels``   the setting under which the training labels were solved.
    """
    if profile == "strict":
        model.setParam("limits/time", TIME_LIMIT)
        model.setSeparating(scip.SCIP_PARAMSETTING.OFF)
        model.setPresolve(scip.SCIP_PARAMSETTING.OFF)
        model.setParam("numerics/feastol", 1e-9)
        model.setParam("numerics/dualfeastol", 1e-9)
    elif profile == "default":
        model.resetParams()
        model.setParam("limits/time", TIME_LIMIT)
    elif profile != "labels":
        raise ValueError(f"unknown solver profile {profile!r}")
    model.hideOutput(True)


def solve(lay: Layout, fixed_sides=None, profile="strict", keep_values=False):
    """Build and solve, optionally with some obstacle-side groups fixed.

    ``fixed_sides`` is a pair ``(bins, side)``: ``bins`` holds the four face
    binaries of each fixed group and ``side`` the face to enforce.  The
    enforced face gets ``b = 0`` and the other three ``b = 1``.  Fixing only
    narrows variable bounds, so any plan found is feasible for the full MILP.
    """
    start = time.perf_counter()
    model, x = build(lay)
    apply_profile(model, profile)
    if fixed_sides is not None:
        bins, chosen = fixed_sides
        for group, side in zip(bins, chosen):
            for face, column in enumerate(group):
                value = 0 if face == side else 1
                model.chgVarLb(x[column], value)
                model.chgVarUb(x[column], value)
    model.optimize()
    seconds = time.perf_counter() - start

    status = str(model.getStatus())
    found = model.getNSols() > 0
    objective = float(model.getObjVal()) if found else None
    values = None
    if found and keep_values:
        best = model.getBestSol()
        values = np.array([model.getSolVal(best, v) for v in x])
    nodes = int(model.getNNodes())
    model.freeProb()
    return Solution(status, objective, seconds, nodes, values)


def same_objective(reference, candidate, tol=1e-4):
    """The paper's match criterion against the full-MILP optimum."""
    return candidate is not None and abs(candidate - reference) <= tol * (abs(reference) + 1.0)
