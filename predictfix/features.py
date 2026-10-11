"""Inputs and targets for SideGRU.

For every vehicle-obstacle pair the network sees one sequence over the
horizon steps n = 1..N.  Each step carries the 28 features of Table I in the
paper: problem data plus straight-line and constant-velocity guesses of where
the vehicle will be.  None of them needs a solver.

The target of a step is the face with the largest signed clearance on an
optimal trajectory (ties go to the earlier face in left, right, below, above).
"""

import numpy as np

FEATURE_NAMES = [
    "tau", "t",
    "start_x", "start_y", "start_vx", "start_vy", "goal_x", "goal_y",
    "obs_xmin", "obs_xmax", "obs_ymin", "obs_ymax", "obs_cx", "obs_cy", "obs_w", "obs_h",
    "line_x", "line_y", "cv_x", "cv_y",
    "line_minus_xmin", "xmax_minus_line", "line_minus_ymin", "ymax_minus_line",
    "line_minus_cx", "line_minus_cy",
    "goal_dx", "goal_dy",
]


def _inflated(rect, buffer):
    return rect[0] - buffer, rect[1] + buffer, rect[2] - buffer, rect[3] + buffer


def side_features(lay):
    """Feature sequences and the matching obstacle-side groups.

    Returns ``sequences``, one ``(N, 28)`` array per vehicle-obstacle pair,
    and ``groups``, the face-binary columns ``(K * L * N, 4)`` of every group
    in the same order (vehicle, obstacle, step).
    """
    sc = lay.scenario
    N = lay.horizon_steps
    sequences = []
    for k in range(sc.n_vehicles):
        x0, y0, vx0, vy0 = sc.start[k]
        gx, gy = sc.goal[k, 0], sc.goal[k, 1]
        for rect in sc.obstacles:
            xmin, xmax, ymin, ymax = _inflated(rect, sc.obstacle_buffer)
            cx, cy = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
            w, h = xmax - xmin, ymax - ymin
            rows = []
            for n in range(1, N + 1):
                tau = n / max(1, N)
                lx, ly = x0 + tau * (gx - x0), y0 + tau * (gy - y0)
                bx, by = x0 + n * sc.dt * vx0, y0 + n * sc.dt * vy0
                rows.append([
                    tau, n * sc.dt,
                    x0, y0, vx0, vy0, gx, gy,
                    xmin, xmax, ymin, ymax, cx, cy, w, h,
                    lx, ly, bx, by,
                    lx - xmin, xmax - lx, ly - ymin, ymax - ly,
                    lx - cx, ly - cy,
                    gx - x0, gy - y0,
                ])
            sequences.append(np.asarray(rows, dtype=np.float32))
    groups = lay.sides.reshape(-1, 4)
    return sequences, groups


def side_labels(lay, x):
    """Targets from an optimal solution ``x``, shaped like the sequences.

    Returns ``labels`` ``(K * L, N)`` and ``valid`` ``(K * L, N, 4)``, where
    ``valid`` marks every face that the trajectory actually clears.
    """
    sc = lay.scenario
    N = lay.horizon_steps
    labels, valid = [], []
    for k in range(sc.n_vehicles):
        px = x[lay.state[k, 0, 1:]]
        py = x[lay.state[k, 1, 1:]]
        for rect in sc.obstacles:
            xmin, xmax, ymin, ymax = _inflated(rect, sc.obstacle_buffer)
            clearance = np.stack([xmin - px, px - xmax, ymin - py, py - ymax],
                                 axis=1).astype(np.float32)
            label = clearance.argmax(axis=1)
            ok = clearance >= -1e-5
            ok[np.arange(N), label] |= ~ok.any(axis=1)
            labels.append(label)
            valid.append(ok)
    return np.asarray(labels, dtype=np.int64), np.asarray(valid, dtype=bool)
