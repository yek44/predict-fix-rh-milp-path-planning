"""Predict, fix the most confident groups, and recover when that fails."""

import time
from dataclasses import dataclass, field

import numpy as np

from . import milp
from .features import side_features

SCHEDULE = (0.2, 0.3)  # free fractions tried before the full MILP


def fixed_groups(score, rho):
    """Indices of the groups to fix: all but the round(rho * G) lowest scores."""
    n_free = int(round(rho * len(score)))
    free = set(np.argsort(score)[:n_free].tolist()) if n_free else set()
    return [g for g in range(len(score)) if g not in free]


@dataclass
class Prediction:
    groups: np.ndarray       # (G, 4) face-binary columns
    side: np.ndarray         # (G,) predicted face
    confidence: np.ndarray   # (G,) its softmax probability
    feature_s: float
    inference_s: float

    @property
    def seconds(self):
        return self.feature_s + self.inference_s


@dataclass
class Plan:
    solution: milp.Solution      # the plan that is applied
    rho: float                   # free fraction that produced it, 1.0 = full MILP
    seconds: float               # prediction plus every solve attempted
    attempts: list = field(default_factory=list)  # (rho, Solution) in order

    @property
    def first_attempt_found(self):
        return self.attempts[0][1].found


def predict(lay, predictor):
    start = time.perf_counter()
    sequences, groups = side_features(lay)
    feature_s = time.perf_counter() - start
    side, confidence, inference_s = predictor(sequences)
    return Prediction(groups, side, confidence, feature_s, inference_s)


def solve_residual(lay, pred, rho, score=None, profile="strict", keep_values=False):
    """Fix the selected groups to their predicted side and solve the rest.

    ``score`` ranks the groups; it defaults to the prediction confidence.
    """
    score = pred.confidence if score is None else score
    keep = fixed_groups(score, rho)
    return milp.solve(lay, (pred.groups[keep], pred.side[keep]),
                      profile=profile, keep_values=keep_values)


def plan(lay, predictor, schedule=SCHEDULE, profile="strict", full=None,
         keep_values=False, score=None):
    """The deployed method.

    Try each free fraction in ``schedule`` and accept the first residual
    problem that returns a feasible plan.  If none does, solve the full MILP.
    Pass ``full`` when the full solve has already been timed on this
    instance; its time is then charged instead of solving again.
    """
    pred = predict(lay, predictor)
    seconds = pred.seconds
    attempts = []
    for rho in schedule:
        sol = solve_residual(lay, pred, rho, score, profile, keep_values)
        seconds += sol.seconds
        attempts.append((rho, sol))
        if sol.found:
            return Plan(sol, rho, seconds, attempts)
    if full is None:
        full = milp.solve(lay, profile=profile, keep_values=keep_values)
    attempts.append((1.0, full))
    return Plan(full, 1.0, seconds + full.seconds, attempts)
