"""Planning scenarios: start and goal states, rectangular obstacles, limits.

All instances in the paper live on one workspace and share the vehicle
limits below.  The main experiments keep the 13-obstacle map fixed and draw
the start and goal at random; the transfer study also draws the map.
"""

from dataclasses import dataclass, field, replace

import numpy as np

DT = 0.1
HORIZON = 4.0

POS_BOUNDS = ((-2.5, 16.0), (-7.5, 11.0))
VEL_BOUNDS = ((-5.0, 5.0), (-5.0, 5.0))
INPUT_BOUNDS = ((-3.0, 3.0), (-3.0, 3.0))

# The training map: [xmin, xmax, ymin, ymax] per obstacle.
CANONICAL_MAP = np.array([
    [-1.5, 1.0, -6.5, -2.5],
    [-1.5, 1.0, -1.5, 5.0],
    [-1.5, 1.0, 6.0, 10.0],
    [2.5, 5.5, -6.5, -2.5],
    [2.5, 5.5, 2.0, 6.0],
    [7.0, 10.5, -6.5, -2.5],
    [7.0, 10.5, 2.0, 6.0],
    [12.0, 15.0, -6.5, -2.5],
    [12.0, 15.0, 2.0, 6.0],
    [2.5, 7.0, 7.5, 10.0],
    [8.5, 15.0, 7.5, 10.0],
    [2.5, 7.0, -1.5, 1.0],
    [8.5, 15.0, -1.5, 1.0],
])

# Obstacle side lengths and placement attempts for the random-map regimes.
SPARSE_MAPS = (1.5, 4.0, 200)
DENSE_MAPS = (2.0, 4.5, 3000)


@dataclass(frozen=True)
class Scenario:
    """One planning problem.

    ``start`` and ``goal`` hold one row ``[px, py, vx, vy]`` per vehicle and
    ``obstacles`` one row ``[xmin, xmax, ymin, ymax]`` per rectangle.
    Vehicle-vehicle separation is enforced when there is more than one
    vehicle and ``safe_separation`` is positive.
    """

    start: np.ndarray
    goal: np.ndarray
    obstacles: np.ndarray
    horizon_steps: int = int(round(HORIZON / DT))
    dt: float = DT
    pos_bounds: tuple = POS_BOUNDS
    vel_bounds: tuple = VEL_BOUNDS
    input_bounds: tuple = INPUT_BOUNDS
    obstacle_buffer: float = 0.05
    obstacle_big_m: float = 800.0
    safe_separation: float = 0.0
    separation_big_m: float = 25.0
    pair_margin: float = 1.0
    terminal_weights: tuple = field(default=(50.0, 10.0))  # position, velocity

    @property
    def n_vehicles(self) -> int:
        return int(self.start.shape[0])

    def advanced(self, state: np.ndarray) -> "Scenario":
        """The same problem re-posed from a new measured state."""
        return replace(self, start=np.asarray(state, dtype=float))


def _workspace():
    (x_lo, x_hi), (y_lo, y_hi) = POS_BOUNDS
    return x_lo, x_hi, y_lo, y_hi


def _inside(px, py, rect, margin=0.0):
    return (rect[0] - margin <= px <= rect[1] + margin
            and rect[2] - margin <= py <= rect[3] + margin)


def _overlap(r1, r2, gap=0.0):
    return not (r1[1] + gap < r2[0] or r2[1] + gap < r1[0]
                or r1[3] + gap < r2[2] or r2[3] + gap < r1[2])


def _min_distance(points):
    best = float("inf")
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            best = min(best, float(np.linalg.norm(points[i] - points[j])))
    return best


def sample_start_goal(rng, n_vehicles, obstacles, min_pair_dist=0.45,
                      obstacle_margin=0.20, border_margin=0.5,
                      max_attempts=1000):
    """Uniform start and goal positions at rest, clear of every obstacle."""
    x_lo, x_hi, y_lo, y_hi = _workspace()
    x_lo, x_hi = x_lo + border_margin, x_hi - border_margin
    y_lo, y_hi = y_lo + border_margin, y_hi - border_margin

    def blocked(*points):
        return any(_inside(px, py, rect, obstacle_margin)
                   for rect in obstacles for px, py in points)

    starts = np.zeros((n_vehicles, 4))
    goals = np.zeros((n_vehicles, 4))
    for _ in range(max_attempts):
        ok = True
        for v in range(n_vehicles):
            sx, sy = rng.uniform(x_lo, x_hi), rng.uniform(y_lo, y_hi)
            gx, gy = rng.uniform(x_lo, x_hi), rng.uniform(y_lo, y_hi)
            if blocked((sx, sy), (gx, gy)):
                ok = False
                break
            starts[v, :2] = sx, sy
            goals[v, :2] = gx, gy
        if ok and _min_distance(np.vstack([starts[:, :2], goals[:, :2]])) >= min_pair_dist:
            return starts, goals

    # Give up on spacing between vehicles, but still keep clear of obstacles.
    v = 0
    while v < n_vehicles:
        sx, sy = rng.uniform(x_lo, x_hi), rng.uniform(y_lo, y_hi)
        gx, gy = rng.uniform(x_lo, x_hi), rng.uniform(y_lo, y_hi)
        if not blocked((sx, sy), (gx, gy)):
            starts[v, :2] = sx, sy
            goals[v, :2] = gx, gy
            v += 1
    return starts, goals


def random_obstacles(rng, n_obstacles, keep_clear, min_size, max_size,
                     point_margin, max_attempts):
    """Non-overlapping axis-aligned rectangles that avoid the given points.

    A rectangle that cannot be placed within ``max_attempts`` is dropped, so
    dense maps occasionally come out one obstacle short.
    """
    x_lo, x_hi, y_lo, y_hi = _workspace()
    placed = []
    for _ in range(n_obstacles):
        for _ in range(max_attempts):
            w = rng.uniform(min_size, max_size)
            h = rng.uniform(min_size, max_size)
            cx = rng.uniform(x_lo + w / 2, x_hi - w / 2)
            cy = rng.uniform(y_lo + h / 2, y_hi - h / 2)
            rect = np.array([cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2])
            if any(_inside(px, py, rect, point_margin) for px, py in keep_clear):
                continue
            if any(_overlap(rect, other, gap=0.15) for other in placed):
                continue
            placed.append(rect)
            break
    return np.array(placed) if placed else np.zeros((0, 4))


def canonical_scenario(rng, horizon=HORIZON):
    """Random start and goal on the training map."""
    starts, goals = sample_start_goal(rng, 1, CANONICAL_MAP)
    return Scenario(starts, goals, CANONICAL_MAP.copy(),
                    horizon_steps=int(round(horizon / DT)))


def canonical_stream(seed, n, skip=0):
    """``n`` canonical scenarios from one seeded stream, after ``skip`` draws."""
    rng = np.random.RandomState(seed)
    for _ in range(skip):
        canonical_scenario(rng)
    return [canonical_scenario(rng) for _ in range(n)]


# Fixed streams.  The training set is the first 1000 draws of seed 13 and the
# validation set the next 100; the held-out test set has its own seed.
def training_scenarios(n=1000):
    return canonical_stream(13, n)


def validation_scenarios(n=100):
    return canonical_stream(13, n, skip=1000)


def test_scenarios(n=100):
    return canonical_stream(555, n)


def unseen_map_scenario(rng, horizon, n_obstacles, sizes=SPARSE_MAPS):
    """A single vehicle on a freshly drawn obstacle map."""
    starts, goals = sample_start_goal(rng, 1, np.zeros((0, 4)))
    min_size, max_size, attempts = sizes
    keep_clear = [(starts[0, 0], starts[0, 1]), (goals[0, 0], goals[0, 1])]
    obstacles = random_obstacles(rng, n_obstacles, keep_clear, min_size,
                                 max_size, point_margin=0.6,
                                 max_attempts=attempts)
    return Scenario(starts, goals, obstacles,
                    horizon_steps=int(round(horizon / DT)))


def crossing_scenario(rng, n_vehicles, n_obstacles=4, horizon=HORIZON,
                      safe_separation=0.5, max_tries=300):
    """Vehicles spread around a circle, each heading for the far side.

    The paths meet near the centre, so the separation constraints bind.
    """
    x_lo, x_hi, y_lo, y_hi = _workspace()
    margin = 1.0
    for _ in range(max_tries):
        cx = rng.uniform(x_lo + 0.4 * (x_hi - x_lo), x_lo + 0.6 * (x_hi - x_lo))
        cy = rng.uniform(y_lo + 0.4 * (y_hi - y_lo), y_lo + 0.6 * (y_hi - y_lo))
        theta0 = rng.uniform(0.0, 2.0 * np.pi)
        legs = []
        for v in range(n_vehicles):
            theta = theta0 + v * np.pi / n_vehicles + rng.uniform(-0.2, 0.2)
            r = rng.uniform(3.5, 5.5)
            d = np.array([np.cos(theta), np.sin(theta)])
            c = np.array([cx, cy])
            legs.append((c - r * d, c + r * d) if v % 2 == 0 else (c + r * d, c - r * d))
        points = [p for leg in legs for p in leg]
        if any(p[0] < x_lo + margin or p[0] > x_hi - margin
               or p[1] < y_lo + margin or p[1] > y_hi - margin for p in points):
            continue
        obstacles = random_obstacles(rng, n_obstacles, [tuple(p) for p in points],
                                     min_size=1.5, max_size=3.0,
                                     point_margin=0.8, max_attempts=400)
        if len(obstacles) < n_obstacles:
            continue
        starts = np.zeros((n_vehicles, 4))
        goals = np.zeros((n_vehicles, 4))
        for v, (s, g) in enumerate(legs):
            starts[v, :2] = s
            goals[v, :2] = g
        return Scenario(starts, goals, obstacles,
                        horizon_steps=int(round(horizon / DT)),
                        safe_separation=safe_separation)
    raise RuntimeError(f"could not draw a {n_vehicles}-vehicle crossing")
