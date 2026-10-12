"""Straggler simulation: clients that compute at different speeds.

Real cross-silo federations are synchronous -- the server waits for the round's
updates -- so a slow hospital ("straggler") either (a) returns a PARTIAL update
computed in fewer local steps, or (b) misses the deadline and is DROPPED from the
round.  Both are simulated here; no wall-clock is actually wasted.

Model
-----
Every client i has a relative compute speed ``s_i`` (1.0 = nominal: it finishes
``local_updates`` steps in exactly one round budget).  The server grants a budget of
``straggler_deadline * local_updates`` nominal step-times per round, so client i can
complete

    capacity_i = floor(deadline * local_updates * s_i)   local steps.

mode="partial"  steps_i = clip(capacity_i, straggler_min_updates, local_updates)
mode="drop"     steps_i = local_updates if capacity_i >= local_updates else 0 (dropped)
mode="none"     steps_i = local_updates for everyone (speeds still reported)

Simulated round time (in nominal step-times) = max over clients that finished of
steps_i / s_i, or the full budget when at least one client was cut off.  Accumulating
it gives accuracy-vs-simulated-time curves alongside accuracy-vs-rounds.

Speed profiles
--------------
tiers      cycle ``straggler_tiers`` over the sub-clients WITHIN each centre, e.g.
           1.0/0.5/0.25 -> every hospital owns one fast, one medium and one slow
           client.  Straggling is then orthogonal to the ground-truth clustering, so
           it cannot be mistaken for the cluster structure CFL is meant to find.
lognormal  s = exp(N(0, sigma)), clipped to [0.05, 4]
uniform    s = 1 + U(-sigma, +sigma), clipped to [0.05, 4]
manual     ``straggler_speeds`` = one comma-separated speed per client

``straggler_dynamic=False`` (default) fixes each client's speed for the whole run
(systems heterogeneity: a hospital's hardware does not change).  ``True`` resamples
every round (transient load).  Everything is driven by ``straggler_seed``, which is
independent of the data seed, so the straggler pattern can be changed without
reshuffling the partition.

Effect on CFL worth remembering: a client that runs fewer steps produces a SMALLER
||dtheta_i||, which feeds Eq. 33 (max_i ||dtheta_i|| > eps_2).  Stragglers therefore
look "converged" to the splitting criteria.  ``steps_done`` is logged every round so
this can be checked rather than assumed.
"""
import numpy as np


class StragglerSchedule:
    """Per-round plan of how many local steps each client gets to run."""

    def __init__(self, cfg, n_clients, clients_per_center=None):
        self.mode = cfg.straggler_mode
        self.profile = cfg.straggler_profile
        self.n = n_clients
        self.per_center = clients_per_center or cfg.clients_per_center
        self.deadline = float(cfg.straggler_deadline)
        self.min_updates = int(cfg.straggler_min_updates)
        self.dynamic = bool(cfg.straggler_dynamic)
        self.sigma = float(cfg.straggler_sigma)
        self.seed = int(cfg.straggler_seed)
        self.tiers = [float(t) for t in str(cfg.straggler_tiers).split(",") if t != ""]
        self.manual = [float(s) for s in str(cfg.straggler_speeds).split(",") if s != ""]
        if self.profile == "manual" and len(self.manual) != n_clients:
            raise ValueError("straggler_speeds needs exactly {} values, got {}".format(
                n_clients, len(self.manual)))
        self._fixed = None if self.dynamic else self._draw(0)

    # ------------------------------------------------------------------ speeds
    def _draw(self, rnd):
        if self.mode == "none":
            return np.ones(self.n)
        if self.profile == "manual":
            return np.array(self.manual, dtype=float)
        if self.profile == "tiers":
            # position inside the client's own centre decides the tier
            return np.array([self.tiers[(i % self.per_center) % len(self.tiers)]
                             for i in range(self.n)], dtype=float)
        rng = np.random.default_rng([self.seed, rnd])
        if self.profile == "lognormal":
            s = np.exp(rng.normal(0.0, self.sigma, self.n))
        elif self.profile == "uniform":
            s = 1.0 + rng.uniform(-self.sigma, self.sigma, self.n)
        else:
            raise ValueError("unknown straggler_profile '{}'".format(self.profile))
        return np.clip(s, 0.05, 4.0)

    def speeds(self, rnd):
        return self._draw(rnd) if self.dynamic else self._fixed

    # -------------------------------------------------------------------- plan
    def plan(self, rnd, local_updates):
        """-> dict(steps, dropped, speeds, budget, round_time).

        ``steps[i]`` is how many local SGD steps client i runs this round;
        ``dropped[i]`` True means it contributes no update at all.
        """
        s = self.speeds(rnd)
        budget = self.deadline * local_updates
        capacity = np.floor(budget * s).astype(int)
        steps = np.full(self.n, local_updates, dtype=int)
        dropped = np.zeros(self.n, dtype=bool)

        if self.mode == "partial":
            steps = np.clip(capacity, self.min_updates, local_updates)
        elif self.mode == "drop":
            late = capacity < local_updates
            steps = np.where(late, 0, local_updates)
            dropped = late
        elif self.mode != "none":
            raise ValueError("straggler_mode must be 'none', 'partial' or 'drop'")

        finished = ~dropped
        times = np.where(finished, steps / np.maximum(s, 1e-9), 0.0)
        round_time = float(budget if dropped.any() else (times.max() if self.n else 0.0))
        return {"steps": steps.tolist(), "dropped": dropped.tolist(),
                "speeds": [float(v) for v in s], "budget": float(budget),
                "round_time": round_time}

    def describe(self):
        s = self.speeds(0)
        return "straggler mode={} profile={} deadline={} dynamic={} speeds={}".format(
            self.mode, self.profile if self.mode != "none" else "-", self.deadline,
            self.dynamic, [round(float(v), 3) for v in s])
