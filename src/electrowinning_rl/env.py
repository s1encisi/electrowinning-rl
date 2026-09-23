"""Gymnasium interface and a batched rollout engine sharing the same transition."""

from typing import ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class BatchEnvironment:
    """Finite-horizon setpoint search, not an electrochemical dynamics model.

    Episode time remaining is observed. A horizon ending is a true task terminal;
    GAE must not bootstrap across it. All decision coordinates, including duration,
    are adjustable, so RL and evolutionary search share exactly the same domain.
    """
    def __init__(self, problem, config, count=None, seed=None):
        self.problem, self.config = problem, config
        self.count = count or config.n_envs
        self.rng = np.random.default_rng(config.seed if seed is None else seed)
        self.x = np.empty((self.count, problem.n_var))
        self.weights = np.empty((self.count, 4))
        self.steps = np.zeros(self.count, dtype=int)
        self.reset()

    def reset(self, mask=None):
        mask = np.ones(self.count, dtype=bool) if mask is None else np.asarray(mask, bool)
        n = int(mask.sum())
        self.x[mask] = self.problem.sample(self.rng, n)
        self.weights[mask] = self.rng.dirichlet(np.ones(4), size=n)
        self.steps[mask] = 0
        return self.observe()

    def observe(self):
        remaining = (1 - self.steps / self.config.horizon)[:, None]
        return np.column_stack([self.problem.normalize(self.x), self.weights, remaining]).astype(np.float32)

    def step(self, actions):
        actions = np.asarray(actions)
        if actions.shape != self.x.shape or not np.isfinite(actions).all():
            raise ValueError(f"Expected finite actions with shape {self.x.shape}")
        if np.any(self.steps >= self.config.horizon):
            raise RuntimeError("Reset terminal environments before stepping")
        delta = self.config.action_fraction * (self.problem.upper - self.problem.lower)
        self.x = np.clip(self.x + np.clip(actions, -1, 1) * delta,
                         self.problem.lower, self.problem.upper)
        self.steps += 1
        result = self.problem.evaluate(self.x)
        reward = -(result.objectives / self.config.process.objective_scale * self.weights).sum(1)
        done = self.steps >= self.config.horizon
        return self.observe(), reward.astype(np.float32), result.costs.astype(np.float32), done, result


class ElectrowinningEnv(gym.Env):
    metadata: ClassVar[dict] = {"render_modes": []}

    def __init__(self, problem, config):
        self.problem, self.config = problem, config
        d = problem.n_var
        self.action_space = spaces.Box(-1., 1., shape=(d,), dtype=np.float32)
        self.observation_space = spaces.Box(
            np.array([-1.] * d + [0.] * 5, np.float32), np.ones(d + 5, np.float32))
        self.engine = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None or self.engine is None:
            actual_seed = seed if seed is not None else int(self.np_random.integers(2**31))
            self.engine = BatchEnvironment(self.problem, self.config, count=1, seed=actual_seed)
        else:
            self.engine.reset()
        return self.engine.observe()[0], {}

    def step(self, action):
        if self.engine is None:
            raise RuntimeError("Call reset() before step()")
        obs, reward, cost, done, result = self.engine.step(np.asarray(action)[None, :])
        return obs[0], float(reward[0]), bool(done[0]), False, {
            "costs": cost[0], "objectives": result.objectives[0],
            "constraints": result.constraints[0], "feasible": bool(result.feasible[0]),
        }
