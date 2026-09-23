"""Weight-conditioned PPO with five projected Lagrange multipliers.

Rollouts batch surrogate calls on CPU. PyTorch handles policy/value updates.
Raw Gaussian actions are stored for exact tanh-transformed log probabilities.
"""

import logging
import math
import time

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as F

from .env import BatchEnvironment

LOGGER = logging.getLogger(__name__)


def gae(rewards, values, dones, last_values, gamma, lam):
    """GAE over (time, environments[, costs]); terminals break all recurrences."""
    advantages = np.zeros_like(rewards, dtype=np.float32)
    running = np.zeros_like(last_values, dtype=np.float32)
    for t in reversed(range(len(rewards))):
        next_value = last_values if t == len(rewards) - 1 else values[t + 1]
        mask = 1.0 - dones[t].astype(np.float32)
        while mask.ndim < rewards[t].ndim:
            mask = mask[..., None]
        delta = rewards[t] + gamma * next_value * mask - values[t]
        running = delta + gamma * lam * mask * running
        advantages[t] = running
    return advantages, advantages + values


def transformed_log_prob(distribution, raw_action):
    # Stable log(1 - tanh(u)^2), including very large |u|.
    correction = 2 * (math.log(2) - raw_action - F.softplus(-2 * raw_action))
    return (distribution.log_prob(raw_action) - correction).sum(-1)


class ActorCritic(nn.Module):
    def __init__(self, observation_dim, action_dim, hidden=64):
        super().__init__()

        def mlp(out):
            return nn.Sequential(nn.Linear(observation_dim, hidden), nn.Tanh(),
                                 nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, out))

        self.actor, self.reward_critic, self.cost_critics = mlp(action_dim), mlp(1), mlp(5)
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        nn.init.zeros_(self.actor[-1].bias)

    def forward(self, observations):
        distribution = Normal(self.actor(observations), self.log_std.clamp(-5, 1).exp())
        return distribution, self.reward_critic(observations).squeeze(-1), self.cost_critics(observations)


def train(problem, config, output_dir, surrogate_sha256=None):
    torch.manual_seed(config.seed)
    if config.device == "cuda" and not torch.cuda.is_available():
        LOGGER.warning("CUDA unavailable; using CPU")
    device = torch.device("cuda" if config.device == "cuda" and torch.cuda.is_available() else "cpu")
    torch.set_num_threads(1)
    rng = np.random.default_rng(config.seed)
    env = BatchEnvironment(problem, config)
    model = ActorCritic(problem.n_var + 5, problem.n_var, config.hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    multipliers = np.zeros(5, dtype=np.float32)
    history = []
    started = time.perf_counter()
    batch_count = config.n_envs * config.rollout_steps
    updates = config.total_steps // batch_count

    def tensor(x):
        return torch.as_tensor(x, dtype=torch.float32, device=device)

    for update in range(updates):
        buffer = {key: [] for key in ("obs", "action", "logp", "reward", "cost", "done", "vr", "vc")}
        feasible = []
        for _ in range(config.rollout_steps):
            obs = env.observe()
            with torch.no_grad():
                dist, vr, vc = model(tensor(obs))
                raw_action = dist.sample()
                logp = transformed_log_prob(dist, raw_action)
            _, reward, cost, done, result = env.step(torch.tanh(raw_action).cpu().numpy())
            values = (obs, raw_action.cpu().numpy(), logp.cpu().numpy(), reward, cost,
                      done, vr.cpu().numpy(), vc.cpu().numpy())
            for key, value in zip(buffer, values):
                buffer[key].append(value)
            feasible.append(result.feasible)
            env.reset(done)
        b = {key: np.asarray(value) for key, value in buffer.items()}
        with torch.no_grad():
            _, last_r, last_c = model(tensor(env.observe()))
        ar, rr = gae(b["reward"], b["vr"], b["done"], last_r.cpu().numpy(), config.gamma, config.gae_lambda)
        ac, rc = gae(b["cost"], b["vc"], b["done"], last_c.cpu().numpy(), config.gamma, config.gae_lambda)
        advantage = ar - (ac * multipliers).sum(-1)
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        obs_t = tensor(b["obs"].reshape(-1, problem.n_var + 5))
        action_t = tensor(b["action"].reshape(-1, problem.n_var))
        old_logp = tensor(b["logp"].reshape(-1))
        adv_t, rr_t, rc_t = tensor(advantage.reshape(-1)), tensor(rr.reshape(-1)), tensor(rc.reshape(-1, 5))
        losses, divergences = [], []
        for _ in range(config.ppo_epochs):
            permutation = rng.permutation(batch_count)
            for start in range(0, batch_count, config.batch_size):
                idx = permutation[start:start + config.batch_size]
                dist, vr, vc = model(obs_t[idx])
                logp = transformed_log_prob(dist, action_t[idx])
                log_ratio = logp - old_logp[idx]
                ratio = log_ratio.exp()
                clipped = torch.clamp(ratio, 1 - config.clip_ratio, 1 + config.clip_ratio)
                actor_loss = -torch.minimum(ratio * adv_t[idx], clipped * adv_t[idx]).mean()
                value_loss = F.mse_loss(vr, rr_t[idx]) + F.mse_loss(vc, rc_t[idx])
                # Monte Carlo entropy of the transformed policy, not the base Gaussian.
                entropy = -transformed_log_prob(dist, dist.rsample()).mean()
                loss = actor_loss + 0.5 * value_loss - config.entropy_coef * entropy
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite PPO loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                losses.append(float(loss.detach()))
                divergences.append(float(((ratio - 1) - log_ratio).mean().detach()))
        mean_cost = b["cost"].mean(axis=(0, 1))
        multipliers = np.clip(multipliers + config.lambda_lr * (mean_cost - config.cost_limit),
                              0, config.max_lambda)
        record = {"steps": (update + 1) * batch_count, "reward": float(b["reward"].mean()),
                  "feasible_rate": float(np.mean(feasible)), "loss": float(np.mean(losses)),
                  "approx_kl": float(np.mean(divergences))}
        record.update({f"lambda_{i}": float(v) for i, v in enumerate(multipliers)})
        record.update({f"cost_{i}": float(v) for i, v in enumerate(mean_cost)})
        history.append(record)
        LOGGER.info("PPO %d/%d steps | reward %.3f | feasible %.1f%%",
                    record["steps"], config.total_steps, record["reward"], 100 * record["feasible_rate"])
    torch.save({"state_dict": model.cpu().state_dict(), "config": config.to_dict(),
                "multipliers": multipliers.tolist(), "observation_dim": problem.n_var + 5,
                "action_dim": problem.n_var, "surrogate_sha256": surrogate_sha256}, output_dir / "policy.pt")
    return model, history, time.perf_counter() - started


def load_policy(path):
    # weights_only avoids general Python object deserialization for policy files.
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = ActorCritic(checkpoint["observation_dim"], checkpoint["action_dim"],
                        checkpoint["config"]["hidden"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def evaluate_policy(model, problem, config, seed=None):
    model = model.cpu().eval()
    env = BatchEnvironment(problem, config, count=config.eval_episodes,
                           seed=config.seed + 10000 if seed is None else seed)
    decisions, latencies = [], []
    with torch.no_grad():
        # Warm-up is excluded from latency statistics.
        model(torch.from_numpy(env.observe()))
        for _ in range(config.horizon):
            obs = torch.from_numpy(env.observe())
            start = time.perf_counter()
            dist, _, _ = model(obs)
            action = torch.tanh(dist.mean).numpy()
            latencies.append(1000 * (time.perf_counter() - start))
            env.step(action)
            decisions.append(env.x.copy())
    return np.concatenate(decisions), {
        "policy_batch_size": config.eval_episodes,
        "policy_batch_ms_p50": float(np.percentile(latencies, 50)),
        "policy_batch_ms_p95": float(np.percentile(latencies, 95)),
        "latency_scope": "CPU policy forward pass, excludes surrogate and I/O",
    }
