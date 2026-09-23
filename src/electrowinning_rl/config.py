"""Validated configuration; paths are supplied by the caller, never hard-coded."""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class ProcessConfig:
    cu_limit: float = 8.0
    as_min: float = 4.0
    current_density_max: float = 340.0
    cell_voltage_max: float = 2.5
    cu_as_ratio_max: float = 0.8
    cathodes: int = 35
    cathode_area: float = 2.2638
    cells: int = 16
    copper_price: float = 98.44  # currency/kg; illustrative, not a market quote
    reprocessing_cost: float = 0.14
    feedstock_cost: float = 82.64
    treatment_cost: float = 2.5
    treatment_as_reference: float = 11.64
    electricity_price: float = 0.6  # currency/kWh
    operating_cost: float = 200.0  # currency/h
    objective_scale: list[float] = field(default_factory=lambda: [10., 10., 10000., 500000.])


@dataclass
class Config:
    condition: int = 1
    seed: int = 42
    samples: int = 2400
    trees: int = 64
    split: str = "chronological"
    total_steps: int = 4096
    n_envs: int = 16
    rollout_steps: int = 32
    horizon: int = 24
    hidden: int = 64
    ppo_epochs: int = 4
    batch_size: int = 256
    learning_rate: float = 0.0003
    gamma: float = 0.97
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    entropy_coef: float = 0.005
    lambda_lr: float = 0.5
    cost_limit: float = 0.02  # mean normalized violation per constraint per transition
    max_lambda: float = 50.0
    action_fraction: float = 0.1
    eval_episodes: int = 32
    population: int = 64
    generations: int = 24
    device: str = "cpu"
    process: ProcessConfig = field(default_factory=ProcessConfig)

    def validate(self):
        if self.condition not in (1, 2, 3):
            raise ValueError("condition must be 1, 2 or 3")
        for name in ("samples", "trees", "total_steps", "n_envs", "rollout_steps", "horizon",
                     "hidden", "ppo_epochs", "batch_size", "eval_episodes", "population", "generations"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.samples < 100 or self.population < 4:
            raise ValueError("samples must be >=100 and population >=4")
        if self.total_steps % (self.n_envs * self.rollout_steps):
            raise ValueError("total_steps must be divisible by n_envs * rollout_steps")
        if self.n_envs * self.rollout_steps < 2:
            raise ValueError("A rollout needs at least two transitions")
        if self.split not in ("chronological", "random"):
            raise ValueError("split must be chronological or random")
        if self.device not in ("cpu", "cuda"):
            raise ValueError("device must be cpu or cuda")
        for name in ("gamma", "gae_lambda", "clip_ratio", "action_fraction"):
            if not 0 < getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in (0, 1]")
        for name in ("learning_rate", "lambda_lr", "max_lambda"):
            if not getattr(self, name) > 0:
                raise ValueError(f"{name} must be positive")
        if self.cost_limit < 0 or self.entropy_coef < 0:
            raise ValueError("cost_limit and entropy_coef must be nonnegative")
        p = self.process
        for name in ("cu_limit", "as_min", "current_density_max", "cell_voltage_max",
                     "cu_as_ratio_max", "cathodes", "cathode_area", "cells"):
            if not getattr(p, name) > 0:
                raise ValueError(f"process.{name} must be positive")
        if len(p.objective_scale) != 4 or any(x <= 0 for x in p.objective_scale):
            raise ValueError("process.objective_scale must contain four positive numbers")
        # Reject NaN/Infinity in JSON and all numeric configuration values.
        json.dumps(asdict(self), allow_nan=False)
        return self

    @classmethod
    def load(cls, path=None, **overrides):
        values = json.loads(Path(path).read_text(encoding="utf-8")) if path else {}
        values.update({k: v for k, v in overrides.items() if v is not None})
        if isinstance(values.get("process"), dict):
            values["process"] = ProcessConfig(**values["process"])
        return cls(**values).validate()

    def to_dict(self):
        return asdict(self)
