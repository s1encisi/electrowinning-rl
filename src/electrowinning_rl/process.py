"""Shared process contract for RL, random search and NSGA-II.

Objectives use minimization convention: Cu, -As, energy, -profit.
Positive G means a constraint violation. Economic values are scenario assumptions.
"""

from dataclasses import dataclass

import numpy as np

from .config import ProcessConfig

TARGETS = ["cu_out", "as_out", "voltage"]
OBJECTIVES = ["cu_out", "negative_as_out", "energy_kwh", "negative_profit"]
CONSTRAINTS = ["copper", "arsenic", "current_density", "cell_voltage", "copper_arsenic_ratio"]


def domain(condition):
    if condition == 1:
        names = ["cu_in", "temperature_a", "current_a", "flow_a", "duration"]
        lo, hi = [29, 48, 8000, 111, 2], [55, 65, 27000, 123, 8]
    elif condition == 2:
        names = ["cu_in", "temperature_b", "current_b", "flow_b", "duration"]
        lo, hi = [29, 55, 8000, 113, 2], [55, 65, 27000, 123, 8]
    elif condition == 3:
        names = ["cu_in", "temperature_a", "current_a", "flow_a",
                 "temperature_b", "current_b", "flow_b", "duration"]
        lo = [29, 40, 8000, 111, 40, 8000, 113, 2]
        hi = [55, 65, 27000, 123, 65, 27000, 123, 8]
    else:
        raise ValueError("condition must be 1, 2 or 3")
    return names, np.array(lo, dtype=float), np.array(hi, dtype=float)


def synthetic_response(x, condition):
    """An explicit artificial response surface, NOT a calibrated process simulator.

    This function has no dependency on original measurements or fitted weights.
    Different output sensitivities create feasible/infeasible regions and trade-offs.
    """
    _, lo, hi = domain(condition)
    z = (np.asarray(x) - lo) / (hi - lo)
    feed, duration = z[:, 0], z[:, -1]
    temperature, current, flow = z[:, 1], z[:, 2], z[:, 3]
    if condition == 3:
        temperature = (temperature + z[:, 4]) / 2
        current = (current + z[:, 5]) / 2
        flow = (flow + z[:, 6]) / 2
    cu = 9.5 + 1.5 * feed - 4.0 * current - 1.2 * temperature - 1.5 * duration + 0.6 * flow
    arsenic = 9.2 - 2.2 * current + 0.8 * feed - 0.4 * duration + 0.2 * temperature
    voltage = 24.0 + 16.0 * current - 3.0 * temperature + 0.8 * feed + 0.5 * np.sin(np.pi * flow)
    return np.column_stack([cu, arsenic, voltage])


@dataclass
class Evaluation:
    objectives: np.ndarray
    constraints: np.ndarray
    costs: np.ndarray
    outputs: np.ndarray

    @property
    def feasible(self):
        return np.all(self.constraints <= 1e-8, axis=1)


class ProcessProblem:
    def __init__(self, condition, surrogate, config=None):
        self.condition, self.surrogate = condition, surrogate
        self.config = config or ProcessConfig()
        self.names, self.lower, self.upper = domain(condition)
        self.n_var = len(self.names)
        self.evaluations = 0

    def sample(self, rng, count):
        return rng.uniform(self.lower, self.upper, size=(count, self.n_var))

    def normalize(self, x):
        return 2 * (x - self.lower) / (self.upper - self.lower) - 1

    def evaluate(self, x):
        x = np.atleast_2d(np.asarray(x, dtype=float))
        if x.shape[1] != self.n_var or not np.isfinite(x).all():
            raise ValueError(f"Expected finite decisions of shape (N, {self.n_var})")
        if np.any(x < self.lower - 1e-6) or np.any(x > self.upper + 1e-6):
            raise ValueError("Decisions are outside the configured process domain")
        outputs = np.asarray(self.surrogate.predict(x), dtype=float)
        if outputs.shape != (len(x), 3) or not np.isfinite(outputs).all():
            raise ValueError("Surrogate must return finite (N, 3) outputs: Cu, As, voltage")
        if np.any(outputs < 0):
            raise ValueError("Negative concentrations/voltage are invalid process predictions")
        self.evaluations += len(x)
        cu, arsenic, voltage = outputs.T
        p = self.config
        current, flow, duration = x[:, 2], x[:, 3], x[:, -1]
        if self.condition == 3:
            # Both algorithms use the same average-flow convention and equal unit voltages.
            power = voltage * (current + x[:, 5])
            density = np.maximum(current, x[:, 5]) / (p.cathodes * p.cathode_area)
            flow = (flow + x[:, 6]) / 2
        else:
            power = voltage * current
            density = current / (p.cathodes * p.cathode_area)
        energy = power * duration / 1000  # V * A * h -> kWh
        recovered_kg = np.maximum(0, x[:, 0] - cu) * flow * duration  # g/L == kg/m^3
        margin = p.copper_price - p.reprocessing_cost - p.feedstock_cost
        treatment = p.treatment_cost * (p.treatment_as_reference + cu) * flow * duration
        profit = margin * recovered_kg - treatment - p.electricity_price * energy - p.operating_cost * duration
        f = np.column_stack([cu, -arsenic, energy, -profit])
        g = np.column_stack([
            cu - p.cu_limit, p.as_min - arsenic, density - p.current_density_max,
            voltage / p.cells - p.cell_voltage_max,
            cu / np.maximum(arsenic, 1e-8) - p.cu_as_ratio_max,
        ])
        scales = np.array([p.cu_limit, p.as_min, p.current_density_max,
                           p.cell_voltage_max, p.cu_as_ratio_max])
        return Evaluation(f, g, np.maximum(g, 0) / scales, outputs)


class SyntheticOracle:
    def __init__(self, condition):
        self.condition = condition

    def predict(self, x):
        return synthetic_response(x, self.condition)
