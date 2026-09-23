# Algorithm and engineering contract

## Process domain

Modes 1 and 2 use five decisions: inlet copper concentration (g/L), temperature (°C),
current (A), flow (m³/h), and duration (h). Mode 3 uses eight decisions by providing
separate temperature/current/flow values for units A and B. `process.domain` is the
single source of bounds. RL, random search and NSGA-II all use it.

The surrogate directly predicts outlet copper, outlet arsenic, and voltage. A direct
multi-output model avoids measured-voltage features that would not be available
when optimizing future setpoints. It is a new public demonstration model; it does
not load old model binaries or claim numerical equivalence to them.

For the serial mode, the predicted voltage is used for both units. The common
throughput proxy is `(flow_a + flow_b) / 2`. These are explicit approximations.
The synthetic surface is algebraic and independently generates labels; it is not
fitted to any original measurements. Time ordering of the synthetic rows is merely
generation order and does not establish real temporal generalization.

## Objectives and units

For a single unit:

```text
energy_kwh = voltage * current * duration / 1000
recovered_kg = max(0, cu_in - cu_out) * flow * duration
margin = copper_price - reprocessing_cost - feedstock_cost
treatment = treatment_cost * (treatment_as_reference + cu_out) * flow * duration
profit = margin * recovered_kg - treatment - electricity_price * energy_kwh
         - operating_cost * duration
F = [cu_out, -as_out, energy_kwh, -profit]
```

Concentration in g/L equals kg/m³ numerically. Serial energy sums unit power.
Profit is always in scenario currency; it is never rescaled differently between
algorithms. Treatment uses the configurable arsenic reference 11.64 g/L rather
than predicted arsenic, matching the chosen scenario convention.

Signed inequalities (feasible when every `G_i <= 1e-8`):

```text
G = [cu_out - 8,
     4 - as_out,
     current_density - 340,
     voltage / 16 - 2.5,
     cu_out / as_out - 0.8]
current_density = current / (35 * 2.2638)
```

The serial mode uses the larger current density. Limits are all configurable.
Costs are `max(G, 0) / [8, 4, 340, 2.5, 0.8]`. Invalid surrogate outputs raise an
error rather than silently turning NaN or negative infinity into attractive solutions.

## Learning

Each episode samples initial decisions uniformly and preferences from a four-way
Dirichlet distribution. Observations concatenate normalized decisions, preferences
and the remaining fraction of the finite horizon. Actions are `tanh(u)` with
`u ~ Normal(mu, sigma)` and change each decision by at most 10% of its range.
The next state is clipped to the shared domain.

The surrogate is evaluated once per batch of environments. The four-objective
reward uses fixed scale `[10, 10, 10000, 500000]`. These scales are not fitted on
evaluation data. A reward critic and a five-output cost critic supply GAE estimates:

```text
delta_t = reward_t + gamma * (1 - done_t) * V_(t+1) - V_t
A_t = delta_t + gamma * gae_lambda * (1 - done_t) * A_(t+1)
A_lagrange = A_reward - sum_i(lambda_i * A_cost_i)
lambda_i = clip(lambda_i + lambda_lr * (mean_cost_i - cost_limit), 0, max_lambda)
```

PPO optimizes the clipped importance ratio using the complete transformed
log probability, including the tanh Jacobian. Entropy is a Monte Carlo estimate of
the transformed distribution. Gradients are norm-clipped. Cost limits apply to
mean normalized per-transition violations; they are not a probabilistic guarantee
of safe trajectories. End-of-horizon transitions terminate the finite search task;
rollout-buffer boundaries bootstrap when the episode is still active.

Hyperparameters are fixed before the release benchmark. There is no selection of
the best seed, holdout-driven tuning, expert-policy warm start, multi-agent system,
or claim that the environment models real process dynamics. A stronger policy,
longer training, or expert warm start can be future evaluated changes with explicit
extra compute budgets.

## Validation design

Exact duplicate decision rows are deduplicated before the fixed split. By default,
rows are sorted by unique numeric `sample_index`; 60% train, 20% validation and 20%
test are reserved in order. Optional random splitting uses the declared seed.
The median imputer is fitted inside each candidate pipeline on training rows.
Two ExtraTrees depths (12 and unlimited) are compared using validation RMSE scaled
by training-target standard deviations. The selected pipeline is refitted on
train+validation and tested once on the untouched final holdout.

All test split indices and parameters are saved. Changing test labels is explicitly
tested to leave model selection and fitted predictions unchanged. For longitudinal,
batch-correlated or plant-level data, the split must additionally respect the
deployment grouping; row-level chronological separation alone cannot resolve every
form of dependence.

## Reproducibility and performance

The CLI stores data and model SHA-256 values, package versions, resolved configuration,
losses, violations and dual variables. A policy checkpoint records the SHA-256 of its
surrogate and rejects evaluation with a different artifact. PyTorch policy loading
uses `weights_only=True`; joblib models must still come from trusted sources.

CPU surrogate inference uses one ExtraTrees worker and batches observations rather
than repeatedly loading model copies. PyTorch uses one CPU thread in training. Policy
latency measures warmed-up CPU forward passes, including value heads, for the stated
batch size; it excludes the surrogate, environment and file I/O. It is not an
end-to-end plant-control latency measurement.

Useful primary documentation: [scikit-learn leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html),
[pymoo Problem interface](https://pymoo.org/interface/problem.html),
[Gymnasium environment API](https://gymnasium.farama.org/api/env/).
