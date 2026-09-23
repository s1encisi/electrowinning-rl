# Release benchmark

Executed on 2026-09-24 with the committed `configs/benchmark.json` and seeds 11, 23, 42. All data are synthetic.

## Protocol

- Nine runs: three modes × three seeds. Each seed controls the generated dataset, forest and policy.
- Per run: 4,000 samples; fixed ordered 60/20/20 split; 96 trees; 16,384 PPO training transitions.
- PPO evaluation: 64 independently initialized episodes × 24 steps = 1,536 candidates; seed = training seed + 10,000.
- NSGA-II: population 64, 24 generations. Random search receives its actual number of evaluated candidates.
- All searched/evaluated candidates are audited, including infeasible candidates. Feasibility rate is not the fraction of an already-filtered front.
- Feasible objective vectors are deduplicated and nondominated-filtered. Hypervolume uses fixed scale [10, 10, 10000, 500000] and fixed reference [1.5, 0, 3, 1].
- Candidate counts, training/search budgets and front cardinalities are explicitly unequal. Hypervolume comparisons are descriptive, without significance or equal-budget superiority claims.
- The synthetic oracle re-evaluates candidates against the noiseless generator. It diagnoses surrogate feasibility errors within this artificial example, not on a plant.
- Hyperparameters and seeds were declared before execution. All nine runs are reported, including the weaker serial-mode PPO results.

## Feasibility and hypervolume

Mean ± sample standard deviation over three seeds. Feasibility is the percentage of all candidates satisfying every constraint.

| Mode | Method | Surrogate feasibility (%) | Synthetic oracle feasibility (%) | Hypervolume |
|---|---|---:|---:|---:|
| 1 | nsga2 | 87.391 ± 0.777 | 85.764 ± 2.577 | 6.310 ± 0.012 |
| 1 | ppo | 45.182 ± 6.693 | 46.788 ± 6.200 | 6.081 ± 0.063 |
| 1 | random | 29.970 ± 1.222 | 30.751 ± 0.898 | 5.931 ± 0.018 |
| 2 | nsga2 | 88.086 ± 0.861 | 86.654 ± 2.216 | 6.293 ± 0.045 |
| 2 | ppo | 45.074 ± 6.808 | 47.070 ± 6.364 | 6.030 ± 0.116 |
| 2 | random | 30.056 ± 1.212 | 30.751 ± 0.898 | 5.931 ± 0.012 |
| 3 | nsga2 | 84.983 ± 2.055 | 86.675 ± 1.659 | 5.085 ± 0.096 |
| 3 | ppo | 12.912 ± 1.392 | 16.124 ± 6.010 | 4.161 ± 0.193 |
| 3 | random | 19.922 ± 0.768 | 24.740 ± 1.033 | 4.697 ± 0.090 |

PPO improves on random search for modes 1 and 2 in the recorded hypervolume and feasibility metrics, but does not do so in mode 3. NSGA-II performs better overall under these settings. Extra PPO training evaluations must be included when discussing computational cost.

## Runtime and artifacts

Total measured workflow wall time across the nine runs: 243.62 seconds (Windows, Python 3.11.14, CPU). CPU clock/load and library builds affect timings. No GPU training was performed.

For 64-observation policy batches, per-run p50 forward latency ranged from 0.639 to 0.754 ms; the largest recorded per-run p95 was 1.003 ms. These are 24 warmed-up forward samples per run, including the value heads and excluding surrogate inference, environment steps and I/O. They are neither single-decision latency nor a deployment guarantee.

- [Per-method, per-seed metrics](results/summary.csv)
- [Holdout metrics per target](results/surrogate_summary.csv)
- [Aggregates](results/aggregate.csv)
- [Protocol and installed package versions](results/protocol.json)
- [Complete compact run metadata, hashes and metrics](results/runs.json)

Generated datasets, model binaries, full trajectories and training logs remain in ignored run directories. Rerun the CLI to reconstruct them. SHA-256 hashes bind the recorded metrics to the original generated dataset/model artifacts; binary model hashes can vary with library versions.
