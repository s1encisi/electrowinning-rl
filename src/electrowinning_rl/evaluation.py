"""Feasibility and Pareto metrics, always with a fixed objective normalization."""

import numpy as np
import pandas as pd
from pymoo.indicators.hv import HV
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from .process import CONSTRAINTS, OBJECTIVES, TARGETS

HV_REFERENCE = np.array([1.5, 0., 3., 1.])


def summarize(problem, decisions, output_dir, method, oracle=None):
    result = problem.evaluate(decisions)
    feasible = result.feasible
    unique = np.unique(result.objectives[feasible], axis=0)
    indices = (NonDominatedSorting().do(unique, only_non_dominated_front=True)
               if len(unique) else np.array([], dtype=int))
    front = unique[indices]
    scaled = front / problem.config.objective_scale
    inside = np.all(scaled < HV_REFERENCE, axis=1)
    hv = float(HV(ref_point=HV_REFERENCE)(scaled[inside])) if inside.any() else 0.0
    frame = pd.DataFrame(decisions, columns=problem.names)
    for columns, values in ((TARGETS, result.outputs), (OBJECTIVES[2:], result.objectives[:, 2:]),
                            ([f"g_{c}" for c in CONSTRAINTS], result.constraints)):
        for i, name in enumerate(columns):
            frame[name] = values[:, i]
    frame["feasible"] = feasible
    frame.to_csv(output_dir / f"{method}_candidates.csv", index=False)
    pd.DataFrame(front, columns=OBJECTIVES).to_csv(output_dir / f"{method}_pareto.csv", index=False)
    summary = {"method": method, "candidates": len(decisions), "feasible_rate": float(feasible.mean()),
               "pareto_points": len(front), "hypervolume": hv,
               "hv_points_outside_reference": int((~inside).sum()),
               "mean_normalized_violation": float(result.costs.mean()),
               "best_feasible_profit": float(-result.objectives[feasible, 3].min()) if feasible.any() else None}
    if oracle is not None:
        truth = oracle.evaluate(decisions)
        summary["synthetic_oracle_feasible_rate"] = float(truth.feasible.mean())
        summary["surrogate_feasible_but_oracle_infeasible_rate"] = float((feasible & ~truth.feasible).mean())
    return summary
