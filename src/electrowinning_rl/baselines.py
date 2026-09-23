"""Vectorized NSGA-II and random-search baselines using the shared evaluator."""

import time

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize


def run_nsga2(problem, config):
    candidates = []

    class Adapter(Problem):
        def __init__(self):
            super().__init__(n_var=problem.n_var, n_obj=4, n_ieq_constr=5,
                             xl=problem.lower, xu=problem.upper)

        def _evaluate(self, x, out, *args, **kwargs):
            result = problem.evaluate(x)
            out["F"], out["G"] = result.objectives, result.constraints
            candidates.append(x.copy())

    start = time.perf_counter()
    result = minimize(Adapter(), NSGA2(pop_size=config.population), ("n_gen", config.generations),
                      seed=config.seed, verbose=False)
    return np.concatenate(candidates), {
        "search_seconds": time.perf_counter() - start,
        "search_evaluations": int(result.algorithm.evaluator.n_eval),
    }


def run_random(problem, count, seed):
    start = time.perf_counter()
    x = problem.sample(np.random.default_rng(seed), count)
    problem.evaluate(x)
    return x, {"search_seconds": time.perf_counter() - start, "search_evaluations": count}
