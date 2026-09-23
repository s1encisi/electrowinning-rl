"""ExtraTrees training with a fixed holdout and train-only model selection."""

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from .process import TARGETS, domain, synthetic_response


def generate_data(condition, samples=2400, seed=42):
    names, lo, hi = domain(condition)
    rng = np.random.default_rng(seed)
    x = rng.uniform(lo, hi, size=(samples, len(names)))
    y = synthetic_response(x, condition) + rng.normal(0, [0.07, 0.04, 0.08], (samples, 3))
    frame = pd.DataFrame(np.column_stack([x, y]), columns=names + TARGETS)
    frame.insert(0, "sample_index", np.arange(samples))
    return frame


def fit_surrogate(frame, config):
    names, lo, hi = domain(config.condition)
    missing = set(names + TARGETS) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    frame = frame.copy()
    if config.split == "chronological":
        if "sample_index" not in frame:
            raise ValueError("chronological split requires sample_index ordered by observation time")
        index = pd.to_numeric(frame["sample_index"], errors="raise")
        if not np.isfinite(index).all() or index.duplicated().any():
            raise ValueError("sample_index must be finite and unique")
        frame = frame.assign(sample_index=index).sort_values("sample_index", kind="stable")
    # Repeated decisions are one group: retaining one row prevents exact duplicates across splits.
    frame = frame.drop_duplicates(subset=names, keep="first").reset_index(drop=True)
    x, y = frame[names].to_numpy(float), frame[TARGETS].to_numpy(float)
    if len(x) < 100 or not np.isfinite(y).all() or np.any(y < 0):
        raise ValueError("Need >=100 unique decisions with finite nonnegative target values")
    if np.isinf(x).any() or np.any(x < lo) or np.any(x > hi):
        raise ValueError("Training decisions exceed domain bounds or contain Infinity")
    if np.isnan(x).all(axis=0).any():
        raise ValueError("A feature is entirely missing")
    indices = np.arange(len(x))
    if config.split == "random":
        np.random.default_rng(config.seed).shuffle(indices)
    a, b = int(0.6 * len(x)), int(0.8 * len(x))
    train, valid, test = indices[:a], indices[a:b], indices[b:]
    target_scale = np.maximum(np.std(y[train], axis=0), 1e-8)
    candidates = []

    def build(depth):
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("forest", ExtraTreesRegressor(n_estimators=config.trees, max_depth=depth,
                                          random_state=config.seed, n_jobs=1)),
        ])

    for depth in (12, None):
        candidate = build(depth).fit(x[train], y[train])
        pred = candidate.predict(x[valid])
        score = float(np.sqrt(np.mean(((y[valid] - pred) / target_scale) ** 2)))
        candidates.append((score, depth))
    best_score, depth = min(candidates, key=lambda item: item[0])
    # Freeze hyperparameters before this final refit. The holdout never selects a model.
    model = build(depth).fit(x[indices[:b]], y[indices[:b]])
    pred = model.predict(x[test])
    metrics = {name: {"r2": float(r2_score(y[test, i], pred[:, i])),
                      "rmse": float(np.sqrt(mean_squared_error(y[test, i], pred[:, i])))}
               for i, name in enumerate(TARGETS)}
    return model, {
        "split": config.split, "train_rows": len(train), "validation_rows": len(valid),
        "test_rows": len(test), "unique_rows": len(frame), "selected_max_depth": depth,
        "validation_scaled_rmse": best_score, "test": metrics,
        "train_indices": train.tolist(), "validation_indices": valid.tolist(),
        "test_indices": test.tolist(),
    }
