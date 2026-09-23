import numpy as np
import pytest
import torch
from torch.distributions import Normal, TanhTransform, TransformedDistribution

from electrowinning_rl.config import Config
from electrowinning_rl.ppo import gae, transformed_log_prob
from electrowinning_rl.process import TARGETS, domain
from electrowinning_rl.surrogate import fit_surrogate, generate_data


def test_gae_does_not_cross_terminal():
    rewards = np.array([[1.], [2.]], dtype=np.float32)
    adv, returns = gae(rewards, np.zeros_like(rewards), np.array([[True], [False]]),
                       np.array([10.]), gamma=0.9, lam=1)
    np.testing.assert_allclose(adv[:, 0], [1., 11.])
    np.testing.assert_allclose(returns, adv)


def test_gae_five_cost_heads():
    costs = np.ones((2, 3, 5), dtype=np.float32)
    adv, _ = gae(costs, np.zeros_like(costs), np.array([[False] * 3, [True] * 3]),
                 np.ones((3, 5)) * 100, gamma=0.9, lam=1)
    np.testing.assert_allclose(adv[0], 1.9)
    np.testing.assert_allclose(adv[1], 1)


def test_squashed_probability_matches_torch():
    base = Normal(torch.zeros(4, dtype=torch.float64), torch.ones(4, dtype=torch.float64))
    raw = torch.tensor([-2., -0.5, 0.4, 2.], dtype=torch.float64)
    expected = TransformedDistribution(base, [TanhTransform()]).log_prob(torch.tanh(raw)).sum()
    torch.testing.assert_close(transformed_log_prob(base, raw), expected)
    assert torch.isfinite(transformed_log_prob(base, torch.tensor([100., -100., 0., 0.]))).all()


@pytest.mark.parametrize("condition", [1, 2, 3])
def test_holdout_cannot_change_model_selection(condition):
    config = Config(condition=condition, samples=200, trees=8)
    frame = generate_data(condition, config.samples, config.seed)
    model_a, report_a = fit_surrogate(frame, config)
    # Change ONLY holdout labels: chosen parameters and fitted predictions must stay identical.
    poisoned = frame.copy()
    poisoned.loc[report_a["test_indices"], TARGETS] *= 100
    model_b, report_b = fit_surrogate(poisoned, config)
    assert report_a["selected_max_depth"] == report_b["selected_max_depth"]
    assert report_a["validation_scaled_rmse"] == report_b["validation_scaled_rmse"]
    names = domain(condition)[0]
    np.testing.assert_array_equal(model_a.predict(frame[names].to_numpy()), model_b.predict(frame[names].to_numpy()))
    parts = [set(report_a[k]) for k in ("train_indices", "validation_indices", "test_indices")]
    assert not (parts[0] & parts[1] or parts[0] & parts[2] or parts[1] & parts[2])


def test_duplicates_do_not_cross_split():
    import pandas as pd
    config = Config(samples=200, trees=8)
    frame = generate_data(1, 200)
    duplicated = pd.concat([frame, frame], ignore_index=True)
    duplicated["sample_index"] = np.arange(len(duplicated))
    _, report = fit_surrogate(duplicated, config)
    assert report["unique_rows"] == 200


def test_training_only_imputation():
    config = Config(samples=200, trees=8)
    frame = generate_data(1, 200)
    frame.loc[0, "temperature_a"] = np.nan
    model, _ = fit_surrogate(frame, config)
    # Final refit uses 0:160, never the 40 holdout rows.
    expected = np.nanmedian(frame.loc[:159, "temperature_a"])
    assert model.named_steps["imputer"].statistics_[1] == pytest.approx(expected)
