import numpy as np
import pytest

from electrowinning_rl.config import Config
from electrowinning_rl.env import ElectrowinningEnv
from electrowinning_rl.process import ProcessProblem, SyntheticOracle


class ConstantSurrogate:
    def predict(self, x):
        return np.tile([6., 8., 32.], (len(x), 1))


@pytest.mark.parametrize("condition", [1, 2, 3])
def test_physical_units_and_constraints(condition):
    problem = ProcessProblem(condition, ConstantSurrogate())
    x = (problem.lower + problem.upper)[None, :] / 2
    x[:, 2] = 16000
    if condition == 3:
        x[:, 5] = 20000
    result = problem.evaluate(x)
    expected_energy = 32 * (36000 if condition == 3 else 16000) * x[0, -1] / 1000
    assert result.objectives[0, 2] == pytest.approx(expected_energy)
    assert result.objectives[0, 0] == 6
    assert result.objectives[0, 1] == -8  # Residual arsenic is maximized, not minimized.
    assert result.constraints[0, 0] == -2
    assert result.constraints[0, 1] == -4
    assert result.constraints[0, 4] == pytest.approx(-0.05)
    assert result.feasible[0]
    flow = (x[0, 3] + x[0, 6]) / 2 if condition == 3 else x[0, 3]
    mass = (x[0, 0] - 6) * flow * x[0, -1]
    expected_profit = (98.44 - 0.14 - 82.64) * mass - 2.5 * (11.64 + 6) * flow * x[0, -1]
    expected_profit -= 0.6 * expected_energy + 200 * x[0, -1]
    assert -result.objectives[0, 3] == pytest.approx(expected_profit)


@pytest.mark.parametrize("condition", [1, 2, 3])
def test_gymnasium_contract_and_seed(condition):
    from gymnasium.utils.env_checker import check_env
    config = Config(condition=condition)
    env = ElectrowinningEnv(ProcessProblem(condition, SyntheticOracle(condition)), config)
    check_env(env, skip_render_check=True)
    obs1, _ = env.reset(seed=7)
    obs2, _ = env.reset(seed=7)
    np.testing.assert_array_equal(obs1, obs2)
    for step in range(config.horizon):
        obs, reward, terminated, truncated, info = env.step(np.zeros(env.action_space.shape))
        assert np.isfinite(reward)
        assert env.observation_space.contains(obs)
        assert info["costs"].shape == (5,)
        assert terminated == (step == config.horizon - 1)
        assert not truncated


def test_invalid_predictions_fail_closed():
    class Broken:
        def predict(self, x):
            return np.full((len(x), 3), np.nan)
    problem = ProcessProblem(1, Broken())
    with pytest.raises(ValueError, match="finite"):
        problem.evaluate((problem.lower + problem.upper) / 2)
    with pytest.raises(ValueError, match="domain"):
        problem.evaluate(problem.upper + 1)


def test_negative_predictions_fail_closed():
    class Broken:
        def predict(self, x):
            return np.full((len(x), 3), -1)
    problem = ProcessProblem(1, Broken())
    with pytest.raises(ValueError, match="Negative"):
        problem.evaluate((problem.lower + problem.upper) / 2)


def test_invalid_config_rejected():
    with pytest.raises(ValueError, match="divisible"):
        Config(total_steps=101).validate()
    with pytest.raises(ValueError):
        Config(learning_rate=float("nan")).validate()
