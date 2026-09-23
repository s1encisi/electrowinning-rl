import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from electrowinning_rl.cli import main

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("condition", [1, 2, 3])
def test_end_to_end_and_checkpoint_reload(tmp_path, condition):
    output = tmp_path / "demo"
    main(["demo", "--config", str(ROOT / "configs/smoke.json"), "--condition", str(condition),
          "--output", str(output)])
    report = json.loads((output / "metrics.json").read_text())
    assert report["source"] == "synthetic-demo"
    assert {m["method"] for m in report["methods"]} == {"random", "nsga2", "ppo"}
    assert all(np.isfinite(m["hypervolume"]) for m in report["methods"])
    reevaluation = tmp_path / "evaluation"
    main(["evaluate", "--model", str(output / "surrogate.joblib"),
          "--policy", str(output / "policy.pt"), "--output", str(reevaluation)])
    before = pd.read_csv(output / "ppo_candidates.csv")
    after = pd.read_csv(reevaluation / "ppo_candidates.csv")
    pd.testing.assert_frame_equal(before, after)


def test_output_not_overwritten(tmp_path):
    original = tmp_path / "keep.txt"
    original.write_text("original")
    with pytest.raises(SystemExit) as error:
        main(["demo", "--output", str(tmp_path)])
    assert error.value.code == 2
    assert original.read_text() == "original"


def test_help_from_unrelated_directory(tmp_path):
    result = subprocess.run([sys.executable, "-m", "electrowinning_rl", "--help"],
                            cwd=tmp_path, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    assert "benchmark" in result.stdout
