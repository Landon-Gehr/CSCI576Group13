import csv
import statistics
from pathlib import Path

import pytest

from tests.contracts import (
    Controller,
    EnvironmentFactory,
    Episode,
    EvaluationAPI,
    EvaluationResult,
    ObservationMode,
)


@pytest.mark.parametrize("mode", ["full", "partial"])
def test_random_baseline_pipeline(
    mode: ObservationMode,
    env_factory: EnvironmentFactory,
    random_policy_class: type[Controller],
    evaluation: EvaluationAPI,
    tmp_path: Path,
) -> None:
    def run() -> EvaluationResult:
        return evaluation.evaluate_controller(
            env_factory=lambda: env_factory(observation_mode=mode, max_steps=20),
            controller_factory=random_policy_class,
            seeds=[10, 20, 30],
            controller_seed=7,
        )

    result: EvaluationResult = run()
    assert result == run()
    episodes: list[Episode] = result["episodes"]
    assert [row["seed"] for row in episodes] == [10, 20, 30]
    for row in episodes:
        assert 1 <= row["episode_length"] <= 20
        assert row["return"] == row["episode_length"]
        assert row["terminated"] or row["truncated"]
    returns: list[float] = [row["return"] for row in episodes]
    assert result["mean_return"] == pytest.approx(statistics.mean(returns))
    assert result["std_return"] == pytest.approx(statistics.pstdev(returns))
    path: Path = tmp_path / f"random_{mode}.csv"
    evaluation.save_results_csv(episodes, path, controller_id="random", observation_mode=mode)
    with path.open(newline="") as handle:
        saved: list[dict[str, str]] = list(csv.DictReader(handle))
    assert len(saved) == 3
    assert [int(row["seed"]) for row in saved] == [10, 20, 30]
    assert all(row["observation_mode"] == mode for row in saved)
    assert [float(row["return"]) for row in saved] == returns
