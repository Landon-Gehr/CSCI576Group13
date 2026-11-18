import csv
from pathlib import Path

from tests.contracts import (
    Episode,
    EvaluationAPI,
)


def test_csv_preserves_episode_values_and_metadata(
    evaluation: EvaluationAPI,
    tmp_path: Path,
) -> None:
    rows: list[Episode] = [
        {"seed": 1, "controller_seed": 8, "return": 4.0, "episode_length": 2, "terminated": True, "truncated": False},
        {"seed": 2, "controller_seed": 8, "return": 6.0, "episode_length": 3, "terminated": False, "truncated": True},
    ]
    path: Path = tmp_path / "episodes.csv"
    evaluation.save_results_csv(rows, path, controller_id="random,baseline", observation_mode="partial")
    with path.open(newline="") as handle:
        saved: list[dict[str, str]] = list(csv.DictReader(handle))
    assert len(saved) == len(rows)
    for actual, expected in zip(saved, rows):
        assert actual["controller_id"] == "random,baseline"
        assert actual["observation_mode"] == "partial"
        for key in ("seed", "controller_seed", "episode_length"):
            assert int(actual[key]) == expected[key]
        assert float(actual["return"]) == expected["return"]
        for key in ("terminated", "truncated"):
            assert actual[key].lower() == str(expected[key]).lower()
