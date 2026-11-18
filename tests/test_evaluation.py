import pytest

from tests.contracts import (
    CartPoleEnvironment,
    EnvironmentFactory,
    Episode,
    EvaluationAPI,
    EvaluationResult,
    RecordingControllerProtocol,
    ResetResult,
    StepResult,
)


class FixedLengthEnvironment:
    """Independent oracle: seed determines episode length, each reward is 2."""

    def __init__(self) -> None:
        self.reset_seeds: list[int] = []
        self.remaining: int = 0

    def reset(self, seed: int | None = None) -> ResetResult:
        assert seed is not None
        self.reset_seeds.append(seed)
        self.remaining = seed
        return (0.0, 0.0), {}

    def step(self, action: float) -> StepResult:
        assert self.remaining > 0, "Evaluator stepped after termination"
        assert action in (0, 1)
        self.remaining -= 1
        return (0.0, 0.0), 2.0, self.remaining == 0, False, {}


def test_multiseed_evaluation_uses_every_seed_and_correct_statistics(
    evaluation: EvaluationAPI,
    recording_controller: RecordingControllerProtocol,
) -> None:
    environments: list[FixedLengthEnvironment] = []

    def make_environment() -> FixedLengthEnvironment:
        env: FixedLengthEnvironment = FixedLengthEnvironment()
        environments.append(env)
        return env

    result: EvaluationResult = evaluation.evaluate_controller(
        env_factory=make_environment,
        controller_factory=lambda: recording_controller,
        seeds=[1, 2, 3],
        controller_seed=19,
    )
    episodes: list[Episode] = result["episodes"]
    assert [row["seed"] for row in episodes] == [1, 2, 3]
    assert [row["controller_seed"] for row in episodes] == [19, 19, 19]
    assert [row["episode_length"] for row in episodes] == [1, 2, 3]
    assert [row["return"] for row in episodes] == [2.0, 4.0, 6.0]
    assert all(row["terminated"] and not row["truncated"] for row in episodes)
    actual_seeds: list[int] = [seed for env in environments for seed in env.reset_seeds]
    assert actual_seeds == [1, 2, 3]
    assert result["mean_return"] == pytest.approx(4.0)
    assert result["std_return"] == pytest.approx((8 / 3) ** 0.5)
    assert recording_controller.reset_seeds == [19, 19, 19]


def test_single_seed_has_zero_population_standard_deviation(
    evaluation: EvaluationAPI,
    recording_controller: RecordingControllerProtocol,
) -> None:
    result: EvaluationResult = evaluation.evaluate_controller(
        env_factory=FixedLengthEnvironment,
        controller_factory=lambda: recording_controller,
        seeds=[2],
        controller_seed=0,
    )
    assert result["mean_return"] == 4.0
    assert result["std_return"] == 0.0


def test_empty_seed_list_is_rejected(
    evaluation: EvaluationAPI,
    recording_controller: RecordingControllerProtocol,
) -> None:
    with pytest.raises(ValueError):
        evaluation.evaluate_controller(
            env_factory=FixedLengthEnvironment,
            controller_factory=lambda: recording_controller,
            seeds=[],
            controller_seed=0,
        )


def test_same_seeds_reproduce_entire_observation_trajectory(
    evaluation: EvaluationAPI,
    env_factory: EnvironmentFactory,
    recording_controller: RecordingControllerProtocol,
) -> None:
    env: CartPoleEnvironment = env_factory(max_steps=20)
    first: Episode = evaluation.run_episode(env, recording_controller, seed=23, controller_seed=5)
    second: Episode = evaluation.run_episode(env, recording_controller, seed=23, controller_seed=5)
    assert first == second
    assert recording_controller.episode_observations[0] == (recording_controller.episode_observations[1])
