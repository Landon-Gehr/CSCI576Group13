from tests.contracts import (
    CartPoleEnvironment,
    Controller,
    EnvironmentFactory,
    Episode,
    EvaluationAPI,
    RecordingControllerProtocol,
)


def test_random_policy_is_valid_and_reseedable(random_policy_class: type[Controller]) -> None:
    policy: Controller = random_policy_class()
    policy.reset_state(seed=27)
    first: list[int] = [policy.forward((0.0,) * 4) for _ in range(100)]
    assert set(first) == {0, 1}
    policy.reset_state(seed=27)
    assert [policy.forward((0.0,) * 4) for _ in range(100)] == first


def test_random_policy_instances_have_independent_rngs(
    random_policy_class: type[Controller],
) -> None:
    first: Controller = random_policy_class()
    second: Controller = random_policy_class()
    first.reset_state(seed=8)
    second.reset_state(seed=8)
    expected: list[int] = [second.forward((0.0,) * 4) for _ in range(20)]
    second.reset_state(seed=99)
    for _ in range(50):
        second.forward((0.0,) * 4)
    assert [first.forward((0.0,) * 4) for _ in range(20)] == expected


def test_evaluator_resets_state_before_each_episode(
    evaluation: EvaluationAPI,
    env_factory: EnvironmentFactory,
    recording_controller: RecordingControllerProtocol,
) -> None:
    env: CartPoleEnvironment = env_factory(max_steps=3)
    for seed in (11, 12):
        result: Episode = evaluation.run_episode(env, recording_controller, seed=seed, controller_seed=7)
        assert recording_controller.steps_since_reset == result["episode_length"]
    assert recording_controller.reset_seeds == [7, 7]
    assert len(recording_controller.episode_observations) == 2
