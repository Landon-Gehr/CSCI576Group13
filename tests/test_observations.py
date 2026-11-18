import pytest

from tests.contracts import (
    CartPoleEnvironment,
    EnvironmentFactory,
    State,
)


def test_full_observation_exposes_ordered_state(env_factory: EnvironmentFactory) -> None:
    env: CartPoleEnvironment = env_factory(observation_mode="full")
    state: State = (0.1, 0.2, 0.03, -0.4)
    observation, _ = env.reset(seed=0, initial_state=state)
    assert tuple(observation) == pytest.approx(state)


def test_partial_observation_contains_only_position_and_angle(
    env_factory: EnvironmentFactory,
) -> None:
    env: CartPoleEnvironment = env_factory(observation_mode="partial")
    observation, _ = env.reset(seed=0, initial_state=(0.1, 0.2, 0.03, -0.4))
    assert tuple(observation) == pytest.approx((0.1, 0.03))


def test_modes_share_dynamics_and_episode_endings(env_factory: EnvironmentFactory) -> None:
    full: CartPoleEnvironment = env_factory(observation_mode="full", max_steps=10)
    partial: CartPoleEnvironment = env_factory(observation_mode="partial", max_steps=10)
    full.reset(seed=42)
    partial.reset(seed=42)
    for action in [0, 1] * 5:
        full_obs, reward, terminated, truncated, _ = full.step(action)
        partial_obs, p_reward, p_terminated, p_truncated, _ = partial.step(action)
        assert tuple(partial_obs) == pytest.approx((full_obs[0], full_obs[2]))
        assert (p_reward, p_terminated, p_truncated) == (reward, terminated, truncated)
        if terminated or truncated:
            break
