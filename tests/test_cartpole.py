import math

import pytest

from tests.contracts import (
    CartPoleEnvironment,
    EnvironmentFactory,
    State,
)


def test_reset_is_finite_and_repeatable(env_factory: EnvironmentFactory) -> None:
    env: CartPoleEnvironment = env_factory(observation_mode="full")
    first, _ = env.reset(seed=17)
    first = tuple(first)
    env.step(1)
    second, _ = env.reset(seed=17)
    assert len(first) == 4
    assert all(math.isfinite(value) for value in first)
    assert tuple(second) == first


def test_seed_controls_initial_state(env_factory: EnvironmentFactory) -> None:
    env: CartPoleEnvironment = env_factory()
    states: list[tuple[float, ...]] = [tuple(env.reset(seed=seed)[0]) for seed in range(5)]
    assert len(set(states)) > 1


@pytest.mark.parametrize("action,direction", [(0, -1), (1, 1)])
def test_known_transition_from_rest(
    env_factory: EnvironmentFactory,
    action: float,
    direction: int,
) -> None:
    # Reference: Euler integration, dt=.02, force=10, gravity=9.8,
    # cart mass=1, pole mass=.1, pole half-length=.5.
    env: CartPoleEnvironment = env_factory(observation_mode="full")
    env.reset(seed=0, initial_state=(0.0, 0.0, 0.0, 0.0))
    observation, reward, terminated, truncated, _ = env.step(action)
    assert tuple(observation) == pytest.approx((0.0, direction * 0.1951219512195122, 0.0, -direction * 0.2926829268292683))
    assert reward == 1.0
    assert not terminated
    assert not truncated


@pytest.mark.parametrize(
    "state",
    [
        (2.41, 0.0, 0.0, 0.0),
        (-2.41, 0.0, 0.0, 0.0),
        (0.0, 0.0, math.radians(12.1), 0.0),
        (0.0, 0.0, -math.radians(12.1), 0.0),
    ],
)
def test_physical_failure_terminates_episode(env_factory: EnvironmentFactory, state: State) -> None:
    env: CartPoleEnvironment = env_factory(max_steps=100)
    env.reset(seed=0, initial_state=state)
    _, _, terminated, truncated, _ = env.step(0)
    assert terminated
    assert not truncated


def test_time_limit_truncates_without_physical_failure(env_factory: EnvironmentFactory) -> None:
    env: CartPoleEnvironment = env_factory(max_steps=2)
    env.reset(seed=0, initial_state=(0.0, 0.0, 0.0, 0.0))
    _, _, terminated, truncated, _ = env.step(0)
    assert not terminated and not truncated
    _, _, terminated, truncated, _ = env.step(1)
    assert not terminated and truncated


@pytest.mark.parametrize("action", [-1, 2, 0.5])
def test_invalid_action_is_rejected(env_factory: EnvironmentFactory, action: float) -> None:
    env: CartPoleEnvironment = env_factory()
    env.reset(seed=0)
    with pytest.raises(ValueError):
        env.step(action)
