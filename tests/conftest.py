import importlib
from typing import cast

import pytest

from tests.contracts import Controller, EnvironmentFactory, EvaluationAPI, Observation


@pytest.fixture
def env_factory() -> EnvironmentFactory:
    return cast(
        EnvironmentFactory,
        importlib.import_module("simulations.pole_balancing").PoleBalancingSimulation,
    )


@pytest.fixture
def evaluation() -> EvaluationAPI:
    return cast(EvaluationAPI, importlib.import_module("experiments.evaluation"))


@pytest.fixture
def random_policy_class() -> type[Controller]:
    return cast(type[Controller], importlib.import_module("models.random_policy").RandomPolicy)


class RecordingController:
    def __init__(self) -> None:
        self.reset_seeds: list[int | None] = []
        self.episode_observations: list[list[tuple[float, ...]]] = []
        self.steps_since_reset: int = 0

    def reset_state(self, seed: int | None = None) -> None:
        self.reset_seeds.append(seed)
        self.episode_observations.append([])
        self.steps_since_reset = 0

    def forward(self, observation: Observation) -> int:
        assert self.reset_seeds, "Controller must reset before its first action"
        self.episode_observations[-1].append(tuple(observation))
        self.steps_since_reset += 1
        return 0


@pytest.fixture
def recording_controller() -> RecordingController:
    return RecordingController()
