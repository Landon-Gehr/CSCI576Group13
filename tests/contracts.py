from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal, Protocol, TypedDict

Observation = Sequence[float]
ObservationMode = Literal["full", "partial"]
State = tuple[float, float, float, float]
ResetResult = tuple[Observation, dict[str, object]]
StepResult = tuple[Observation, float, bool, bool, dict[str, object]]


Episode = TypedDict(
    "Episode",
    {
        "seed": int,
        "controller_seed": int,
        "return": float,
        "episode_length": int,
        "terminated": bool,
        "truncated": bool,
    },
)


class EvaluationResult(TypedDict):
    episodes: list[Episode]
    mean_return: float
    std_return: float


class Controller(Protocol):
    def reset_state(self, seed: int | None = None) -> None: ...

    def forward(self, observation: Observation) -> int: ...


class RecordingControllerProtocol(Controller, Protocol):
    reset_seeds: list[int | None]
    episode_observations: list[list[tuple[float, ...]]]
    steps_since_reset: int


class Environment(Protocol):
    def reset(self, seed: int | None = None) -> ResetResult: ...

    def step(self, action: float) -> StepResult: ...


class CartPoleEnvironment(Environment, Protocol):
    def reset(self, seed: int | None = None, initial_state: State | None = None) -> ResetResult: ...


class EnvironmentFactory(Protocol):
    def __call__(self, observation_mode: ObservationMode = "full", max_steps: int = 500) -> CartPoleEnvironment: ...


class EvaluationAPI(Protocol):
    def run_episode(
        self,
        env: Environment,
        controller: Controller,
        *,
        seed: int,
        controller_seed: int,
    ) -> Episode: ...

    def evaluate_controller(
        self,
        *,
        env_factory: Callable[[], Environment],
        controller_factory: Callable[[], Controller],
        seeds: Sequence[int],
        controller_seed: int,
    ) -> EvaluationResult: ...

    def save_results_csv(
        self,
        episodes: Sequence[Episode],
        path: Path,
        *,
        controller_id: str,
        observation_mode: ObservationMode,
    ) -> None: ...
