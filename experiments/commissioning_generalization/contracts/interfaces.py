"""Implementation contracts for CGX. No MuJoCo/model implementation is claimed here.

The budget utility below is executable; Protocol classes specify required interfaces.
Never place evaluator truth in PublicContext/AcquiredHistory.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

@dataclass(frozen=True)
class ResourceVector:
    executions: int = 0
    bytes: int = 0
    calibrations: int = 0
    decisions: int = 0

    def __post_init__(self) -> None:
        for name in ("executions", "bytes", "calibrations", "decisions"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")

    def can_afford(self, cost: ResourceVector) -> bool:
        return all(getattr(self, f) >= getattr(cost, f)
                   for f in ("executions", "bytes", "calibrations", "decisions"))

    def subtract(self, cost: ResourceVector) -> ResourceVector:
        if not self.can_afford(cost):
            raise ValueError("Resource cap would be exceeded")
        return ResourceVector(**{f: getattr(self, f)-getattr(cost, f)
                                 for f in ("executions", "bytes", "calibrations", "decisions")})

@dataclass(frozen=True)
class ObservationAction:
    key: str
    kind: str
    ports: tuple[str, ...]
    sample_hz: int
    calibration_pair: tuple[str, str] | None
    upper_cost: ResourceVector

@dataclass(frozen=True)
class PublicContext:
    graph: Mapping[str, Any]
    requirements: Mapping[str, Any]
    nominal_task: Mapping[str, Any]
    initial_record_ref: str
    legal_target_keys: tuple[str, ...]
    declared_prior: tuple[float, ...]

@dataclass(frozen=True)
class AcquiredHistory:
    context: PublicContext
    action_keys: tuple[str, ...]
    acquired_record_refs: tuple[str, ...]
    remaining: ResourceVector
    visible_status: str

@dataclass(frozen=True)
class Prediction:
    target_keys: tuple[str, ...]
    log_probabilities_base2: tuple[float, ...]

class PublicEnvironment(Protocol):
    def legal_actions(self, history: AcquiredHistory) -> Sequence[ObservationAction]: ...
    def acquire(self, action: ObservationAction) -> Mapping[str, Any]: ...

class DiagnosticReader(Protocol):
    def predict(self, history: AcquiredHistory) -> Prediction: ...

class AcquisitionPolicy(Protocol):
    def choose(self, history: AcquiredHistory,
               actions: Sequence[ObservationAction], prediction: Prediction
               ) -> ObservationAction: ...

class FixedPlanner(Protocol):
    def plan(self, initial: AcquiredHistory,
             actions: Sequence[ObservationAction]) -> Sequence[ObservationAction]: ...

class PrivateEvaluator(Protocol):
    # An evaluator is never reachable from a policy object's environment handle.
    def score_prediction(self, case_id: str, prediction: Prediction) -> Mapping[str, float]: ...
    def hidden_audit(self, case_id: str, deployed_state_ref: str) -> Mapping[str, Any]: ...
