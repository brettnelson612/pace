""" """

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from pace.core.ids import SimulationCaseID, SimulationJobID
from pace.core.pace_object import PaceObject


class RunnerType(StrEnum):
    LOCAL = "local"
    SLURM = "slurm"


@dataclass(frozen=True, kw_only=True)
class RunnerSpec(PaceObject):
    runner_type: RunnerType


class SimulationJobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "canceled"


class FailureReason(StrEnum):
    TIMEOUT = "timeout"
    OUT_OF_MEMORY = "out_of_memory"
    NODE_FAILURE = "node_failure"
    PREEMPTED = "preempted"
    NONZERO_EXIT = "nonzero_exit"
    PREFLIGHT = "preflight"


@dataclass(kw_only=True)
class SimulationJob(PaceObject):
    id: SimulationJobID
    simulation_case: SimulationCaseID
    status: SimulationJobStatus
    runner_spec: RunnerSpec
    failure_reason: FailureReason | None
    backend_state: str | None
    created_at: datetime | None
    started_at: datetime | None
    ended_at: datetime | None
