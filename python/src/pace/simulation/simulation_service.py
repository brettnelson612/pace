"""
db
    |___relational
            |___reactors
            |___simulation_jobs
            |___simulation_cases
            |___solver_outputs
            |___sample_specs
            |___sample_datasets
simulation
    |___utilities
            |___bundle_writer
            |___file_io
    |___solvers
            |___openmc
                    |___openmc_adapter.py
                    |___<supporting modules>
            |___moose
                    |___moose_adapter.py
                    |___<supporting modules>
            |___cardinal
                    |___cardinal_adapter.py
                    |___<supporting modules>
    |___simulation_job.py
    |___simulation_case.py
    |___simulation_service.py
    |___runners
            |___local_runner.py
            |___slurm_runner.py

Workflow
1.  simulation service create API called
2.  initial input validation
3.  simulation outline created (which solvers, etc)
4.  for each solver, call its adapter's build class, which includes validation
5.  adapter takes the input model + operating conditions and writes setup files
6.  once openmc outputs are created, cardinal adapter can go to work
7.

Features needed
1.  A tree that defines the steps to follow, given the solver set.
        -   Example: If the set is (openmc, MOOSE Heat Transfer) then I believe those
            two solvers first need to be set up, then cardinal set up using their outputs.
        -   We can consider this to be





2.

"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from pace.core.operating_conditions import OperatingConditions
from pace.core.pace_object import PaceObject
from pace.core.reactor_blueprint import ReactorBlueprint


class SolverName(StrEnum):
    """Labels for the different physics solver tools."""

    OPENMC = "openmc"
    CARDINAL = "cardinal"
    MOOSE_HEAT_TRANSFER = "moose_heat_transfer"
    MOOSE_THM = "moose_thm"


@dataclass(frozen=True, kw_only=True)
class SolverSpec(PaceObject):
    pass


@dataclass(frozen=True, kw_only=True)
class OpenmcCaseSpec(SolverSpec):
    pass


@dataclass(frozen=True, kw_only=True)
class MHeatTransferCaseSpec(SolverSpec):
    pass


@dataclass(frozen=True, kw_only=True)
class CardinalCaseSpec(SolverSpec):
    pass


@dataclass(frozen=True, kw_only=True)
class SimulationCase(PaceObject):
    reactor_blueprint: ReactorBlueprint
    operating_conditions: OperatingConditions
    solver_specs: dict[SolverName, SolverSpec]

    def to_dict(self) -> dict: ...

    @classmethod
    def from_dict(cls, data: dict) -> SimulationCase: ...


@dataclass(frozen=True, kw_only=True)
class RunnerSpec(PaceObject):
    runner_type: RunnerType


class RunnerType(StrEnum):
    LOCAL = "local"
    SLURM = "slurm"


class SimulationJobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELED = "canceled"


class FailureReason(StrEnum):
    TIMEOUT = "timeout"
    OUT_OF_MEMORY = "out_of_memory"
    NODE_FAILURE = "node_failure"
    PREEMPTED = "preempted"
    NONZERO_EXIT = "nonzero_exit"
    PREFLIGHT = "preflight"


@dataclass(kw_only=True)
class SimulationJob(PaceObject):
    simulation_case: SimulationCase
    status: SimulationJobStatus
    runner_spec: RunnerSpec
    failure_reason: FailureReason | None
    backend_state: str | None
    created_at: datetime | None
    started_at: datetime | None
    ended_at: datetime | None


class SimulationService:
    def __init__(self):
        pass

    def submit_simulation_job(): ...

    def build_simulation_case(self):
        """Is this needed? Build methods like this are often used when a caller has only
        some of the ingredients required to build the end product. Here however, they need
        to specify pretty much everything. We cannot go and build any of the solver specs
        for them, for instance — they need to do that themselves as we have no idea what
        they want set up. Things that we likely want to do for them:
            - validation of what they've provided (though we can also expose that API publicly)
            - registering it for them
            - running it (obviously)
            - checking that it doesn't already exist (included in validation)

        The only thing that comes to mind with regards to having a "build" method like this
        is if there is information we want to store in the SimulationCase object that we
        don't feel they need to specify. This could be default values, cardinal stuff, or
        extra things that the solver ecosystem cares but which we don't think PACE users need
        to necessarily care about. So perhaps there is a justification for it... or perhaps
        that extra build logic gets placed into the adapter build methods.

        """
        pass

    def register_simulation_case(self, simulation_case: SimulationCase):
        """Register a simulation case.

        We always perform simulation case validation before registering.
        """

        self.validate_simulation_case(simulation_case)
        self._register_simulation_case(simulation_case)

    def validate_simulation_case(self, simulation_case: SimulationCase):
        """Runs the following validation:

        1. Ensures simulation_case contains all required components
        2. Performs validation of each and every component within the sc
        3. Validates the simulation tree
        4. Checks that this simulation case does not already exist
        5. Checks that we are able to run everything provided.
        """
        pass

    def _register_simulation_case(
        self, simulation_case: SimulationCase
    ) -> SimulationCase: ...


reactor_blueprint = ...
operating_conditions = ...
solver_specs = ...


# let's define the solver spec generation logic
class SolverAdapter:
    """Holds stuff that every adapter cares about/must implement."""

    pass


class OpenmcAdapter(SolverAdapter):
    """Holds everything specific to openmc

    Things that likely need to be on here:
    1.  build_solver_spec() -> OpenmcCaseSpec
        takes the minimal user input, adds anything else needed
    2.  logic to build the openmc code
    3.  logic to read openmc stuff
    4.  logic to

    We need to make sure this class isn't taking on too much work.
    The following should probably exist outside of this class:
    1.  a file/python reader/writer helper
    2.  a
    """

    pass


# simulation_case = SimulationCase(
#     reactor_blueprint=rb,
#     operating_conditions=oc,
#     solver_specs=solver_specs,
# )

# # in pace service or pace front-end
# simulation_service = SimulationService()
# simulation_service.register_simulation_case()
