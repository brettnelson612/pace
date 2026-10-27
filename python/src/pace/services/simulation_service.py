"""
pace/services/simulation_service.py
"""

from pace.core.simulation_case import SimulationCase


class SimulationService:
    def __init__(self):
        pass

    def submit_simulation_job(self): ...

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

    def validate_simulation_case(self, simulation_case: SimulationCase):
        """Runs the following validation:

        1. Ensures simulation_case contains all required components
        2. Performs validation of each and every component within the sc
        3. Validates the simulation tree
        4. Checks that this simulation case does not already exist
        5. Checks that we are able to run everything provided.
        """
        pass
