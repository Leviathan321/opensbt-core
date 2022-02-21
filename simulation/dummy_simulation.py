from models.scenario import Scenario
from simulation.simulator import SimulationOutput, Simulator


class DummySimulator(Simulator):
    def simulate(scenario: Scenario) -> SimulationOutput:
        pass
