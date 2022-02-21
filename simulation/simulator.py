from ctypes import Array
from dataclasses import dataclass
from typing import Dict, List
from models.scenario import ScenarioInstance
from legacy.models import Scenario

class SimulationOutput(object):
    egoTrajectory: Array
    objectTrajectory: Array
    simTime: float
    otherParams: Dict
    pass

@dataclass
class Simulator:
    simTime: int = 30

    @staticmethod
    def simulateOSC(self, oscFilePath: str) -> SimulationOutput:
        pass

    @staticmethod
    def simulate(self, scenario: ScenarioInstance) -> SimulationOutput:
        pass