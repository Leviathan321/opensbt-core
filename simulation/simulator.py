from ctypes import Array
from dataclasses import dataclass
from typing import Dict, List
from models.scenario import ScenarioInstance
import numpy as np
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

@dataclass
class SimulationOutput(object):
    simTime: float
    egoTrajectory: np.ndarray
    objectTrajectory: np.ndarray
    otherParams: Dict
    pass

@dataclass
class Simulator:
    simTime: int = 30

    @staticmethod
    def simulateOSC(self, oscFilePath: str, simTime: float) -> SimulationOutput:
        pass

    