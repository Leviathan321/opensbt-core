from dataclasses import dataclass
from typing import Dict, List
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