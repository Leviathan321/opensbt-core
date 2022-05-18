from dataclasses import dataclass
from typing import Dict, List
import numpy as np
import os
import sys
import json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

@dataclass
class SimulationOutput(object):
    simTime: float
    times: List
    location: Dict
    velocity: Dict
    distance: List
    collisions: List
    actors: Dict
    others:Dict

    def to_json(self):
        return json.dumps(self.__dict__)

    @classmethod
    def from_json(cls, json_str):
        json_dict = json.loads(json_str)
        return cls(**json_dict)
