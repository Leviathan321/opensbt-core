
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List
from pymoo.core.individual import Individual
from abc import ABC, abstractmethod, abstractstaticmethod

import os
import sys
import json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

'''
    Represent data output by a simulator

    Example json representation of a SimulationOutput instance:

    {
        "simTime" : 3,
        "times": [1.0,2.0,3.0],
        "location": { "ego" : [(1,1),(2,2),(3,3)],
                    "adversary" : [(4,1),(4,2),(4,3)},

        "velocity": { "ego" : [0.5,0.5,0.5],
                        "adversary" : [0.9,0.9,0.9],
                        },
        "collisions": [],
        "actors" : {1: "ego",
                        2: "adversary"
                    },
        "otherParams" : {
            "isCollision": False
        }
    }
        
'''
@dataclass
class SimulationOutput(object):
    simTime: float
    times: List
    location: Dict
    velocity: Dict
    speed: Dict
    acceleration: Dict
    yaw: Dict
    collisions: List
    actors: Dict
    otherParams: Dict

    def __init__(self, **kwargs):
        for key,value in kwargs.items():
            setattr(self, key, value)

    def to_json(self):
        return json.dumps(self.__dict__)

    @classmethod
    def from_json(cls, json_str):
        json_dict = json.loads(json_str)
        return cls(**json_dict)


class Simulator(ABC):
        
    @abstractstaticmethod
    def simulate(list_individuals: List[Individual], 
                variable_names: List[str], 
                scenario_path: str, 
                sim_time: float = 10, 
                time_step: float = 0.01, 
                do_visualize: bool = True) -> List[SimulationOutput]:
        pass
        #raise NotImplementedError("Implement the simulation of a batch of scenario instances.")
      
class SimulationType(Enum):
    DUMMY = 0
    CARLA = 1
    PRESCAN = 2
    NONE = 3
