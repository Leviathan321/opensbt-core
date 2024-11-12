
from dataclasses import dataclass
from typing import Dict, List
from pymoo.core.individual import Individual
from abc import ABC, abstractstaticmethod
from opensbt.utils.encoder_utils import NumpyEncoder

import os
import sys
import json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

@dataclass
class SimulationOutput(object):
    
    """
        Class represents data output after execution of a simulation. An example JSON representation of a SimulationOutput instance is:

        {
            "simTime" : 3,
            "times": [1.0,2.0,3.0],
            "location": { 
                        "ego" : [(1,1),(2,2),(3,3)],
                        "adversary" : [(4,1),(4,2),(4,3)
                        },
            "velocity": { 
                        "ego" : [0.5,0.5,0.5],
                        "adversary" : [0.9,0.9,0.9],
                        },
            "collisions": [],
            "actors" : {
                        1: "ego",
                        2: "adversary"
                        },
            "otherParams" : {
                "isCollision": False
            },
            ...
        }
            
    """
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
        return json.dumps(self.__dict__,
                        allow_nan=True, 
                        indent=4,
                        cls=NumpyEncoder)

    @classmethod
    def from_json(cls, json_str):
        json_dict = json.loads(json_str)
        return cls(**json_dict)

class Simulator(ABC):
    """ Base class to be inherited and implemented by a concrete simulator in OpenSBT """

    @abstractstaticmethod
    def simulate(list_individuals: List[Individual], 
                variable_names: List[str], 
                scenario_path: str, 
                sim_time: float = 10, 
                time_step: float = 0.01, 
                do_visualize: bool = True) -> List[SimulationOutput]:
        """
         Instantiates a list of scenarios using the scenario_path, the variable_names and variable values in list_individuals, and
         simulates scenarios in defined simulator.
         
        :param list_individuals: List of individuals. On individual corresponds to one scenario.
        :type list_individuals: List[Individual]
        :param variable_names: The scenario variables.
        :type variable_names: List[str]
        :param scenario_path: The path to the abstract/logical scenario.
        :type scenario_path: str
        :param sim_time: The simulation time, defaults to 10
        :type sim_time: float, optional
        :param time_step: The time step, defaults to 0.01
        :type time_step: float, optional
        :param do_visualize: Visualize or not simulation, defaults to True
        :type do_visualize: bool, optional
        :return: Returns a list of Simulation output instances.
        :rtype: List[SimulationOutput]
        """
        pass