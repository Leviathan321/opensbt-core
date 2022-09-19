from dataclasses import dataclass
from typing import Optional, Dict, List
from unittest import TestCase
import numpy as np
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
    acceleration: Optional[Dict]
    yaw: Optional[Dict]
    collisions: List
    actors: Dict
    otherParams:Dict

    def __init__(self, **kwargs):
        for key,value in kwargs.items():
            setattr(self,key,value)

    def to_json(self):
        return json.dumps(self.__dict__)

    @classmethod
    def from_json(cls, json_str):
        json_dict = json.loads(json_str)
        return cls(**json_dict)
