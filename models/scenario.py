from ctypes import Array
from dataclasses import dataclass
from tokenize import String

from models.object import DynamicObject, Ego

@dataclass
class ScenarioEnvironment:
    xodrPath: String
    pass

@dataclass
class Scenario(object):
    environment: ScenarioEnvironment
    ego: Ego
    dynObject: DynamicObject
    pass

@dataclass
class ScenarioInstance(Scenario):
    pass

