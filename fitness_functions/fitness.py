import sys
from simulation.simulator import SimulationOutput 
import numpy as np
from utils import geometric
import random


'''
   For Carla Simulator
'''

def fitness_min_distance_two_actors_carla(simout: SimulationOutput):
    if  "distance" in simout.otherParams:
        dist = simout.otherParams["distance"]
        result = min(dist)
    else:
        traceEgo = simout.location["ego"]
        tracePed = simout.location["adversary"]
        result = np.min(geometric.distPair(traceEgo,tracePed))
    return result


'''
   For Prescan Simulator
'''

def fitness_min_distance_two_actors_prescan(simout: SimulationOutput):
    if  "distance" in simout.otherParams:
        dist = simout.otherParams["distance"]
        result = min(dist)
    else:
        traceEgo = simout.location["ego"]
        tracePed = simout.location["other"]
        result = np.min(geometric.distPair(traceEgo,tracePed))
    return result

def fitness_random(simout):
    return random.random()

'''
   For Dummy Simulator
'''

def fitness_basic_two_actors(simout: SimulationOutput):
    traceEgo = simout.location["ego"]
    tracePed = simout.location["other"]

    if simout.otherParams['isCollision']:
        # we dont want scenarios where collisions occure since we have no real driving model
        return sys.float_info.max
    else:
        value = np.min(geometric.distPair(traceEgo,tracePed))
    return value
