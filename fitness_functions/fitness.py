import sys
from simulation.simulator import SimulationOutput 

import numpy as np
import math
from simulation import dummy_simulation as ds
from scipy.spatial.distance import cdist
from utils import geometric
import random
import logging

def fitness_min_distance_two_actors(simout: SimulationOutput):
    dist = simout.otherParams["distanceEgo"]
    v = min(dist)
    return v

def fitness_random(simout):
    return random.random()

def fitness_basic_two_actors(simout: SimulationOutput):
    # consider the distance at last sim step between ego and pedestrian in a non collision situations as 
    # measure for criticality of scenario ( consider that objects proceed to
    # move and will collide in future). If a collision already ocurred, the
    # case is not interesting, since no real simulator is used

    traceEgo = simout.location["ego"]
    tracePed = simout.location["adversary"]

    if simout.otherParams['isCollision']:
        # we dont want scenarios where collisions occure since we have no real driving model
        return sys.float_info.max
    else:
        # calculate the distance between last points if no collision occured; the closer the more critical
        l_e = np.size(traceEgo,1)
        l_p = np.size(tracePed,1)

        # evaluate minimal distance between points of ego and ped
        # value1 = np.min(cdist(list(coordEgo),list(coordPed)))

        # evaluate minimal distance between CORRESPONDING points of ego and ped
        value2 = np.min(geometric.distPair(traceEgo,tracePed))
   
    return value2
# fcn = ds.DummySimulator.simulate

# initEgo = [0,0,90,10]
# initPed = [0,0,90,3]
# init = initEgo + initPed
# print(init)
# f, simout = fitness_basic_two_actors(init, 20, 1, fcn )
# print("fitness is: "+str(f))

# fig = plt.plotSim(simout, map=(0,0,500,500))
