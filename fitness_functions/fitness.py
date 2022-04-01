import sys
import os

from simulation.simulator import SimulationOutput 

# sys.path.insert(0, '../simulation')
# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.dirname(SCRIPT_DIR))

import numpy as np
from simulation import dummy_simulation as ds
import visualization.plotter as plt
from scipy.spatial.distance import cdist
from utils import geometric


def fitness_basic_two_actors(vars, simTime, samplingTime,simulateFcn):
    # consider the distance at last sim step between ego and pedestrian in a non collision situations as 
    # measure for criticality of scenario ( consider that objects proceed to
    # move and will collide in future). If a collision already ocurred, the
    # case is not interesting, since no real simulator is used

    simout = simulateFcn(vars,simTime=simTime,samplingTime=samplingTime)

    traceEgo = simout.egoTrajectory
    tracePed = simout.objectTrajectory

    if simout.otherParams['isCollision']:
        # we dont want scenarios where collisions occure since we have no real driving model
        return sys.float_info.max, simout
    else:
        # calculate the distance between last points if no collision occured; the closer the more critical
        l_e = np.size(traceEgo,1)
        l_p = np.size(tracePed,1)

        # create coordinate tuples
        coordEgo = zip(traceEgo[1,:],traceEgo[2,:])
        coordPed = zip(tracePed[1,:],tracePed[2,:])     
  
        # evaluate minimal distance between points of ego and ped
        # value1 = np.min(cdist(list(coordEgo),list(coordPed)))

        # evaluate minimal distance between CORRESPONDING points of ego and ped
        value2 = np.min(geometric.distPair(list(coordEgo),list(coordPed)))
   
    return value2, simout
# fcn = ds.DummySimulator.simulate

# initEgo = [0,0,90,10]
# initPed = [0,0,90,3]
# init = initEgo + initPed
# print(init)
# f, simout = fitness_basic_two_actors(init, 20, 1, fcn )
# print("fitness is: "+str(f))

# fig = plt.plotSim(simout, map=(0,0,500,500))
