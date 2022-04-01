from cmath import pi, sin
from math import cos
from simulation.simulator import SimulationOutput
from dynamics import basic_dynamics
from utils import geometric
import numpy.linalg as linalg

class DummySimulator(object):
    samplingTime = 1
    
    @staticmethod
    def simulate(vars, simTime: float, samplingTime=samplingTime) -> SimulationOutput:
        # print("*** INPUT SIMULATE ***")
        # print("vars: " + str(vars))
        # print("simTime: " + str(simTime))
        # print("samplingTime: " +  str(samplingTime))

        egoInitialVelocity = vars[3]
        pedInitialVelocity = vars[7]

        egoTrajectory = basic_dynamics.planMotion(vars[0:2], vars[2], vars[3],simTime,samplingTime)
        objectTrajectory = basic_dynamics.planMotion(vars[4:6], vars[6], vars[7],simTime,samplingTime)

        lineEgoPoints = (egoTrajectory[1:3,0],egoTrajectory[1:3,1])
        linePedPoints = (objectTrajectory[1:3,0],objectTrajectory[1:3,1])

        colpoint = geometric.intersection(list(lineEgoPoints),list(linePedPoints))

        otherparams = {}

        if colpoint!=[]:          
            dist_ego_colpoint = geometric.dist(colpoint, list(egoTrajectory[1:3,0]))
            dist_ped_colpoint = geometric.dist(colpoint, list(objectTrajectory[1:3,0]))
            
            t_col_ego = dist_ego_colpoint/egoInitialVelocity
            t_col_ped = dist_ped_colpoint/pedInitialVelocity
            
            # collision occurs when both objects reach line crossing at
            # the same time (with some tolerance)
            
            t_tolerance = 1; #time tolerance for missed collision  

            otherparams = {}
            if  (t_col_ego - t_col_ped) < t_tolerance:
                otherparams['isCollision'] = True
            else:
                otherparams['isCollision'] = False

            # print(" *** GENERATED OUTPUT ***")
            # # print("egoTrajectory: "+ str(egoTrajectory))
            # # print("objectTrajectory: "+ str(objectTrajectory))
            # print("otherparams: "+ str(otherparams))

        else:
            otherparams['isCollision'] = False

        return SimulationOutput(simTime,egoTrajectory=egoTrajectory,objectTrajectory=objectTrajectory, otherParams=otherparams)
