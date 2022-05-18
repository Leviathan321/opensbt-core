from cmath import pi, sin
from math import cos
from simulation.simulator import SimulationOutput
from dummySimulation.dynamics import basic_dynamics
from utils import geometric
import numpy as np
import json

class DummySimulator(object):
    samplingTime = 1

    ## Simulates a set of scenarios and returns the output
    @staticmethod
    def simulateBatch(listIndividuals, featureNames, xosc: str, simTime: float,samplingTime = samplingTime):
        results = []
        for ind in listIndividuals:
            simout =  DummySimulator.simulate(ind, featureNames, filepath=xosc, simTime=simTime, samplingTime=samplingTime) 
            results.append(simout)
        return results

    @staticmethod
    def simulate(vars, featureNames, filepath, simTime: float, samplingTime=samplingTime) -> SimulationOutput:
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

        ego_location = [pos for pos in zip(list(egoTrajectory[1,:]),list(egoTrajectory[2,:])) ]
        
        obj_location = [pos for pos in zip(list(objectTrajectory[1,:]),list(objectTrajectory[2,:])) ]

        result = {
                "simTime" : 0,
                "times": list(egoTrajectory[0,:]),
                "location": { "ego" : ego_location,
                            "adversary" : obj_location},

                "velocity": { "ego" : list(egoTrajectory[3,:]),
                                "adversary" : list(objectTrajectory[3,:]),
                                },
                "distance" : [],
                "collisions": [],
                "actors" : {1: "ego",
                                2: "adversary"
                            },
                "otherParams" : {}
        }
        
        otherParams = {}

        if colpoint!=[]:          
            dist_ego_colpoint = geometric.dist(colpoint, list(egoTrajectory[1:3,0]))
            dist_ped_colpoint = geometric.dist(colpoint, list(objectTrajectory[1:3,0]))
            
            t_col_ego = dist_ego_colpoint/egoInitialVelocity
            t_col_ped = dist_ped_colpoint/pedInitialVelocity
            
            # collision occurs when both objects reach line crossing at
            # the same time (with some tolerance)
            
            t_tolerance = 1; #time tolerance for missed collision  
            if  (t_col_ego - t_col_ped) < t_tolerance:
                otherParams['isCollision'] = True
            else:
                otherParams['isCollision'] = False

            result["collisions"] = [[2]]
            result["otherParams"] = otherParams

        else:
            otherParams['isCollision'] = False
        
        result["otherParams"] = otherParams


        return SimulationOutput.from_json(json.dumps(result))
