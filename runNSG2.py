from algorithm.nsga2_TC import *
from fitness_functions import fitness
from simulation.carla_simulation import CarlaSimulator
from simulation.dummy_simulation import DummySimulator
import os
import logging

os.chmod(os.getcwd(),0o777)
logging.basicConfig(filename="log.txt",filemode='w', level=logging.ERROR)

############
## Problem definition

xosc = None
var_min = None
var_max = None
featureNames = None
simulateFcn = None
## scenario parameters
simTime=10
samplingTime = 1

## EXAMPLE DUMMY SIMULATOR
def setExp1():
    # (x,y, orientation, velocity) for both actors -> 8 genoms
    # if  lower and upper boundary are equal mutation throws error
    global xosc,var_min,var_max,featureNames,simulateFcn,fitnessFcn
    xosc = None
    var_min = [ 0, 0, 0,1, 100, 100, 0,5]
    var_max = [ 100, 200, 200, 50, 110, 200,20,10]

    featureNames = [
                "egoX",
                "egoY",
                "orientationEgo",
                "velocityEgo",
                "objX",
                "objY",
                "orientationObj",
                "velocityObj"
    ]
    fitnessFcn = fitness.fitness_basic_two_actors
    simulateFcn = DummySimulator.simulateBatch

## EXAMPLE CARLA SIMULATOR
def setExp2():
    global xosc,var_min,var_max,featureNames,simulateFcn,fitnessFcn
    xosc = os.getcwd() + "/scenarios/FollowLeadingVehicle_generic.xosc"
    var_min = [0]
    var_max = [10]
    featureNames = ["leadingSpeed"]
    fitnessFcn = fitness.fitness_min_distance_two_actors
    #fitnessFcn = fitness.fitness_random
    simulateFcn = CarlaSimulator.simulateBatch

def setExp3():
    # example to test integration (provided scenario is already an instance)
    global xosc,var_min,var_max,featureNames,simulateFcn,fitnessFcn
    xosc = os.getcwd() + "/scenarios/test_1_short.xosc"
    featureNames = ["dummy"]
    var_min = [0]
    var_max = [10]
    fitnessFcn = fitness.fitness_min_distance_two_actors
    simulateFcn = CarlaSimulator.simulateBatch
    
def setExp4():
    global xosc,var_min,var_max,featureNames,simulateFcn
    pass
    # TODO
    #simulateFcn = PrescanSimulator.simulate


def setExp5():
    # (x,y, orientation, velocity) for both actors -> 8 genoms
    # if  lower and upper boundary are equal mutation throws error
    global xosc,var_min,var_max,featureNames,simulateFcn,fitnessFcn
    xosc = None
    var_min = [ 0, 0, 0,1, 100, 100, 0,5]
    var_max = [ 100, 200, 200, 50, 110, 200,20,10]

    featureNames = [
                "egoX",
                "egoY",
                "orientationEgo",
                "velocityEgo",
                "objX",
                "objY",
                "orientationObj",
                "velocityObj"
    ]
    fitnessFcn = fitness.fitness_random
    simulateFcn = DummySimulator.simulateBatch

def criticalFcn(fit,simout):
    if((simout.otherParams['isCollision'] == True) or (fit[0] > 2 and fit[0] < 7  or  fit[0] < 0.9)):
        return True
    else:
        return False
        
criticalFcn = criticalFcn
nFitnessFcts = 1
initialPopulationSize = 4
nGenerations = 1

###### set experiment

setExp1()
#setExp2()
#setExp3()

#######

if __name__ == "__main__":
    pop, critical, stats = nsga2_TC(initialPopulationSize, 
                    nGenerations,
                    var_min, 
                    var_max, 
                    fitnessFcn, 
                    nFitnessFcts, 
                    criticalFcn, 
                    simulateFcn,
                    featureNames,
                    xosc,
                    initial_pop=[],
                    simTime=simTime,
                    samplingTime=samplingTime)

    print("# individuals: "+ str(len(pop)))
    print("# most critical:" + str([str(entry) for entry in zip(featureNames,pop[0])]))
    print("# most critical fitness: " + str(pop[0].fitness.values) )
    
