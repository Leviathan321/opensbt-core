#import simulation.prescan_simulation
#from simulation.prescan_simulation import PrescanSimulator
from algorithm.nsga2_TC import *
from fitness_functions import fitness
#from simulation.carla_simulation import CarlaSimulator
from simulation.dummy_simulation import DummySimulator
import os
import logging

from visualization import plotter

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
simTime=50
samplingTime = 1

## EXAMPLE DUMMY SIMULATOR
def setExp1():
    # (x,y, orientation, velocity) for both actors -> 8 genoms
    # if  lower and upper boundary are equal mutation throws error
    global xosc,var_min,var_max,featureNames,simulateFcn,fitnessFcn
    xosc = "DummyExperiment"
    var_min = [ 0, 0, 150,1, 100, 100, 0,5]
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
    fitnessFcn = fitness.fitness_min_distance_two_actors_carla
    #fitnessFcn = fitness.fitness_random
    simulateFcn = CarlaSimulator.simulateBatch

def setExp3():
    # example to test integration (provided scenario is already an instance)
    global xosc,var_min,var_max,featureNames,simulateFcn,fitnessFcn
    xosc = os.getcwd() + "/scenarios/test_1_short.xosc"
    featureNames = ["dummy"]
    var_min = [0]
    var_max = [10]
    fitnessFcn = fitness.fitness_min_distance_two_actors_carla
    simulateFcn = CarlaSimulator.simulateBatch

'''
    Prescan simulation
'''
def setExp4():
    global xosc,var_min,var_max,featureNames,simulateFcn,fitnessFcn,simTime
    xosc =  os.getcwd() + "/../Experiments/Experiment_1/Experiment_1.pb"

    # dummy values
    var_min = [ 0, 0]
    var_max = [ 100, 200]

    featureNames = [
                "objY",
                "orientationObj",
    ]
    simTime = 2
    fitnessFcn = fitness.fitness_min_distance_two_actors_prescan
    simulateFcn = PrescanSimulator.simulateBatch

def setExp5():
    global xosc,var_min,var_max,featureNames,simulateFcn,fitnessFcn,simTime
    #xosc =  os.getcwd() + "/../Experiments/ACC_ISO_test_005\ACC_ISO_test_005.pb"
    xosc = "C:/Users/Public/Documents/Experiments/TestScenarios/ACC/ACC_ISO_test_005/ACC_ISO_test_005.pb"
    # dummy values
    var_min = [10,20]
    var_max = [40,100]

    featureNames = [
                "otherVelocity", # in m/w
                "otherStartPosition" # in m
    ]
    simTime = 5
    fitnessFcn = fitness.fitness_min_distance_two_actors_prescan
    simulateFcn = PrescanSimulator.simulateBatch

def criticalFcn(fit,simout):
    if((simout.otherParams['isCollision'] == True) or (fit[0] > 2 and fit[0] < 7  or  fit[0] < 0.9)):
        return True
    else:
        return False

criticalFcn = criticalFcn
nFitnessFcts = 2
initialPopulationSize = 4
nGenerations = 1

###### set experiment

setExp1()
#setExp2()
#setExp3()
#setExp4()
#setExp5()

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
