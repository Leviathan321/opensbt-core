# import simulation.prescan_simulation
# from simulation.prescan_simulation import PrescanSimulator

from pickletools import optimize
from algorithm.nsga2_TC import *
from algorithm.nsga2_DT import *
from evaluation import fitness
from evaluation import critical

from simulation.dummy_simulation import DummySimulator
import os
import logging
import re
from visualization import plotter
import argparse
from enum import Enum

os.chmod(os.getcwd(),0o777)
logging.basicConfig(filename="log.txt",filemode='w', level=logging.ERROR)

import sys

#########

# supported simulations

class SimulationType(Enum):
    DUMMY = 0, 
    CARLA =  1,
    PRESCAN = 2


############
## Problem definition

xosc = None
var_min = None
var_max = None
featureNames = None
simulateFcn = None
fitnessFcn = None
criticalFcn = None
## scenario parameters
simTime=50
samplingTime = 1
optimize = []

''' 
EXAMPLE DUMMY SIMULATOR
'''

def setExp1():
    # (x,y, orientation, velocity) for both actors -> 8 genoms
    # if  lower and upper boundary are equal mutation throws error
    global xosc,var_min,var_max,featureNames,simulateFcn,fitnessFcn,optimize,criticalFcn
    xosc = "DummyExperiment"
    var_min = [ 0,1, 0,5]
    var_max = [ 360, 50,360,10]

    featureNames = [
                "orientationEgo",
                "velocityEgo",
                "orientationObj",
                "velocityObj"
    ]
    #fitnessFcn = fitness.fitness_basic_two_actors
    #fitnessFcn = fitness.fitness_random_dual
    fitnessFcn = fitness.fitness_basic_two_actors
    optimize = ['min']
    simulateFcn = DummySimulator.simulateBatch
    criticalFcn = critical.criticalFcn

''' 
EXAMPLE CARLA SIMULATOR
'''

def setExp2():
    global xosc,var_min,var_max,featureNames,simulateFcn,fitnessFcn,optimize,criticalFcn
    xosc = os.getcwd() + "/scenarios/FollowLeadingVehicle_generic.xosc"
    var_min = [0]
    var_max = [10]
    featureNames = ["leadingSpeed"]
    fitnessFcn = fitness.fitness_min_distance_two_actors_carla
    optimize = ['min']
    #fitnessFcn = fitness.fitness_random
    simulateFcn = CarlaSimulator.simulateBatch
    criticalFcn = critical.criticalFcn


def setExp3():
    # example to test integration (provided scenario is already an instance)
    global xosc,var_min,var_max,featureNames,simulateFcn,fitnessFcn,criticalFcn
    xosc = os.getcwd() + "/scenarios/test_1_short.xosc"
    featureNames = ["dummy"]
    var_min = [0]
    var_max = [10]
    fitnessFcn = fitness.fitness_min_distance_two_actors_carla
    simulateFcn = CarlaSimulator.simulateBatch
    criticalFcn = critical.criticalFcn


'''
    Prescan simulation
'''
def setExp4():
    global xosc,var_min,var_max,featureNames,simulateFcn,fitnessFcn,simTime,criticalFcn
    xosc =  os.getcwd() + "/../experiments/Experiment_1/Experiment_1.pb"

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
    criticalFcn = critical.criticalFcn


def setExp5():
    global xosc,var_min,var_max,featureNames,simulateFcn,fitnessFcn,simTime,criticalFcn
    #xosc =  os.getcwd() + "/../Experiments/ACC_ISO_test_005\ACC_ISO_test_005.pb"
    xosc = "C:/Users/Public/Documents/Experiments/TestScenarios/ACC/ACC_ISO_test_005/ACC_ISO_test_005.pb"
    # dummy values
    var_min = [10,20]
    var_max = [40,100]

    featureNames = [
                "otherVelocity", # in m/s
                "otherStartPosition" # in m
    ]
    simTime = 5
    fitnessFcn = fitness.fitness_min_distance_two_actors_prescan
    simulateFcn = PrescanSimulator.simulateBatch
    criticalFcn = critical.criticalFcn


def setExp6():
    global xosc,var_min,var_max,featureNames,simulateFcn,fitnessFcn,simTime,criticalFcn
    xosc =  os.getcwd() + "/../experiments/Demo_AVP_sc1/Demo_AVP_sc1.pb"

    # dummy values
    var_min = [1,1,1]
    var_max = [10,10,10]

    featureNames = [
                "otherVelocity", # in m/s
                "otherStartBackingOutTime" # in s,
                "egoVelocity" # in m/s
    ]
    simTime = 10
    fitnessFcn = fitness.fitness_min_distance_two_actors_prescan
    simulateFcn = PrescanSimulator.simulateBatch
    criticalFcn = critical.criticalFcn


def setExp7():
    global xosc,var_min,var_max,featureNames,simulateFcn,fitnessFcn,simTime,criticalFcn
    xosc =  os.getcwd() + "/../experiments/Demo_AVP_sc2/Demo_AVP_sc2.pb"

    # dummy values
    var_min = [1,1,1]
    var_max = [3,10,10]

    featureNames = [
                "otherVelocity", # in m/s
                "otherStartMovementTime" # in s,
                "egoVelocity" # in m/s
    ]
    simTime = 10
    fitnessFcn = fitness.fitness_min_distance_two_actors_prescan
    simulateFcn = PrescanSimulator.simulateBatch
    criticalFcn = critical.criticalFcn


'''
    HEEDS PRESCAN EXPERIMENT (COMPILED)
'''

def setExp8():
    global xosc,var_min,var_max,featureNames,simulateFcn,fitnessFcn,simTime,optimize,criticalFcn
    xosc =  os.getcwd() + "/../experiments/HeedWorkshopExperiment/Demo_AVP_cs/Leuven_AVP_ori/Demo_AVP.pb"
    
    # dummy values
    var_min = [1,1,0.1,1]
    var_max = [1.5,3,0.5,3]

    # use prefix in featurename to define the actor correpondence (e.g. Other_<par>, Ego_<par>)
    featureNames = [ 
                "Ego_HostVelGain", # in m/s
                "Other_Velocity_mps", # in m/s
                "Other_Time_s", # in s,
                "Other_Accel_mpss"
    ]

    # TODO modify simulink model to store all trajectories of all actors (trajectory.mat is stored only for ego?)
    # TODO modify simulink model to set max simulation time

    #simTime = 10
    fitnessFcn = fitness.fitness_min_ttc_vimpact
    optimize = ['min','max']
    simulateFcn = PrescanSimulator.simulateBatch_compiled
    criticalFcn = critical.criticalFcn

''' Experiment from DENSO with writing trajectories at runtime in .csv file'''
def setExp9():
    global xosc,var_min,var_max,featureNames,simulateFcn,fitnessFcn,simTime,optimize,criticalFcn
    xosc =  os.getcwd() + "/../experiments/Demo_AVP_cs_writeexe/Leuven_AVP_ori/Demo_AVP.pb"
    
    # dummy values
    var_min = [1,1,0.4,1]
    var_max = [1.2,3,0.5,3]

    # use prefix in featurename to define the actor correpondence (e.g. Other_<par>, Ego_<par>)
    featureNames = [ 
                "Ego_HostVelGain", # in m/s
                "Other_Velocity_mps", # in m/s
                "Other_Time_s", # in s,
                "Other_Accel_mpss"
    ]

    # TODO modify simulink model to store all trajectories of all actors (trajectory.mat is stored only for ego?)
    # TODO modify simulink model to set max simulation time

    #simTime = 10
    fitnessFcn = fitness.fitness_min_ttc_vimpact
    optimize = ['min','max']
    simulateFcn = PrescanSimulator.simulateBatch_compiled
    criticalFcn = critical.criticalFcn


#######

experimentsSwitcher = {
   1: setExp1,
   2: setExp2,
   3: setExp3,
   4: setExp4,
   5: setExp5,
   6: setExp6,
   7: setExp7,
   8: setExp8,
   9: setExp9
}

examplesType = {
  1: SimulationType.DUMMY,
   2: SimulationType.CARLA,
   3: SimulationType.CARLA,
   4: SimulationType.PRESCAN,
   5: SimulationType.PRESCAN,
   6: SimulationType.PRESCAN,
   7: SimulationType.PRESCAN,
   8: SimulationType.PRESCAN,
   9: SimulationType.PRESCAN
}

######

''' Set default search parameters '''
initialPopulationSize = 4
nGenerations = 1
algorithm= 0
timeSearch = 10

########

parser = argparse.ArgumentParser(description="Pass parameters for search.")
parser.add_argument('-e', dest='expNumber', type=str, action='store', help='Hardcoded example scenario to use (possible 1, 9).')
parser.add_argument('-i', dest='nIterations', type=int, default=nGenerations, action='store', help='Number iterations to perform.')
parser.add_argument('-n', dest='sizePopulation', type=int, default=initialPopulationSize, action='store', help='The size of the initial population of scenario candidates.')
parser.add_argument('-a', dest='algorithm', type=int, default=0, action='store', help='The algorithm to use for search, 0 for nsga2, 1 for nsga2dt')
parser.add_argument('-t', dest='timeSearch', type=int, default=timeSearch, action='store', help='The time to use for search with nsga2-DT (actual search time can be above the threshold, since algorithm might perform nsga2 iterations, when time limit is already reached')
parser.add_argument('-f', dest='xosc', type=str, action='store', help='The path to the .pb file of the Prescan Experiment')

parser.add_argument('-min', dest='var_min', nargs="+", type=float, action='store', help='The upper bound of each parameter')
parser.add_argument('-max', dest='var_max', nargs="+", type=float, action='store', help='The lower bound of each parameter')
parser.add_argument('-m', dest='feature_names', nargs="+", type=str, action='store', help='The names of the features to modify')


args = parser.parse_args()

#######

# override params if set by user
print(args)
if not args.expNumber is None and not args.xosc is None:
    print("Flags set not correctly: Experiment file and example experiment cannot be set at the same time")
    sys.exit()
elif args.expNumber is None and args.xosc is None:
    print("Flags set not correctly: No file is provided or no example experiment selected.")
    sys.exit()
if not args.sizePopulation is None:
    initialPopulationSize = args.sizePopulation
if not args.nIterations is None:
    nGenerations = args.nIterations
if not args.algorithm is None:
    algorithm = args.algorithm
if not args.timeSearch is None:
    timeSearch = args.timeSearch
if not args.var_max is None:
    var_max = args.var_max
if not args.var_min is None:
    var_min = args.var_min
if not args.var_min is None:
    var_min = args.var_min
if not args.feature_names is None:
    feature_names = args.feature_names
###### set experiment
# pass as first argument the number of the experiment to use

selectedExperiment = "" 

####### have indiviualized imports 

if not args.expNumber is None:
    # expNumber provided
    print("Using given experiment")
    selExpNumber = re.findall("[1-9]+",args.expNumber)[0]
    if (examplesType[int(selExpNumber)].value == SimulationType.CARLA.value ):
        from simulation.carla_simulation import CarlaSimulator
        print("carla libs imported")
    elif (examplesType[int(selExpNumber)].value == SimulationType.PRESCAN.value ):
        # import simulation.prescan_simulation
        # from simulation.prescan_simulation import PrescanSimulator
        # print("prescan libs imported")
        pass
    else:
        pass

    ####### set experiment
    experimentsSwitcher.get(int(selExpNumber))()

elif (not args.xosc is None):
    xosc = args.xosc

    print("-- Experiment provided by file")

    if args.var_min is None:
        print("-- Minimal bounds for search are not set.")
        sys.exit()
    
    if args.var_max is None:
        print("-- Maximal bounds for search are not set.")
        sys.exit()

    # set feature_names 
    if args.feature_names is None:
        featureNames = ["feature_"+ str(i) for i in range(len(var_min))]
    
    simTime = 10

    if xosc.endswith('.pb'):
        fitnessFcn = fitness.fitness_min_distance_two_actors_prescan
        simulateFcn = PrescanSimulator.simulateBatch
    elif xosc.endswith('.xosc') :
        #import libs 
        from simulation.carla_simulation import CarlaSimulator

        fitnessFcn = fitness.fitness_min_distance_two_actors_carla
        simulateFcn = CarlaSimulator.simulateBatch
    else:
        print("-- File is not supported.")
        sys.exit()

    criticalFcn = critical.criticalFcn
#######

if __name__ == "__main__":
    pop = None
    critial = None
    execTime = None
    
    if algorithm == 0:
        pop, all_solutions, critical, stats, execTime = nsga2_TC(initialPopulationSize,
                        nGenerations,
                        var_min,
                        var_max,
                        fitnessFcn,
                        optimize,
                        criticalFcn,
                        simulateFcn,
                        featureNames,
                        xosc,
                        initial_pop=[],
                        simTime=simTime,
                        samplingTime=samplingTime,
                        mode="standalone",
                        )
    elif algorithm == 1:
           pop, all_solutions, critical, execTime = nsga2_DT(initialPopulationSize, 
                    nGenerations,
                    var_min, 
                    var_max, 
                    fitnessFcn, 
                    optimize, 
                    criticalFcn, 
                    simulateFcn,
                    featureNames,
                    xosc,
                    simTime=simTime,
                    time_search=timeSearch
                    )

    print("# individuals: "+ str(len(pop)))
    print("# most critical:" + str([str(entry) for entry in zip(featureNames,pop[0])]))
    print("# most critical fitness: " + str(pop[0].fitness.values) )
    print("# execution time: " + str(execTime))
