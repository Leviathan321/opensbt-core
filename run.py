#import simulation.prescan_simulation
#from simulation.prescan_simulation import PrescanSimulator

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
fitnessFcns = None
criticalFcn = None
## scenario parameters
simTime=50
samplingTime = 0.5
optimize = []

''' 
EXAMPLE DUMMY SIMULATOR
'''

def setExp1():
    # (x,y, orientation, velocity) for both actors -> 8 genoms
    # if  lower and upper boundary are equal mutation throws error
    global xosc,var_min,var_max,featureNames,simulateFcn,fitnessFcns,optimize,criticalFcn
    xosc = "DummyExperiment"
    var_min = [ 0,1, 0,5]
    var_max = [ 360, 50,360,10]
    featureNames = [
                "orientationEgo",
                "velocityEgo",
                "orientationObj",
                "velocityObj"
    ]
    fitnessFcns = [fitness.fitness_basic_two_actors]
    optimize = ['min']
    simulateFcn = DummySimulator.simulateBatch
    criticalFcn = critical.criticalFcn

''' 
EXAMPLE CARLA SIMULATOR
'''

def setExp2():
    global xosc,var_min,var_max,featureNames,simulateFcn,fitnessFcns,optimize,criticalFcn
    xosc = os.getcwd() + "/scenarios/FollowLeadingVehicle_generic.xosc"
    var_min = [0]
    var_max = [10]
    featureNames = ["leadingSpeed"]
    fitnessFcns = [fitness.fitness_min_distance_two_actors_carla]
    optimize = ['min']
    simulateFcn = CarlaSimulator.simulateBatch
    criticalFcn = critical.criticalFcn

'''
EXAMPLE PRESCAN SIMULATOR
'''

''' Experiment from DENSO with writing trajectories at runtime in .csv file'''
def setExp3():
    global xosc,var_min,var_max,featureNames,simulateFcn,fitnessFcns,simTime,optimize,criticalFcn
    xosc =  os.getcwd() + "/../experiments/Leuven_AVP_ori_18b_210722/Leuven_AVP_ori/Demo_AVP.pb"
    
    # dummy values
    var_min = [0.5,0,0,1]
    var_max = [0.6,1,0.2,3]

    # use prefix in featurename to define the actor correpondence (e.g. Other_<par>, Ego_<par>)
    featureNames = [ 
                "Ego_HostVelGain", # in m/s
                "Other_Velocity_mps", # in m/s
                "Other_Time_s", # in s,
                "Other_Accel_mpss"
    ]

    #simTime = 10
    fitnessFcns = [fitness.fitness_min_distance_two_actors_prescan]
    optimize = ['min']
    simulateFcn = PrescanSimulator.simulateBatch_compiled_csv
    criticalFcn = critical.criticalFcn


#######

experimentsSwitcher = {
   1: setExp1,
   2: setExp2,
   3: setExp3
}

examplesType = {
  1: SimulationType.DUMMY,
   2: SimulationType.CARLA,
   3: SimulationType.PRESCAN
}


######

''' Set default search parameters '''

initialPopulationSize = 4
nGenerations = 1
algorithm = 0
timeSearch = 10

########

parser = argparse.ArgumentParser(description="Pass parameters for search.")
parser.add_argument('-e', dest='expNumber', type=str, default="1", action='store', help='Hardcoded example scenario to use (possible 1, 3).')
parser.add_argument('-i', dest='nIterations', type=int, default=nGenerations, action='store', help='Number iterations to perform.')
parser.add_argument('-n', dest='sizePopulation', type=int, default=initialPopulationSize, action='store', help='The size of the initial population of scenario candidates.')
parser.add_argument('-a', dest='algorithm', type=int, default=algorithm, action='store', help='The algorithm to use for search, 0 for nsga2, 1 for nsga2dt.')
parser.add_argument('-t', dest='timeSearch', type=int, default=timeSearch, action='store', help='The time to use for search with nsga2-DT (actual search time can be above the threshold, since algorithm might perform nsga2 iterations, when time limit is already reached.')
parser.add_argument('-f', dest='xosc', type=str, action='store', help='The path to the .pb file of the Prescan Experiment.')

parser.add_argument('-min', dest='var_min', nargs="+", type=float, action='store', help='The lower bound of each parameter.')
parser.add_argument('-max', dest='var_max', nargs="+", type=float, action='store', help='The upper bound of each parameter.')
parser.add_argument('-m', dest='feature_names', nargs="+", type=str, action='store', help='The names of the features to modify.')


args = parser.parse_args()

#######

if not args.expNumber is None and not args.xosc is None:
    print("Flags set not correctly: Experiment file and example experiment cannot be set at the same time")
    sys.exit()
elif args.expNumber is None and args.xosc is None:
    print("Flags set not correctly: No file is provided or no example experiment selected.")
    sys.exit()

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
        feature_names = ["feature_"+ str(i) for i in range(len(var_min))]
    
    simTime = 10

    if xosc.endswith('.pb'):
        fitnessFcns = [fitness.fitness_min_distance_two_actors_prescan]
        simulateFcn = PrescanSimulator.simulateBatch
    elif xosc.endswith('.xosc') :
        fitnessFcns = [fitness.fitness_min_distance_two_actors_carla]
        simulateFcn = CarlaSimulator.simulateBatch
    else:
        print("-- File is not supported.")
        sys.exit()

'''
override params if set by user
'''

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
#######

if __name__ == "__main__":
    pop = None
    execTime = None
    
    if algorithm == 0:
        pop, all_solutions, critical, all_simoutput, stats, execTime = nsga2_TC(initialPopulationSize,
                        nGenerations,
                        var_min,
                        var_max,
                        fitnessFcns,
                        optimize,
                        criticalFcn,
                        simulateFcn,
                        featureNames,
                        xosc,
                        initial_pop=[],
                        simTime=simTime,
                        samplingTime=samplingTime,
                        mode="standalone"
                        )
    elif algorithm == 1:
           pop, all_solutions, critical, all_simoutput, execTime = nsga2_DT(initialPopulationSize, 
                    nGenerations,
                    var_min, 
                    var_max, 
                    fitnessFcns, 
                    optimize, 
                    criticalFcn, 
                    simulateFcn,
                    featureNames,
                    xosc,
                    simTime=simTime,
                    samplingTime=samplingTime,
                    time_search=timeSearch
                    )

    print("# individuals: "+ str(len(pop)))
    print("# most critical:" + str([str(entry) for entry in zip(featureNames,pop[0])]))
    print("# most critical fitness: " + str(pop[0].fitness.values) )
    print("# execution time: " + str(execTime))