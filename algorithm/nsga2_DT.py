
from algorithm.nsga2_TC import nsga2_TC
import algorithm.regions as regions
import time
import os
from pathlib import Path
from datetime import datetime
from utils.structures import update_list_unique
import random
import numpy as np

from deap import algorithms
from deap import base
from deap.benchmarks.tools import diversity, convergence, hypervolume
from deap import creator
from deap import tools

from matplotlib import pyplot as plt
from simulation.simulator import SimulationOutput
from visualization import writer

## simulation parameters
simTime=10
samplingTime=60

## genetic algorithm parameters

nGenerations = 20 # 22
initialPopulationSize = 1 # should be % 4 == 0, when using tools.selTournamentDCD
crossoverProbability = 0.6
mutationRate = 0.2

SUPPRESS_PLT = True
DEBUG = True
RESULTS_FOLDER =  "/results/simoutput/"
PRECISION = 10

''' Input:
        initialPopulationSize: number of best scenario instances to select each iteration
        nGenerations: number of generations for nsga2 to search
        var_min: minimal bound of each parameter
        var_max: maximal bound of each parameter
        fitnessFcn: pointer to a fitness function for scenario evaluation
        optimize: array of min/max declaration for each element of result of fitness fcn
        criticalFcn:  pointer to a criticality function for post-simulative scenario selection
        simulateFcn: pointer to a function to simulate scenario instance
        featureNames: array of parameters to be variied
        xosc: scenario name
        simTime: simulation time
        initial_pop=[]: initial population to start search
        num_rep = 3: number of iterations with DT classifier (outer loop) 
'''

def nsga2_DT(initialPopulationSize, 
            nGenerations,
                    var_min, 
                    var_max, 
                    fitnessFcn, 
                    optimize, 
                    criticalFcn, 
                    simulateFcn,
                    featureNames,
                    xosc,
                    simTime,
                    time_search = 10):   
    algorithmName="nsga2DT"
    subFolderName = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    outputPath = str(os.getcwd()) + RESULTS_FOLDER + Path(xosc).stem + os.sep + subFolderName
    simoutAll = {}
    
    all_critical_dict = {}
    all_solutions = []
    all_best = [] # store the population after a complete NSGA2 run

    #######

    toolbox = base.Toolbox()

    def uniform(low, up):
        return [round(random.uniform(a, b),PRECISION) for a, b in zip(low, up)]
    
    ### For population initialization

    # set which variable has to be minimized or maximized, default: minimize

    weights = ()
    for v in optimize:
        if v=='max':
            weights = weights + (+1,)
        else:
            weights = weights + (-1,)
    creator.create("FitnessMin", base.Fitness, weights=weights)
    creator.create("Individual", list, typecode='d', fitness=creator.FitnessMin)
    toolbox.register("attr_float", uniform, var_min, var_max)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    #####
 
    bounds = [[var_min,var_max]] # full search space
    solutions_region =  [toolbox.population(n=initialPopulationSize)]

    # TODO assure that no duplicates in population

    print(f"+++ Initial bounds: {bounds}")


    # iterative over all regions 
    # TODO constrain by time

    t_start_search = time.time()

    dt_iteration = 1
    while time.time() - t_start_search < time_search:
        
        for i in range(0,len(bounds)):
            bound = bounds[i]
            print(bound[0])
            print(bound[1])
            pop_search = solutions_region[i]
            # requirement by tournament selection len(pop) is % 4
            if  len(solutions_region[i]) % 4:
                n = findNextDivident(len(solutions_region[i]),4)
                print("len pop regions i: " + str(len(solutions_region[i])))
                print("++ next divident: " + str(n))
                pop_search = fillUpPopulation(toolbox,solutions_region[i],all_solutions, n - len(solutions_region[i]),bound[0],bound[1])

            assert len(pop_search) % 4 == 0

            # Search in critical reagion with nsga2
            best_run, all_solutions_run, criticalDict_Sub, stats2,execTimeTc = nsga2_TC(
                    initialPopulationSize=initialPopulationSize,
                    nGenerations=nGenerations,
                    fitnessFcn=fitnessFcn, 
                    optimize=optimize, 
                    criticalFcn=criticalFcn, 
                    simulateFcn=simulateFcn,
                    featureNames=featureNames,
                    xosc=xosc,
                    initial_pop=pop_search, 
                    var_min=bound[0], 
                    var_max=bound[1],
                    mode="submode",
                    simulationOutputAll=simoutAll)

            update_list_unique(all_best,best_run)
            update_list_unique(all_solutions,all_solutions_run)

            all_critical_dict.update(criticalDict_Sub)
            
            print(f"Size_critical_dict: {len(all_critical_dict)}")
            print(f"Size all_solutions: {len(all_solutions)}")

            assert len(all_solutions) == len(all_critical_dict)

        # Apply DT to get new critical regions
        solutions_region, newBounds = regions.getCriticalRegions(all_solutions,all_critical_dict, \
            var_min, \
            var_max, \
            feature_names=featureNames, \
            outputPath=outputPath, \
            name="iter_" + str(dt_iteration)
        )
        dt_iteration = dt_iteration + 1
        bounds = newBounds

        # No critical regions
        if len(bounds) == 0:
            print("+++ No critical regions output by DT")
            break
        else:
            #print(pop_regions)
            print("+++ critical regions found:" + str(len(bounds)))

    all_best.sort(key=lambda x: x.fitness.values)

    t_end = time.time()
    execTime = t_end - t_start_search
    print("++ Writing results ++") 

    writer.write_results(simoutAll,algorithmName,all_best,xosc,featureNames,execTime,outputPath,xosc, all_best)   

    return all_best, all_solutions, all_critical_dict, execTime

def findNextDivident(n,k):
    i = 1
    while k*i < n:
        print( k*i)
        i = i + 1
    return k*i

''' The tournament crossover mutation operator requires that the population size is % 4.
    In a critical region the number of inds is variable. Therefore we add individuals
    by 
    
    1) checking remaining inds from all solutions
    2) adding randomly
    
    pop: sorted population
    givenPop: sorted subset of pop
    num <= len(all) - len(givenPop)
'''
def fillUpPopulation(toolbox,givenPop,all,num,bound_min,bound_max):
    print(f"length all pop: {len(all)}")
    print(f"length given pop: {len(givenPop)}")
    print(f"required number: {num}")
    result = givenPop
    i = 0
    for ind in all:
        
        if (ind not in givenPop and \
            is_in_bound(ind,bound_min,bound_max) and  \
            i < num):

            result.append(ind)
            i = i + 1
    if i < num:
        # if no candidates available, create randomly

        def uniform_extra_pop(low, up):
            return [round(random.uniform(a, b),PRECISION) for a, b in zip(low, up)]
    
        toolbox.register("attr_float_extra", uniform_extra_pop, bound_min, bound_max)
        toolbox.register("individual_extra", tools.initIterate, creator.Individual, toolbox.attr_float_extra)
        toolbox.register("population_extra", tools.initRepeat, list, toolbox.individual_extra)

        rand_pop = toolbox.population_extra(n=(num -i))

        result.extend(rand_pop)
    print(f"length resulting pop: {len(result)}")
    return result

def is_in_bound(ind,bound_min,bound_max):
    return np.less_equal(np.array(ind) ,np.array(bound_max)).all()  \
        and np.greater_equal(np.array(ind) ,np.array(bound_min)).all()
