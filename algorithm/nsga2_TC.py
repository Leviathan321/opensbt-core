#from simulation import prescan_simulation
from cmath import e
from datetime import datetime
from pathlib import Path
from pickle import FALSE

import random
import matplotlib.pyplot as plt
import numpy as np
import math

import sys

from deap import algorithms
from deap import base
from deap.benchmarks.tools import diversity, convergence, hypervolume
from deap import creator
from deap import tools
from algorithm.operators import crossover

from evaluation import fitness
from dummySimulation.dynamics import basic_dynamics as bd
from simulation.simulator import SimulationOutput
from visualization import writer

from utils.structures import update_list_unique

from visualization import plotter
from random import randrange
import os
import time

## genetic algorithm parameters
crossoverProbability = 0.6
mutationRate = 0.2

EVALUATE_IN_BATCH = True
DEBUG = False
RESULTS_FOLDER = "/results/"

def nsga2_TC(initialPopulationSize,
            nGenerations, 
                    var_min, 
                    var_max, 
                    fitnessFcn, 
                    optimize, 
                    criticalFcn, 
                    simulateFcn,
                    featureNames,
                    xosc,
                    initial_pop,
                    simTime,
                    samplingTime,
                    mode="standalone",
                    simulationOutputAll = {}):   

    algorithmName="nsga2"

    start = time.time()

    assert np.less_equal(var_min,var_max).all()

    criticalDict = {}
     
    all_solutions = []

    ## fitness fcts and critical fct
    def evaluateFcnBatch(individuals):
        simouts = simulateFcn(individuals,featureNames, xosc, simTime=simTime,samplingTime=samplingTime)
        fits = []
        for individual,simout in zip(individuals,simouts):
            fit = fitnessFcn(simout)
            simulationOutputAll[str(individual)] = simout
            #fit = fitness.fitness_min_distance_two_actors(simout)
            time = int(round(datetime.now().timestamp()))
            random.seed(time)

            if isinstance(fit, float):
                value = (fit,)
            else:
                value = fit  
                
            if str(individual) not in criticalDict.keys():
                criticalDict[str(individual)] = isCritical(value,simout)

            if individual not in all_solutions:
                all_solutions.append(individual)
            
            assert( len(all_solutions) == len(criticalDict))

            # if not SUPPRESS_PLT:
            #     savePath = str(os.getcwd()) + "/results/simoutput/" + "test" + "_" + str(individual) + "_fit_" + str(value) + ".png"
            #     plotter.plotOutput(simout=simout,features=featureNames,featureValues=individual,savePath=savePath)
            fits.append(value)        
        
        return fits

    def isCritical(fit, simout: SimulationOutput):
        return criticalFcn(fit=fit,simout=simout)

    BOUND_LOW, BOUND_UP = var_min, var_max
    NDIM = len(var_min)

    print("NSGAII - STARTED")
    print("logical scenario: " + str(xosc))
    print("population size: " + str(len(initial_pop)))
    print("number generations: " + str(nGenerations))
    print("lower bound: " + str(BOUND_LOW))
    print("upper bound: " + str(BOUND_UP))


    if( not hasattr(creator,"FitnessMin")):
        # set which variable has to be minimized or maximized, default: minimize
        weights = ()
        for v in optimize:
            if v=='max':
                weights = weights + (+1,)
            else:
                weights = weights + (-1,)
        creator.create("FitnessMin", base.Fitness, weights=weights)
    if( not hasattr(creator,"Individual")):
        creator.create("Individual", list, typecode='d', fitness=creator.FitnessMin)

    toolbox = base.Toolbox()

    def uniform(low, up):
        return [random.uniform(a, b) for a, b in zip(low, up)]
    

    toolbox.register("attr_float", uniform, BOUND_LOW, BOUND_UP)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    #toolbox.register("evaluate", evaluateFcn)
    toolbox.register("mate", crossover.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=20)
    #toolbox.register("mate", tools.cxMessyOnePoint)

    toolbox.register("mutate", tools.mutPolynomialBounded, eta=20,low=BOUND_LOW, up=BOUND_UP, indpb=mutationRate)
    toolbox.register("select", tools.selNSGA2)
    # selected individuals using selNSGA2 occure only once in the result */

    seed=None
    random.seed(seed)

    NGEN = nGenerations

    CXPB = crossoverProbability

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    # stats.register("avg", numpy.mean, axis=0)
    # stats.register("std", numpy.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"



    if len(initial_pop)==0:
        MU = initialPopulationSize
        print("population initialized")
        pop = toolbox.population(n=MU)
    else:
        MU = len(initial_pop)
        pop = initial_pop

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    
    if DEBUG:
        print("invalid_ind: " + str(invalid_ind))

    if EVALUATE_IN_BATCH:
        fitnesses = evaluateFcnBatch(invalid_ind)
    else:
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)

    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # This is just to assign the crowding distance to the individuals
    # no actual selection is done
    pop = toolbox.select(pop, len(pop))

    record = stats.compile(pop)
    logbook.record(gen=0, evals=len(invalid_ind), **record)
    
    print(logbook.stream)


    #all_solutions.extend(pop)
    
    assert np.array([str(all_solutions[i]) == list(criticalDict.keys())[i]  for i in range(len(all_solutions))]).all()

    # Begin the generational process
    for gen in range(1, NGEN):
        # Vary the population
        # offspring = tools.selNSGA2(pop,int(len(pop)/2))
        # "Each individual from the input list won’t be selected more than twice using tournament selection."
        offspring = tools.selTournamentDCD(pop, len(pop))
        offspring = [toolbox.clone(ind) for ind in offspring]

        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            
            try:
                assert( not True in [math.isnan(v) for v in ind1])
                assert( not True in [math.isnan(v) for v in ind2])
            except AssertionError as e:
                print("Before crossover")
                print(ind1)
                print(ind2)
                raise e

            if random.random() <= CXPB:
                toolbox.mate(ind1, ind2)

            try:
                assert( not True in [math.isnan(v) for v in ind1])
                assert( not True in [math.isnan(v) for v in ind2])
            except AssertionError as e:
                print("After crossover")
                print(ind1)
                print(ind2)
                raise e

            toolbox.mutate(ind1)
            toolbox.mutate(ind2)

            try:
                assert( not True in [math.isnan(v) for v in ind1])
                assert( not True in [math.isnan(v) for v in ind2])
            except AssertionError as e:
                print("After mutation")
                print(ind1)
                print(ind2)
                raise e
            del ind1.fitness.values, ind2.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        
        # Evaluate invalid individuuals only once
        occurred = {}
        unique_invalid_ind = []
        duplicates = []
        for ind in invalid_ind:
            str_ind = str(ind)
            if str_ind not in occurred:
                unique_invalid_ind.append(ind)
                occurred[str_ind] = True
            else:
                duplicates.append(ind)

        # assert( len(all_solutions) == len(criticalDict))
        assert np.array([str(all_solutions[i]) == list(criticalDict.keys())[i]  for i in range(len(all_solutions))]).all()

        if EVALUATE_IN_BATCH:
            fitnesses = evaluateFcnBatch(unique_invalid_ind)
        else:
            fitnesses = toolbox.map(toolbox.evaluate, unique_invalid_ind)

        for ind, fit in zip(unique_invalid_ind, fitnesses):
            ind.fitness.values = fit
            # set fitness of all duplicates that have not been evaluated
            for dup in duplicates:
                if str(dup) == str(ind):
                    dup.fitness.values = fit

        # add to have all evaluations
        # update_list_unique(all_solutions,offspring)
        # update_list_unique(all_solutions,offspring)
        
        assert( len(all_solutions) == len(criticalDict))
        assert np.array([str(all_solutions[i]) == list(criticalDict.keys())[i]  for i in range(len(all_solutions))]).all()

        # Select the next generation population
        # MU <= |pop + offspring| <= 2MU
        #print(f"+++ Individuals with fitness after CX in population: {[(ind.fitness.values,ind) for ind in pop]}")
        #print(f"+++ Individuals with fitness after CX in offsprings: {[(ind.fitness.values,ind) for ind in offspring]}")
        #print(f"+++ Size of candidate set with offsprings: {len(pop)+ len(offspring)}")
        pop = toolbox.select(pop + offspring, MU)
        record = stats.compile(pop)
        logbook.record(gen=gen, evals=len(invalid_ind), **record)
        print(logbook.stream)

    print("Final population hypervolume is %f" % hypervolume(pop))

    nCritical = 0
    for i in range(len(all_solutions)):
        str_ind = str(all_solutions[i])
        if (criticalDict[str_ind] == True ) :
            nCritical = nCritical + 1
 
    print("# critical individuals during search: "+ str(nCritical))
    print("# not critical individuals during search: "+ str(len(all_solutions) - nCritical))

    end = time.time()
    execTime = end - start
    
    # Results

    # Do not ouput if it is a subroutine
    if mode=="standalone":   
        print("++ Writing results (from the best) ++") 
        subFolderName = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        path = str(os.getcwd()) + RESULTS_FOLDER + Path(xosc).stem + os.sep + subFolderName
        writer.write_results(simulationOutputAll,algorithmName,pop,xosc,featureNames,execTime,path,scenario=xosc,all_pops=pop)
    
    assert np.array([str(all_solutions[i]) == list(criticalDict.keys())[i]  for i in range(len(all_solutions))]).all()

    return pop, all_solutions, criticalDict, logbook, execTime

