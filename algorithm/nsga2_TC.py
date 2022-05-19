#from simulation import prescan_simulation
from datetime import datetime
from pickle import TRUE

import random
import matplotlib.pyplot as plt
import numpy as np

from deap import algorithms
from deap import base
from deap.benchmarks.tools import diversity, convergence, hypervolume
from deap import creator
from deap import tools

from fitness_functions import fitness
from dummySimulation.dynamics import basic_dynamics as bd

from simulation.dummy_simulation import DummySimulator 
#from simulation.carla_simulation import CarlaSimulator
#from simulation.prescan_simulation import PrescanScenario, PrescanSimulator
from simulation.simulator import SimulationOutput

from visualization import plotter
from random import randrange

## simulation parameters
simTime=10
samplingTime=60

## genetic algorithm parameters
nGenerations = 2 # 22
initialPopulationSize = 1 # should be % 4 == 0, when using tools.selTournamentDCD
crossoverProbability = 0.6
mutationRate = 0.1

SUPPRESS_PLT = True

EVALUATE_IN_BATCH = True

def nsga2_TC(initialPopulationSize,
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
                    samplingTime=samplingTime):   
                    
    assert np.less_equal(var_min,var_max).all()

    criticalDict = {}

    ## HACK Use an extra function to execute scenarios in batch; in future use original evaluation function from deap 
    ## with an optimized threaded map processing function
    ## fitness fcts and critical fct

    def evaluateFcnBatch(individuals):
        simouts = simulateFcn(individuals,featureNames, xosc, simTime=simTime,samplingTime=samplingTime)
        fits = []
        for individual,simout in zip(individuals,simouts):
            fit = fitnessFcn(simout)
            #fit = fitness.fitness_min_distance_two_actors(simout)

            # dummy, add criticality information in evaluation function
            time = int(round(datetime.now().timestamp()))
            random.seed(time)

            if str(individual) not in criticalDict.keys():
                criticalDict[str(individual)] = isCritical([fit],simout)

            value = fit,
            fits.append(value)        
        
        return fits

    def evaluateFcn(individual):
        simout = simulateFcn(individual,featureNames, xosc, simTime=simTime,samplingTime=samplingTime)
        fit = fitnessFcn(simout)

        # dummy, add criticality information in evaluation function
        time = int(round(datetime.now().timestamp()))
        random.seed(time)

        if str(individual) not in criticalDict.keys():
            criticalDict[str(individual)] = isCritical([fit],simout)
        
        return fit,

    def isCritical(fit, simout: SimulationOutput):
        return criticalFcn(fit=fit,simout=simout)

    BOUND_LOW, BOUND_UP = var_min, var_max
    NDIM = len(var_min)

    print("NSGAII - STARTED")
    print("logical scenario: " + str(xosc))
    print("population size: " + str(len(initial_pop)))
    print("lower bound: " + str(BOUND_LOW))
    print("upper bound: " + str(BOUND_UP))

    if( not hasattr(creator,"FitnessMin")):
            creator.create("FitnessMin", base.Fitness, weights=(-1.0*nFitnessFcts,))
    if( not hasattr(creator,"Individual")):
        creator.create("Individual", list, typecode='d', fitness=creator.FitnessMin)

    toolbox = base.Toolbox()

    def uniform(low, up, size=None):
        try:
            return [random.uniform(a, b) for a, b in zip(low, up)]
        except TypeError:
            return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]

    toolbox.register("attr_float", uniform, BOUND_LOW, BOUND_UP, NDIM)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evaluateFcn)
    toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0)
    
    toolbox.register("mutate", tools.mutPolynomialBounded, eta=80,low=BOUND_LOW, up=BOUND_UP, indpb=mutationRate)
    toolbox.register("select", tools.selNSGA2)
    # selected individuals using selNSGA2 occure only once in the result */

    seed=None
    random.seed(seed)

    NGEN = nGenerations

    if len(initial_pop)==0:
        MU = initialPopulationSize
    else:
        MU = len(initial_pop)
        
    CXPB = crossoverProbability

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    # stats.register("avg", numpy.mean, axis=0)
    # stats.register("std", numpy.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"

    if initial_pop == []:
        print("population initialized")
        pop = toolbox.population(n=MU)
    else:
        pop = initial_pop
    
    # TODO consider case when population size is very low

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    
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

    # Begin the generational process
    for gen in range(1, NGEN):
        # Vary the population
        # offspring = tools.selNSGA2(pop,int(len(pop)/2))
        # "Each individual from the input list won’t be selected more than twice using tournament selection."
        offspring = tools.selTournamentDCD(pop, len(pop)-2)
        offspring = [toolbox.clone(ind) for ind in offspring]

        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            if random.random() <= CXPB:
                toolbox.mate(ind1, ind2)

            toolbox.mutate(ind1)
            toolbox.mutate(ind2)
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
    for i in range(len(pop)):
        if ( criticalDict[str(pop[i])] == True ) :
            nCritical = nCritical + 1

    print("# critical individuals: "+ str(nCritical))
    print("# not critical individuals: "+ str(len(pop) - nCritical))

    
    if not SUPPRESS_PLT:
        plotter.plotScenario(simulateFcn,pop[0],simTime=simTime,samplingTime=samplingTime)

    return pop, criticalDict, logbook