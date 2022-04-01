import array
from datetime import datetime
from logging import critical
import random
import json
import matplotlib.pyplot as plt
import numpy

from math import sqrt
from deap import algorithms
from deap import base
from deap import benchmarks
from deap.benchmarks.tools import diversity, convergence, hypervolume
from deap import creator
from deap import tools

from fitness_functions import fitness
from dynamics import basic_dynamics as bd
from simulation.dummy_simulation import DummySimulator as DS
from simulation.simulator import SimulationOutput
from visualization import plotter
from random import randrange

############
# Problem definition

# (x,y, orientation, velocity) for both actors -> 8 genoms
# if  lower and upper boundary are equal mutation throws error

var_min = [ 0, 0, 0,1, 100, 100, 0,5]
var_max = [ 0, 0, 20, 50, 100, 100,20,10]

# scenario parameters

simTime=60
samplingTime = 1
simulateFcn = DS.simulate
nFitnessFcts = 1
criticalDict = {}

# genetic algorithm parameters

nGenerations = 10 # 22
initialPopulationSize = 500 # should be % 4 == 0, when using tools.selTournamentDCD
crossoverProbability = 0.6
numberDimensions = 8
mutationRate = 1.0/numberDimensions

# fitness fcts and critical fct

def evaluateFcn(individual):
    fit, simout= fitness.fitness_basic_two_actors(individual,
                                            simTime=simTime,
                                            samplingTime=samplingTime,
                                            simulateFcn=simulateFcn)

    # dummy, add criticality information in evaluation function
    random.seed(datetime.now())
    notCritical = randrange(3)
    criticalDict[str(individual)] = isCritical([fit],simout)
    
    return fit,

def isCritical(Fit, simout: SimulationOutput):
    if((simout.otherParams['isCollision'] == True) or (Fit[0] < 1)):
        return True
    else:
        return False


#evaluateFcn = @(xvars) fitness.fitness_basic_two_actors(xvars,simTime=simTime,samplingTime=samplingTime,simulateFcn=simulateFcn)
##############

def nsgaII_testcase(var_min=var_min, var_max=var_max, evaluateFcn=evaluateFcn,nFitnessFcts=nFitnessFcts):   
    BOUND_LOW, BOUND_UP = var_min, var_max
    NDIM = numberDimensions

    creator.create("FitnessMin", base.Fitness, weights=(-1.0*nFitnessFcts,))
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
    toolbox.register("mutate", tools.mutUniformInt, low=BOUND_LOW, up=BOUND_UP, indpb=mutationRate)
    toolbox.register("select", tools.selNSGA2)

    seed=None
    random.seed(seed)

    NGEN = nGenerations
    MU = initialPopulationSize
    CXPB = crossoverProbability

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    # stats.register("avg", numpy.mean, axis=0)
    # stats.register("std", numpy.std, axis=0)
    stats.register("min", numpy.min, axis=0)
    stats.register("max", numpy.max, axis=0)

    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"

    pop = toolbox.population(n=MU)

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
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
        offspring = tools.selTournamentDCD(pop, len(pop))
        offspring = [toolbox.clone(ind) for ind in offspring]

        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            if random.random() <= CXPB:
                toolbox.mate(ind1, ind2)

            toolbox.mutate(ind1)
            toolbox.mutate(ind2)
            del ind1.fitness.values, ind2.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Select the next generation population
        pop = toolbox.select(pop + offspring, MU)
        record = stats.compile(pop)
        logbook.record(gen=gen, evals=len(invalid_ind), **record)
        print(logbook.stream)

    print("Final population hypervolume is %f" % hypervolume(pop))

    return pop, logbook

if __name__ == "__main__":
    # with open("pareto_front/zdt1_front.json") as optimal_front_data:
    #     optimal_front = json.load(optimal_front_data)
    # Use 500 of the 1000 points in the json file
    # optimal_front = sorted(optimal_front[i] for i in range(0, len(optimal_front), 2))

    pop, stats = nsgaII_testcase()
    pop.sort(key=lambda x: x.fitness.values)

    #print(stats)

    dictScenario = {
            "egoX": 0,
            "egoY": 0,
            "orientationEgo":0,
            "velocityEgo":0,
            "objX": 0,
            "objY": 0,
            "orientationObj":0,
            "velocityObj":0,
        }

    dictScenario["egoX"] = pop[0][0]
    dictScenario["egoY"] = pop[0][1]
    dictScenario["orientationEgo"] = pop[0][2]
    dictScenario["velocityEgo"] = pop[0][3]
    dictScenario["objX"] = pop[0][4]
    dictScenario["objY"] = pop[0][5]
    dictScenario["orientationObj"] = pop[0][6]
    dictScenario["velocityObj"] = pop[0][7]

    print("Most critical scenario instance:" + str(dictScenario))

    # print("Convergence: ", convergence(pop, optimal_front))
    # print("Diversity: ", diversity(pop, optimal_front[0], optimal_front[-1]))
    # front = numpy.array([ind.fitness.values for ind in pop])
    # print(front)
    # optimal_front = numpy.array(optimal_front)
    # plt.scatter(optimal_front[:,0], optimal_front[:,1], c="r")
    # plt.scatter(front[:,0], front[:,1], c="b")

    print("# individuals: "+ str(len(pop)))

    nCritical = 0
    for i in range(len(pop)):
        if ( criticalDict[str(pop[i])] == True ) :
            nCritical = nCritical + 1

    print("# critical individuals: "+ str(nCritical))
    print("# not critical individuals: "+ str(len(pop) - nCritical))

    plotter.plotScenario(pop[0],simTime=simTime,samplingTime=samplingTime)