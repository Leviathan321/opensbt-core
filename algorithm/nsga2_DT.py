
from logging import critical
from algorithm.nsga2_TC import nsga2_TC
import algorithm.regions as regions

## simulation parameters

simTime=10
samplingTime=60

## genetic algorithm parameters
nGenerations = 2 # 22
initialPopulationSize = 1 # should be % 4 == 0, when using tools.selTournamentDCD
crossoverProbability = 0.6
mutationRate = 0.2

SUPPRESS_PLT = True

def nsga2_DT(initialPopulationSize, 
            nGenerations,
                    var_min, 
                    var_max, 
                    fitnessFcn, 
                    nFitnessFcts, 
                    criticalFcn, 
                    simulateFcn,
                    featureNames,
                    xosc,
                    criticalDict,
                    initial_pop=[]):

    pop, criticalDict, stats = nsga2_TC(initialPopulationSize, 
                    nGenerations,
                    var_min, 
                    var_max, 
                    fitnessFcn, 
                    nFitnessFcts, 
                    criticalFcn, 
                    simulateFcn,
                    featureNames,
                    xosc,
                    initial_pop,
                    criticalDict=criticalDict)
    critValues = []

    for ind in pop:
        critValues.append(criticalDict[str(ind)])
        
    pop_regions_ind, newBounds = regions.getCriticalRegions(pop,critValues, var_min=var_min, var_max=var_max)
    
    all_pops = []
    all_pops.extend(pop)

    if len(newBounds) > 0:
        pop_regions = []

        # get individuals instead only indices
        for pop_region_ind in pop_regions_ind:
            temp = [pop[ind] for ind in pop_region_ind]
            pop_regions.append(temp)
        
        #print(pop_regions)
        print("#critical regions found:" + str(len(newBounds)))

        for i in range(0,len(newBounds)):
            bound = newBounds[i]
            print(bound[0])
            print(bound[1])
            pop_new = []
            # requirement by tournament selection len(pop) is % 4
            # if  not len(pop_regions[i]) % 4:
            #     n = findNextDivident(len(pop_regions[i]),4)
            #     pop_new = fillUpPopulation(pop_regions[i],pop,n)
            #     assert pop_new % 4 == True
            pop_run_bi, critical_bound, stats2 = nsga2_TC(
                    initialPopulationSize=initialPopulationSize,
                    nGenerations=nGenerations,
                    fitnessFcn=fitnessFcn, 
                    nFitnessFcts=nFitnessFcts, 
                    criticalFcn=criticalFcn, 
                    simulateFcn=simulateFcn,
                    featureNames=featureNames,
                    xosc=xosc,
                    initial_pop=pop_regions[i], 
                    var_min=bound[0], 
                    var_max=bound[1],
                    criticalDict=criticalDict)

            all_pops.extend(pop_run_bi)

            # TODO
            # the crititcalDict needs to be passed to TC
            criticalDict.update(critical_bound)
        
            all_pops.sort(key=lambda x: x.fitness.values)

    return all_pops,criticalDict



# def printResult(population,simTime,samplingTime):
#     pop = population
#     pop.sort(key=lambda x: x.fitness.values)

#     dictScenario = {}
#     i = 0
#     for featureName in featureNames:
#         dictScenario[featureName] = pop[0][i]
#         i = i + 1 
    
#     print("# individuals: "+ str(len(pop)))
#     print("# most critical:" + str(pop[0]))
#     print("# most critical fitness: " + str(pop[0].fitness.values) )
    
    # if not SUPPRESS_PLT:
    #     plotter.plotScenario(simulateFcn,pop[0],simTime,samplingTime)

# def findNextDivident(n,k):
#     i = 1
#     while math.pow(k,i) < n:
#         i = i + 1
#     return i

# pop: sorted population
# givenPop: sorted subset of pop
# num < len(pop) - len(givenPop)
# def fillUpPopulation(givenPop,all,num):
#     s  = set(all)- set(givenPop)
#     return givenPop + list(s)[:num]

