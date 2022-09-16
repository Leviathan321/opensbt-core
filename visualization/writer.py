from visualization import plotter
import os
import csv

N_PLOT = 3  # select how many scenarios instances from the best to plot

def write_results(simulationOutputAll,algorithmName,fitnessFcnNames,pop,xosc,featureNames,execTime,path,scenario,all_pops,n_plot=N_PLOT):  
    if not os.path.isdir(path):
        os.makedirs(path)
    for i in range(0,n_plot):
        simout = simulationOutputAll[str(pop[i])]
        index = i + 1
        
        plotter.plotScenario(simout, featureNames=featureNames,xosc = xosc, candidate = pop[i],fitness=pop[i].fitness.values,savePath=path + os.sep  + str(index))
        plotter.plotSolutions(all_pops,scenario,num=100,savePath=path)
        #plotter.plotScenario(simulateFcn, featureNames=featureNames,xosc = xosc, candidates = [pop[i]], simTime=simTime,samplingTime=samplingTime,savePath=savePath)
    
    # write report of execution
    header = ['attribute', 'value'] 

    with open(path + os.sep + 'evaluation.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)

        writer.writerow(["algorithm" , algorithmName])
        writer.writerow(["fitness_functions_used" , fitnessFcnNames])
        writer.writerow(["n_individuals" , str(len(pop))])
        writer.writerow(["most_critical" , str([str(entry) for entry in zip(featureNames,pop[0])])])
        writer.writerow(["most_critical_fitness" , str(pop[0].fitness.values)] )
        writer.writerow(["total execution_time" , str(execTime)])

    # write down individuals and fitness
    header = ['id','individual']
    for i in range(len(pop[0].fitness.values)):
        header.append("f" + str(i))

    with open(path + os.sep + 'fitness.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        i = 1
        for ind in pop:
            fits = [val for val in ind.fitness.values]
            row = [i,str(ind)]
            row.extend(fits)
            writer.writerow(row)
            i = i + 1 

