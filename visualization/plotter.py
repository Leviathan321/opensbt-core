from cmath import sqrt
from simulation.simulator import SimulationOutput
from matplotlib import pyplot as plt
import numpy 
from matplotlib.patches import Rectangle
from simulation.dummy_simulation import DummySimulator
import os

'''
    Plot the fitness values of the calculated (optimal) solutions
'''
def plotSolutions(all_pops,scenario, num = 10,savePath = None):
    if len(all_pops) < num:
        num = len(all_pops)
    fig = plt.figure()
    if len(all_pops[0].fitness.values) == 2:
        print("++ plotting solutions ++")
        fit1 = []
        fit2 = []
        # plot the fitness
        for i in range(num):
            fit1.append(all_pops[i].fitness.values[0])
            fit2.append(all_pops[i].fitness.values[1])

        fig.text(.5, .15, scenario, ha='center')

        plt.title(f"Solutions for scenario: {scenario}")
        plt.xlabel("fitness value 1") 
        plt.ylabel("fitness value 2")

        plt.plot(fit1,fit2,'ro')

    if savePath is not None:
        fig.savefig(savePath + os.sep +  "pareto.pdf", format='pdf')
        plt.show(block=False)
        plt.close(fig)
    else:
        plt.show(block=False)

    return fig

def plotOutput(simout: SimulationOutput, featureNames, featureValues, fitness, savePath = None):
    fig = plt.figure()
    scenario = "Example_" + str(fitness)
    for i in range(0,len(featureValues)):
        scenario = scenario + featureNames[i] + "=" + str( "{:.2f}".format(featureValues[i]))
        if i < len(featureValues) - 1:
            scenario = scenario + "\n"
    fig.text(.5, .15, scenario, ha='center')

    plt.title("Simulation of scenario")
    plt.xlabel("x [m]") 
    plt.ylabel("y [m]")

    # plot paths
    ego = simout.location["ego"]
    other =  simout.location["other"]

    sizeE = len(ego)
    sizeP = len(other)

    x_ego = [v[0] for v in ego]
    y_ego = [v[1] for v in ego]

    plt.plot(x_ego[sizeE-1],y_ego[sizeE-1],"sb")
    plt.plot(x_ego,y_ego, "ob")
    plt.text(x_ego[0],y_ego[0],"egos' trajectory")

    x_other = [v[0] for v in other]
    y_other = [v[1] for v in other]

    plt.plot(x_other[sizeP-1],y_other[sizeP-1],"sr")
    plt.plot(x_other,y_other, "or")
    plt.text(x_other[0],y_other[0],"others' trajectory")

    plt.axis('equal')

    # if map is not None:
    #     plotMap(fig,map[0],map[1],map[2],map[3])

    if savePath is not None:
        fig.savefig(savePath + "_trajectory.pdf", format='pdf')
        plt.show(block=False)
        plt.close(fig)
    else:
        plt.show(block=False)

    plotDistance(simout,scenario=scenario,savePath=savePath)

    return

def plotDistance(simout: SimulationOutput,scenario,savePath = None):
    fig = plt.figure()
    fig.text(.5, .15, scenario, ha='center')

    plt.title("Distance Profile ego - other vehicle")
    plt.xlabel("t [s]") 
    plt.ylabel("d [m]")

    ego = simout.location["ego"]
    other =  simout.location["other"]

    x_ego = [v[0] for v in ego]
    y_ego = [v[1] for v in ego]
    x_other = [v[0] for v in other]
    y_other = [v[1] for v in other]

    distance = []
    for i in range(0,len(x_ego)):
        dif = pow(x_ego[i] - x_other[i],2) + pow(y_ego[i] - y_other[i],2)
        distance.append(sqrt(dif))
    
    plt.plot(simout.times, distance)

    if savePath is not None:
        fig.savefig(savePath + "_distance.pdf", format='pdf')
        plt.show(block=False)
        plt.close(fig)
    else:
        plt.show(block=False)

    return fig


def plotMap(fig, x,y,width,height):
    ax = fig.add_subplot(111)
    ax.add_patch(Rectangle((x, y), width, height,color="black", fc ='none', 
                        ec ='g',
                        lw = 1))

def plotScenario(simulateFcn,candidates,simTime,samplingTime,xosc, featureNames,savePath=None):
    simouts = simulateFcn(candidates,xosc=xosc, featureNames=featureNames, simTime=simTime, samplingTime=samplingTime)
    for (candidate, simout) in zip(candidates,simouts):
        plotOutput(simout=simout,
                 featureValues=candidate,
                 featureNames=featureNames,
                 savePath=savePath)
    

def plotScenario(simulationOutput,candidate,xosc,featureNames,fitness,savePath):
    plotOutput(simout=simulationOutput,
                 featureValues=candidate,
                 featureNames=featureNames,
                 fitness=fitness,
                 savePath=savePath)
