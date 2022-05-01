from simulation.simulator import SimulationOutput
from matplotlib import pyplot as plt
import numpy 
from matplotlib.patches import Rectangle
from simulation.dummy_simulation import DummySimulator

def plotSim(simout: SimulationOutput,featureValues, features=None, map=None):
    fig = plt.figure()
    if features is None:
        features = [
                "egoX",
                "egoY",
                "orientationEgo",
                "velocityEgo",
                "objX",
                "objY",
                "orientationObj",
                "velocityObj"]
    scenario = ""
    for i in range(0,len(featureValues)):
        scenario = scenario + features[i] + "=" + str( "{:.2f}".format(featureValues[i]))
        if i < len(featureValues) - 1:
            scenario = scenario + "\n"
    fig.text(.5, .15, scenario, ha='center')
    plt.title("Simulation of scenario")
    plt.xlabel("x [m]") 
    plt.ylabel("y [m]")
    ego = simout.egoTrajectory
    obj = simout.objectTrajectory
    sizeE = numpy.size(ego,1)
    sizeP = numpy.size(obj,1)
    
    plt.plot(ego[1,1],ego[2,1],"sb")
    plt.plot(ego[1,1:sizeE-1],ego[2,1:sizeE-1], "ob")
    plt.text(ego[1,0],ego[2,0],"egos' trajectory")

    plt.plot(obj[1,1],obj[2,1],"sr")
    plt.plot(obj[1,1:sizeP-1],obj[2,1:sizeP-1], "or")
    plt.text(obj[1,0],obj[2,0],"objs' trajectory")

    plt.axis('equal')

    # if map is not None:
    #     plotMap(fig,map[0],map[1],map[2],map[3])

    plt.show(block=False)
    plt.show()

    return fig

def plotMap(fig, x,y,width,height):
    ax = fig.add_subplot(111)
    ax.add_patch(Rectangle((x, y), width, height,color="black", fc ='none', 
                        ec ='g',
                        lw = 1))

def plotScenario(simulateFcn,vars,simTime,samplingTime,featureNames=None):
    simout = simulateFcn(vars, simTime=simTime, samplingTime=samplingTime)
    plotSim(simout,vars,featureNames)
    
