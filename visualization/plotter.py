from math import sqrt
from simulation.simulator import SimulationOutput
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle
import os
import numpy as np

'''
    Plot the fitness values of the calculated (optimal) solutions
'''


def plotSolutions(all_pops, scenario, num=10, savePath=None):
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

        plt.plot(fit1, fit2, 'ro')

    if savePath is not None:
        fig.savefig(savePath + os.sep + "pareto.pdf", format='pdf')
        plt.show(block=False)
        plt.close(fig)
    else:
        plt.show(block=False)

    return fig


def plotOutput(simout: SimulationOutput, featureNames, featureValues, fitness, savePath=None):
    if "car_length" in simout.otherParams:
        car_length = float(simout.otherParams["car_length"])
    else:
        car_length = float(3.9)

    if "car_width" in simout.otherParams:
        car_width = float(simout.otherParams["car_width"])
    else:
        car_width = float(1.8)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(1, 1, 1)

    ln1, = plt.plot([], [], 'r.')  # plot a trace of Ego
    ln2, = plt.plot([], [], 'b.')  # plot a trace of Ped

    plt.title("Simulation of scenario")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")

    trace_ego = np.array(simout.location["ego"])  # time series of Ego position
    trace_adv = np.array(simout.location["adversary"])  # time series of Ped position

    x_ego = trace_ego[:, 0]
    y_ego = trace_ego[:, 1]
    x_adv = trace_adv[:, 0]
    y_adv = trace_adv[:, 1]

    yaw_ego = np.array(simout.yaw["ego"])  # time series of Ego velocity

    '''Finding suitable axis limits for visualization'''
    borders_x = np.array([np.min(x_ego), np.max(x_ego), np.min(x_adv), np.max(x_adv)])
    borders_y = np.array([np.min(y_ego), np.max(y_ego), np.min(y_ego), np.max(y_ego)])

    left_border_x = min(borders_x)
    right_border_x = max(borders_x)
    center_x = (left_border_x + right_border_x) / 2

    left_border_y = min(borders_y)
    right_border_y = max(borders_y)
    center_y = (left_border_y + right_border_y) / 2

    maximum_size_of_axis = max((right_border_y - left_border_y), (right_border_x- left_border_x))

    # ax.axis('equal')
    ax.set(xlim=(center_x - maximum_size_of_axis * 0.55, center_x + maximum_size_of_axis * 0.55),
           ylim=(center_y - maximum_size_of_axis * 0.55, center_y + maximum_size_of_axis * 0.55))

    patch = Rectangle((0, 0), width=car_width, height=car_length,
                      color='yellow')
    patch.set_width(car_width)
    patch.set_height(car_length)

    def update(i):
        ln1.set_data(x_ego[i], y_ego[i])
        ln2.set_data(x_adv[i], y_adv[i])

        rotation_angle = (yaw_ego[i] - 90) % 360
        patch.set_angle(rotation_angle)
        shift_angle = rotation_angle - 180 / np.pi * np.arctan(car_width / car_length)
        shift_x = 0.5 * np.sqrt(car_width ** 2 + car_length ** 2) * np.sin(shift_angle * np.pi / 180)
        shift_y = 0.5 * np.sqrt(car_width ** 2 + car_length ** 2) * np.cos(shift_angle * np.pi / 180)

        patch.set_xy([x_ego[i] + shift_x, y_ego[i] - shift_y])
        ax.add_patch(patch)

        return

    ani = FuncAnimation(fig, update, frames=len(simout.times))

    writer = PillowWriter(fps=60)

    full_path = str(savePath) + "_trajectory.gif"
    ani.save(full_path, writer=writer)

    return


def plotDistance(simout: SimulationOutput, scenario, savePath=None):
    fig = plt.figure()
    fig.text(.5, .15, scenario, ha='center')

    plt.title("Distance Profile ego - other vehicle")
    plt.xlabel("t [s]")
    plt.ylabel("d [m]")

    ego = simout.location["ego"]
    other = simout.location["adversary"]

    x_ego = [v[0] for v in ego]
    y_ego = [v[1] for v in ego]
    x_other = [v[0] for v in other]
    y_other = [v[1] for v in other]

    distance = []
    for i in range(0, len(x_ego)):
        dif = abs(x_ego[i] - x_other[i]) ** 2 + abs(y_ego[i] - y_other[i]) ** 2
        distance.append(sqrt(dif))

    plt.plot(simout.times, distance)

    if savePath is not None:
        fig.savefig(savePath + "_distance.pdf", format='pdf')
        plt.show(block=False)
        plt.close(fig)
    else:
        plt.show(block=False)

    return fig


def plotMap(fig, x, y, width, height):
    ax = fig.add_subplot(111)
    ax.add_patch(Rectangle((x, y), width, height, color="black", fc='none',
                           ec='g',
                           lw=1))


def plotScenario(simulateFcn, candidates, simTime, samplingTime, xosc, featureNames, savePath=None):
    simouts = simulateFcn(candidates, xosc=xosc, featureNames=featureNames, simTime=simTime, samplingTime=samplingTime)
    for (candidate, simout) in zip(candidates, simouts):
        plotOutput(simout=simout,
                   featureValues=candidate,
                   featureNames=featureNames,
                   savePath=savePath)


def plotScenario(simulationOutput, candidate, xosc, featureNames, fitness, savePath):
    plotOutput(simout=simulationOutput,
               featureValues=candidate,
               featureNames=featureNames,
               fitness=fitness,
               savePath=savePath)


