from math import sqrt
from simulation.simulator import SimulationOutput
from matplotlib import pyplot as plt
from matplotlib.transforms import Affine2D
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
import os
import numpy as np
from visualization.configuration import *

def plot_gif(parameter_values, simout: SimulationOutput, savePath=None, fileName=None):
    if "car_length" in simout.otherParams:
        car_length = float(simout.otherParams["car_length"])
    else:
        car_length = float(3.9)

    if "car_width" in simout.otherParams:
        car_width = float(simout.otherParams["car_width"])
    else:
        car_width = float(1.8)

    if "adversary" in simout.location:
        name_adversary = "adversary"
    else:
        name_adversary = "other"
    
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)

    ln1, = plt.plot([], [], 'r.')  # plot a trace of Ego
    ln2, = plt.plot([], [], 'b.')  # plot a trace of Ped

    label_parameters = str(["{:.3f}".format(v) for v in parameter_values])
    plt.title(f"Simulation of scenario {label_parameters}")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")

    trace_ego = np.array(simout.location["ego"])  # time series of Ego position
    trace_adv = np.array(simout.location[name_adversary])  # time series of Ped position

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

    ax.set(xlim=(center_x - maximum_size_of_axis * 0.55, center_x + maximum_size_of_axis * 0.55),
           ylim=(center_y - maximum_size_of_axis * 0.55, center_y + maximum_size_of_axis * 0.55))

    patch = Rectangle((0, 0), width=car_width, height=car_length,
                      color='yellow')
    patch.set_width(car_width)
    patch.set_height(car_length)
    
    skip = 4
    def update(i):
        ln1.set_data(x_ego[0::skip][i], y_ego[0::skip][i])
        ln2.set_data(x_adv[0::skip][i], y_adv[0::skip][i])

        rotation_angle = (yaw_ego[0::skip][i] - 90) % 360
        patch.set_angle(rotation_angle)
        shift_angle = rotation_angle - 180 / np.pi * np.arctan(car_width / car_length)
        shift_x = 0.5 * np.sqrt(car_width ** 2 + car_length ** 2) * np.sin(shift_angle * np.pi / 180)
        shift_y = 0.5 * np.sqrt(car_width ** 2 + car_length ** 2) * np.cos(shift_angle * np.pi / 180)

        patch.set_xy([x_ego[0::skip][i] + shift_x, y_ego[0::skip][i] - shift_y])
        ax.add_patch(patch)

        return
    if "sampling_rate" in simout.otherParams:
        recorded_fps = simout.otherParams["sampling_rate"]
    else:
        recorded_fps = 100
    #print(f"number of time steps: {len(simout.times)}")
    ani = FuncAnimation(fig, update, frames=len(simout.times[0::skip]))
    writer = PillowWriter(fps=recorded_fps)
    ani.save(str(savePath) + str(fileName) + ".gif", writer=writer)

    return
