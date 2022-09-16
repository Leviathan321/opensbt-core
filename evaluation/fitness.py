import sys
from simulation.simulator import SimulationOutput 

import numpy as np
import math
from simulation import dummy_simulation as ds
from scipy.spatial.distance import cdist
from utils import geometric
import random
import logging

''' Consider: 
        1. When defining functions make sure required attributes are in the SimulationOutput format 
        2. A fitness function maps a SimulationOutput instance to a float (several return values not supported)
'''

'''
   For Carla Simulator
'''

def fitness_min_distance_two_actors_carla(simout: SimulationOutput):
    if  "distance" in simout.otherParams:
        dist = simout.otherParams["distance"]
        result = min(dist)
    else:
        traceEgo = simout.location["ego"]
        tracePed = simout.location["adversary"]
        result = np.min(geometric.distPair(traceEgo,tracePed))
    return result


'''
   For Prescan Simulator
'''

def fitness_min_distance_two_actors_prescan(simout: SimulationOutput):
    if  "distance" in simout.otherParams:
        dist = simout.otherParams["distance"]
        result = min(dist)
    else:
        traceEgo = simout.location["ego"]
        tracePed = simout.location["other"]
        result = np.min(geometric.distPair(traceEgo,tracePed))
    return result

def fitness_min_ttc(simout: SimulationOutput):
    return simout.otherParams["min_ttc"]

def fitness_vimpact(simout: SimulationOutput):
    return simout.otherParams["ego_vimpact"]

def fitness_random(simout):
    return random.random()

'''
   For Dummy Simulator
'''

def fitness_basic_two_actors(simout: SimulationOutput):
    traceEgo = simout.location["ego"]
    tracePed = simout.location["adversary"]

    if simout.otherParams['isCollision']:
        # we dont want scenarios where collisions occure since we have no real driving model
        return sys.float_info.max
    else:
        value = np.min(geometric.distPair(traceEgo,tracePed))
    return value

''' For Dummy for testing '''

def fitness_random(simout: SimulationOutput):
    return 10*random.random()


''' Fitness function to resolve front and rear collisions '''

''' TESTED WITH DUMMYSIMULATOR ONLY'''


def fitness_parallel(z_parallel, car_length):
    steep_back = 10
    steep_front = 2
    # z = yPed - yEgo - car_length/2
    n = len(z_parallel)
    result = np.zeros(n)
    for i in range(n):
        if z_parallel[i] < -car_length:
            result[i] = 0
        elif z_parallel[i] < 0:
            result[i] = (np.exp(steep_back * (z_parallel[i] + car_length)) - np.exp(0)) / np.exp(
                steep_back * car_length)
        else:
            result[i] = np.exp(-steep_front * z_parallel[i]) / np.exp(0)
    return result


def fitness_perpendicular(z_perpendicular, car_width):
    sigma = 0.05
    # z = xPed - xEgo
    n = len(z_perpendicular)
    result = np.zeros(n)
    for i in range(n):
        if abs(z_perpendicular[i]) < car_width / 2:
            result[i] = 1
        else:
            result[i] = np.exp(-(0.5 / sigma) * (abs(z_perpendicular[i]) - car_width / 2) ** 2)
    return result


def fitness_severity(relative_velocity_sqr):
    result = relative_velocity_sqr

    return result


def fitness_combined(simout: SimulationOutput):
    if "car_length" in simout.otherParams:
        car_length = float(simout.otherParams["car_length"])
    else:
        car_length = float(3.9)

    if "car_width" in simout.otherParams:
        car_width = float(simout.otherParams["car_width"])
    else:
        car_width = float(1.8)

    trace_ego = np.array(simout.location["ego"])  # time series of Ego position
    trace_adv = np.array(simout.location["adversary"])  # time series of Ped position

    x_ego = trace_ego[:, 0]
    y_ego = trace_ego[:, 1]
    x_adv = trace_adv[:, 0]
    y_adv = trace_adv[:, 1]

    speed_ego = np.array(simout.velocity["ego"])  # time series of Ego velocity
    speed_adv = np.array(simout.velocity["adversary"])  # time series of Ped velocity

    z_parallel = y_adv - y_ego - car_length / 2
    z_perpendicular = x_adv - x_ego

    f_1 = fitness_parallel(z_parallel, car_length)
    f_2 = fitness_perpendicular(z_perpendicular, car_width)
    f_3 = fitness_severity(speed_ego)  # f_3 is not normalised to one, as it shows severity

    # print('shape f_1 = ', f_1.shape)
    # print('shape f_1 * f_2 = ', (f_1 * f_2).shape)

    result = np.max(f_1 * f_2 * f_3)
    return float(result)


''' Fitness function for a general case '''


def fitness_linear_2D(simout: SimulationOutput):
    if "car_length" in simout.otherParams:
        car_length = float(simout.otherParams["car_length"])
    else:
        car_length = float(3.9)

    if "car_width" in simout.otherParams:
        car_width = float(simout.otherParams["car_width"])
    else:
        car_width = float(1.8)

    trace_ego = np.array(simout.location["ego"])  # time series of Ego position
    trace_adv = np.array(simout.location["adversary"])  # time series of Ped position

    speed_ego = np.array(simout.velocity["ego"])  # time series of Ego velocity
    speed_adv = np.array(simout.velocity["adversary"])  # time series of Ped velocity

    yaw_ego = np.array(simout.yaw["ego"])  # time series of Ego velocity
    yaw_adv = np.array(simout.yaw["adversary"])  # time series of Ped velocity

    ''' Global coordinates '''
    x_ego = trace_ego[:, 0]
    y_ego = trace_ego[:, 1]
    x_adv = trace_adv[:, 0]
    y_adv = trace_adv[:, 1]


    ''' Coordinates, with respect to ego: e2 is parallel to the direction of ego '''
    e2_x = np.cos(yaw_ego * math.pi / 180)
    e2_y = np.sin(yaw_ego * math.pi / 180)
    e1_x = e2_y
    e1_y = -e2_x

    z_parallel = (x_adv - x_ego) * e2_x + (y_adv - y_ego) * e2_y - car_length / 2
    z_perpendicular = (x_adv - x_ego) * e1_x + (y_adv - y_ego) * e1_y

    f_1 = fitness_parallel(z_parallel, car_length)
    f_2 = fitness_perpendicular(z_perpendicular, car_width)
    f_3 = speed_ego # f_3 is not normalised to one, as it shows severity

    [w1, w2, w3] = [1, 1, 0.01]
    # result = np.max(f_3)
    result = np.max(f_1 * f_2)
    return float(result)
