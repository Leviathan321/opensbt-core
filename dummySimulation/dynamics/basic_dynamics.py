from array import array
from math import sin, cos, pi, ceil
import numpy as np

def planMotion(startingPosition, orientation, velocity, simTime, samplingTime):
    theta = orientation
    T = simTime
    v = velocity
    t = samplingTime

    S = v * T
    distX = cos(theta * pi / 180) * S
    distY = sin(theta * pi / 180) * S
    nSteps = ceil(T / t)

    asize = nSteps + 1
    arrayTime = np.linspace(0, T, asize)
    arrayX = np.linspace(startingPosition[0], startingPosition[0] + distX, asize)
    arrayY = np.linspace(startingPosition[1], startingPosition[1] + distY, asize)
    arrayV = v * np.ones(asize, dtype=np.int64)
    # arrayV_x = arrayV * sin(theta * pi / 180)
    # arrayV_y = arrayV * cos(theta * pi / 180)
    arrayYaw = theta * np.ones(arrayTime.size)

    # print('theta = ', theta, 'v_x = ', arrayV_x, 'v_y = ', arrayV_y)
    return np.concatenate((arrayTime, arrayX, arrayY, arrayV, arrayYaw)).reshape(5, asize)
