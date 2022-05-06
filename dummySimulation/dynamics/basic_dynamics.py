from array import array
from math import sin,cos,pi,ceil
import numpy as np

def planMotion(startingPosition,orientation,velocity,simTime,samplingTime):
    theta = orientation 
    T = simTime
    v = velocity
    t = samplingTime
    
    S = v*T
    distX = sin(theta*pi/180)*S
    distY = cos(theta*pi/180)*S
    nSteps = ceil(T/t)
    
    asize = nSteps + 1
    arrayTime = np.linspace(0, T, asize)
    arrayX = np.linspace(startingPosition[0], distX, asize)
    arrayY = np.linspace(startingPosition[1], distY, asize)
    arrayV = v * np.ones(asize, dtype=np.int64)

    return np.concatenate((arrayTime,arrayX,arrayY,arrayV)).reshape(4,asize)

# motion = planMotion(np.array([1,1]),0,10,10,1)

# print(motion)