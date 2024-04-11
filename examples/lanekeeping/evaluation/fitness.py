from typing import List, Tuple
from opensbt.evaluation.fitness import Fitness
from opensbt.simulation.simulator import SimulationOutput
import numpy as np


class MaxXTEFitness(Fitness):
    @property
    def min_or_max(self):
        return "max",

    @property
    def name(self):
        return "Max XTE (neg)",

    def eval(self, simout: SimulationOutput, **kwargs) -> Tuple[float]:
        traceXTE = [abs(x) for x in simout.otherParams["xte"]]
        return (max(traceXTE))

class MaxAvgXTEFitness(Fitness):
    @property
    def min_or_max(self):
        return "max", "max"

    @property
    def name(self):
        return "Max XTE (neg)", "Average XTE (neg)"

    def eval(self, simout: SimulationOutput, **kwargs) -> Tuple[float]:
        traceXTE = [abs(x) for x in simout.otherParams["xte"]]
        return (max(traceXTE), np.average(traceXTE))

class MaxAvgXTEVelocityFitness(Fitness):
    @property
    def min_or_max(self):
        return "max", "max"

    @property
    def name(self):
        return "Average XTE (neg)", "Velocity (neg)"

    def eval(self, simout: SimulationOutput, **kwargs) -> Tuple[float]:
        traceXTE = [abs(x) for x in simout.otherParams["xte"]]
        velocities = simout.speed["ego"]
        ind = np.argmax(traceXTE)
        return (np.average(traceXTE),velocities[ind])
    
class MaxXTESteeringChangeFitness(Fitness):
    @property
    def min_or_max(self):
        return "max", "max"

    @property
    def name(self):
        return "Max XTE (neg)", "Max Steering Change (neg)"

    def eval(self, simout: SimulationOutput, **kwargs) -> Tuple[float]:
        traceXTE = [abs(x) for x in simout.otherParams["xte"]]
        velocities = simout.speed["ego"]
        max_xte = np.max(traceXTE)
        steerings = [x for x in simout.otherParams["steerings"]]
        steering_derivative = np.max([abs(d) for d in calc_derivation(values=steerings)])
        return (max_xte, steering_derivative)
    

class MaxXTEVelocityFitness(Fitness):
    @property
    def min_or_max(self):
        return "max", "max"

    @property
    def name(self):
        return "Max XTE (neg)", "Velocity (neg)"

    def eval(self, simout: SimulationOutput, **kwargs) -> Tuple[float]:
        traceXTE = [abs(x) for x in simout.otherParams["xte"]]
        velocities = simout.speed["ego"]
        ind = np.argmax(traceXTE)
        return (np.max(traceXTE),velocities[ind])
        
class MaxXTEAccFitness(Fitness):
    @property
    def min_or_max(self):
        return "max", "max"

    @property
    def name(self):
        return "Max XTE (neg)", "Acceleration (neg)"

    def eval(self, simout: SimulationOutput, **kwargs) -> Tuple[float]:
        traceXTE = [abs(x) for x in simout.otherParams["xte"]]
        acc = simout.acceleration["ego"]
    
        return (np.max(traceXTE),np.max(acc))

class MaxXTECrossingsFitness(Fitness):
    @property
    def min_or_max(self):
        return "max", "max"

    @property
    def name(self):
        return "Max XTE (neg)", "Crossings (neg)"

    def eval(self, simout: SimulationOutput, **kwargs) -> Tuple[float]:
        traceXTE = [abs(x) for x in simout.otherParams["xte"]]
    
        return (np.max(traceXTE),
                calc_cross(simout.otherParams["xte"]))
    
#########################

def calc_cross(trace_xte):
    last_dir = 1
    crosses = 0
    for i in range(1,len(trace_xte)):
        current = mysign(trace_xte[i])
        if current != last_dir:
            crosses +=1
            last_dir = current
    return crosses

def mysign(num):
    if np.sign(num) == 0:
        return 1
    else:
        return np.sign(num)
    
def calc_derivation(values: List, fps = 20, scale: int = 1):
    res=[0]
    for i in range(1,len(values)):
        a = (values[i] - values[i-1]) * fps * scale
        res.append(a)
    return res
