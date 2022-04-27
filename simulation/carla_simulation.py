import numpy as np
from pathlib import Path
from utils.text_operations import substitute
from simulation.simulator import SimulationOutput
import subprocess
import shlex

class CarlaSimulator(object):
    # TODO consider sampling time; check how to set sampling time via matlab
    samplingTime = 1

    @staticmethod
    def simulate(vars, featureNames, filepath: str, simTime: float, samplingTime = samplingTime):
        instanceValues = {}
        for name in featureNames:
            instanceValues[name] = str(vars)

        scenarioInstancePath = substitute(filepath,instanceValues)
        subprocess.call(shlex.split('./simulate-carla.bash {0}'.format(scenarioInstancePath)))

        # TODO get real metrics calculation
        # TODO decide to do evaluation in optimizer oder directly by carla
        
        egoTrajectory = None
        objectTrajectory = None
        otherParams = None

        return SimulationOutput(simTime,egoTrajectory,objectTrajectory,otherParams=otherParams)