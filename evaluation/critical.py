from simulation.simulator import SimulationOutput
from typing import Tuple
''' Define the critical function here '''

def criticalFcn(fit: Tuple ,simout: SimulationOutput):
    if((simout.otherParams['isCollision'] == True) or (fit[0] > 2 and fit[0] < 7  or  fit[0] < 0.9)):
        return True
    else:
        return False
