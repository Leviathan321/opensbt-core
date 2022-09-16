from simulation.simulator import SimulationOutput
from typing import Tuple

''' Define the critical functions here
    
    Critical functions are used for clustering the searchspace after an NSGA2 run.

    Example critical function :

    Scenario critical <-> 

    a) collision ocurred
    b) minimal distance between ego and other vehicle < 0.5m
    c) mimimal ttc is < 1s 
'''
def criticalFcn(fit: Tuple ,simout: SimulationOutput):

    # include min_ttc if available

    TTC_THRESH = 1.0
    if "min_ttc"  in simout.otherParams:
        min_ttc = simout.otherParams["min_ttc"]
    else:
        min_ttc = TTC_THRESH + 0.1

    isCollision = simout.otherParams['isCollision']
    if( isCollision) or (fit[0] < 0.5) or min_ttc < TTC_THRESH:
        return True
    else:
        return False

