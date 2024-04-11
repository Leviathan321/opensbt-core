from opensbt.evaluation.critical import Critical
from opensbt.simulation.simulator import SimulationOutput

from examples.lanekeeping.config import CRITICAL_XTE, CRITICAL_AVG_XTE, MAX_ACC

##############

class MaxXTECriticality(Critical):
    def eval(self, vector_fitness, simout: SimulationOutput = None):
        # we fail the scenario, if max xte > 3 (< 3 because inverted fitness fnc)
        return abs(vector_fitness[0]) > CRITICAL_XTE

class AvgXTECriticality(Critical):
    def eval(self, vector_fitness, simout: SimulationOutput = None):
        # we fail the scenario, if max xte > 3 (< 3 because inverted fitness fnc)
        return abs(vector_fitness[0]) > CRITICAL_AVG_XTE

class MaxAccCriticality(Critical):
    def eval(self, vector_fitness, simout: SimulationOutput = None):
        # we fail the scenario, if max xte > 3 (< 3 because inverted fitness fnc)
        return abs(vector_fitness[1]) > MAX_ACC

class MaxXTECriticalitySteering_Simple(Critical):
    def eval(self, vector_fitness, simout: SimulationOutput = None):
        # we fail the scenario, if max xte > 3 (< 3 because inverted fitness fnc)
        return abs(vector_fitness[0]) > 2.5