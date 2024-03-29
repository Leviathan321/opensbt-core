from model_ga.problem import *
from simulation.simulator import *
from algorithm.algorithm import *
from experiment.search_configuration import *

@dataclass
class Experiment(object):
    def __init__(self, name: str, problem: Problem, algorithm: str, search_configuration: SearchConfiguration):
        self.name = name
        self.problem = problem
        self.algorithm = algorithm
        self.search_configuration = search_configuration

