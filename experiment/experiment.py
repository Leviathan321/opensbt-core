from model_ga.problem import *
from simulation.simulator import *
from algorithm.algorithm import *
from experiment.search_configuration import *

class Experiment(object):
    # TODO refactor nsag2/nsga2-DT into a class to use algorithm instance in algorithm parameter
    def __init__(self, name: str, problem: Problem, algorithm: AlgorithmType, search_configuration: SearchConfiguration):
        self.name = name
        self.problem = problem
        self.algorithm = algorithm
        self.search_configuration = search_configuration