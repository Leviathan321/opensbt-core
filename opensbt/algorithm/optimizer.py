from abc import ABC, abstractclassmethod, abstractmethod
from typing import Dict

from opensbt.experiment.search_configuration import SearchConfiguration
from opensbt.model_ga.problem import SimulationProblem
from opensbt.model_ga.result import SimulationResult
from pymoo.optimize import minimize
from pymoo.core.problem import Problem  
from pymoo.core.algorithm import Algorithm

class Optimizer(ABC):
    
    algorithm_name: str
    parameters: Dict
    config: SearchConfiguration
    problem: Problem
    algorithm: Algorithm
    termination: object
    save_history: bool
    
    @abstractmethod
    def __init__(self, problem: SimulationProblem, config: SearchConfiguration):
        ''' Create here the algorithm instance to be used in run '''
        pass

    def run(self) -> SimulationResult:
        return minimize(self.problem,
                self.algorithm,
                self.termination,
                save_history=self.save_history,
                verbose=True,
                seed = self.config.seed)