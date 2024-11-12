from abc import ABC, abstractclassmethod, abstractmethod
from typing import Dict

from opensbt.experiment.search_configuration import SearchConfiguration
from opensbt.model_ga.problem import SimulationProblem
from opensbt.model_ga.result import SimulationResult
from pymoo.optimize import minimize
from pymoo.core.problem import Problem  
from pymoo.core.algorithm import Algorithm

class Optimizer(ABC):
    """ Base class for all optimizers in OpenSBT.  Subclasses need to   
        implement the __init__ method. The run method has to be overriden when non pymoo implemented algorithms are used.
        For reference consider the implementation of the NSGA-II-DT optimizer in opensbt/algorithm/nsga2dt_optimizer.py
    """
    
    algorithm_name: str
    parameters: Dict
    config: SearchConfiguration
    problem: Problem
    algorithm: Algorithm
    termination: object
    save_history: bool
    
    @abstractmethod
    def __init__(self, problem: SimulationProblem, config: SearchConfiguration):
        """Initialize here the Optimization algorithm to be used for search-based testing.

        :param problem: The testing problem to be solved.
        :type problem: SimulationProblem
        :param config: The configuration for the search.
        :type config: SearchConfiguration
        """
        pass

    def run(self) -> SimulationResult:
        """Runs the optimizer for a given problem and search configuration.
           Returns the simulation output as an instance of SimulationResult.
           This methods need to overriden when a non pymoo-based algorithm is used (e.g., NSGA-II-DT)
        """
        return minimize(self.problem,
                self.algorithm,
                self.termination,
                save_history=self.save_history,
                verbose=True,
                seed = self.config.seed)