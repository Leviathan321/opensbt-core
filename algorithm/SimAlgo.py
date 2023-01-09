from abc import ABC, abstractclassmethod, abstractmethod

from experiment.search_configuration import SearchConfiguration
from model_ga.problem import ProblemExtended


class SimAlgo(ABC):
    
    algorithm_name: str
    
    @abstractmethod
    def __init__(self, problem: ProblemExtended, config: SearchConfiguration):
        pass

    @abstractmethod
    def run(self):
        pass
