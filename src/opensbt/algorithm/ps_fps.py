import pymoo

from algorithm.ps import PureSampling
from experiment.search_configuration import SearchConfiguration
from model_ga.problem import SimulationProblem
from model_ga.result import SimulationResult
from utils.fps import FPS
from utils.sorting import get_nondominated_population
pymoo.core.problem.Problem = SimulationProblem
from pymoo.core.problem import Problem


class PureSamplingFPS(PureSampling):
    
    algorithm_name = "FPS"

    def __init__(self,
                    problem: Problem,
                    config: SearchConfiguration,
                    sampling_type = FPS):
        super().__init__(
            problem = problem,
            config = config,
            sampling_type = sampling_type)
        
        self.algorithm_name = "FPS"
        self.parameters["algorithm_name"] = self.algorithm_name
