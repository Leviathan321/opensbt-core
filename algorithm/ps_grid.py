import pymoo

from model_ga.individual import IndividualSimulated
pymoo.core.individual.Individual = IndividualSimulated

from model_ga.population import PopulationExtended
pymoo.core.population.Population = PopulationExtended

from model_ga.result  import SimulationResult
pymoo.core.result.Result = SimulationResult

from model_ga.problem import SimulationProblem
pymoo.core.problem.Problem = SimulationProblem
from algorithm.ps import PureSampling
from experiment.search_configuration import SearchConfiguration
from model_ga.problem import SimulationProblem
from model_ga.result import SimulationResult
from utils.sorting import get_nondominated_population
pymoo.core.problem.Problem = SimulationProblem
from pymoo.core.problem import Problem
from utils.sampling import CartesianSampling

class PureSamplingGrid(PureSampling):

    def __init__(self,
                    problem: Problem,
                    config: SearchConfiguration,
                    sampling_type = CartesianSampling):
        super().__init__(
            problem = problem,
            config = config,
            sampling_type = sampling_type)    
        
            
        self.algorithm_name = "GS"

        self.parameters["algorithm_name"] = self.algorithm_name
