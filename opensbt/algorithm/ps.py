

import pymoo
import random
import time
import logging as log

from opensbt.model_ga.individual import IndividualSimulated
pymoo.core.individual.Individual = IndividualSimulated

from opensbt.model_ga.population import PopulationExtended
pymoo.core.population.Population = PopulationExtended

from opensbt.model_ga.result  import SimulationResult
pymoo.core.result.Result = SimulationResult

from opensbt.model_ga.problem import SimulationProblem
pymoo.core.problem.Problem = SimulationProblem

from pymoo.core.algorithm import Algorithm
from opensbt.algorithm.optimizer import Optimizer

from pymoo.core.problem import Problem
from pymoo.operators.sampling.rnd import FloatRandomSampling

from opensbt.experiment.search_configuration import SearchConfiguration
from opensbt.utils.evaluation import evaluate_individuals

from opensbt.utils.sorting import get_nondominated_population

class PureSampling(Optimizer):
    """
    This class provides the parent class for all sampling based search algorithms.
    """
    
    algorithm_name = "RS"

    def __init__(self,
                problem: Problem,
                config: SearchConfiguration,
                sampling_type = FloatRandomSampling):
        """Initializes pure sampling approaches.
        
        :param problem: The testing problem to be solved.
        :type problem: Problem
        :param config: The configuration for the search.
        :type config: SearchConfiguration
        :param sampling_type: Sets by default sampling type to RS.
        :type sampling_type: _type_, optional
        """
        self.config = config
        self.problem = problem
        self.res = None
        self.sampling_type = sampling_type
        self.sample_size = config.population_size
        self.n_splits = 10 # divide the population by this size 
                         # to make the algorithm iterative for further analysis
        self.parameters = { 
                            "sample_size" : self.sample_size
        }
        log.info(f"Initialized algorithm with config: {config.__dict__}")

    def run(self) -> SimulationResult:
        """Overrides the run method of Optimizer by providing custom evaluation of samples and division in "buckets" for further analysis with pymoo.
           (s. n_splits variable)
        :return: Return a SimulationResults object which holds all information from the simulation.
        :rtype: SimulationResult
        """
        config = self.config
        random.seed(config.seed)
        
        problem = self.problem
        sample_size = self.sample_size
        sampled = self.sampling_type()(problem,sample_size)
        n_splits = self.n_splits
        start_time = time.time()

        pop = evaluate_individuals(sampled, problem)

        execution_time = time.time() - start_time

        # create result object
        self.res = self._create_result(problem, pop, execution_time, n_splits)
        
        return self.res 
    
    def _create_result(self, problem, pop, execution_time, n_splits):
        res_holder = SimulationResult()
        res_holder.algorithm = Algorithm()
        res_holder.algorithm.pop = pop
        res_holder.algorithm.evaluator.n_eval = len(pop)
        res_holder.problem = problem
        res_holder.algorithm.problem = problem
        res_holder.exec_time = execution_time
        res_holder.opt = get_nondominated_population(pop)
        res_holder.algorithm.opt = res_holder.opt

        res_holder.history = []  # history is the same instance 
        n_bucket = len(pop) // n_splits
    
        pop_sofar = 0
        for i in range(0,n_splits):
            
            algo = Algorithm()
            algo.pop = pop[(i*n_bucket):min((i+1)*n_bucket,len(pop))]
            pop_sofar += len(algo.pop)
            algo.evaluator.n_eval = pop_sofar
            algo.opt = get_nondominated_population(algo.pop)
            res_holder.history.append(algo)
        
        return res_holder