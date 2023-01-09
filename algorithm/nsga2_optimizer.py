import os
import sys
from pathlib import Path

from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from pymoo.core.problem import Problem
from pymoo.termination import get_termination
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.population import Population
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from algorithm.classification.classifier import ClassificationType
from algorithm.classification.decision_tree.decision_tree import *
from algorithm.optimizer import Optimizer
from experiment.search_configuration import SearchConfiguration
from visualization import output
import quality_indicators.metrics.spread as qi
from model_ga.result import *

class NsgaIIOptimizer(Optimizer):

    algorithm_name = "NSGA-II"

    def __init__(self,
                problem: Problem,
                config: SearchConfiguration):

        self.config = config
        self.problem = problem
        self.res = None

        if self.config.prob_mutation is None:
            self.config.prob_mutation = 1 / problem.n_var

        self.algorithm = NSGA2(
            pop_size=config.population_size,
            n_offsprings=config.num_offsprings,
            sampling=FloatRandomSampling(),
            crossover=SBX(prob=config.prob_crossover, eta=config.eta_crossover),
            mutation=PM(prob=config.prob_mutation, eta=config.eta_mutation),
            eliminate_duplicates=True)

        ''' Prioritize max search time over set maximal number of generations'''
        if config.maximal_execution_time is not None:
            self.termination = get_termination("time", config.maximal_execution_time)
        else:
            self.termination = get_termination("n_gen", config.n_generations)

        self.save_history = True

    def run(self) -> ResultExtended:
        self.res = minimize(self.problem,
                    self.algorithm,
                    self.termination,
                    save_history=self.save_history,
                    verbose=True)

        return self.res
