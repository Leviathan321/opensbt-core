

import pymoo
from algorithm.optimizer import Optimizer
from simulation.simulator import SimulationOutput
from utils.sampling import CartesianSampling
import os
import sys
from pathlib import Path
from typing import List

from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from pymoo.core.problem import Problem
from pymoo.termination import get_termination
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.population import Population
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.sampling.lhs import LHS
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from algorithm.classification.classifier import ClassificationType
from algorithm.classification.decision_tree.decision_tree import *
from evaluation.critical import Critical
from experiment.search_configuration import DefaultSearchConfiguration, SearchConfiguration
from problem.pymoo_test_problem import PymooTestProblem
from simulation.simulator import SimulationOutput
import quality_indicators.metrics.spread as qi
import logging as log

from model_ga.result import *

ALGORITHM_NAME = "RS"
RESULTS_FOLDER = os.sep + "results" + os.sep + "single" +  os.sep
WRITE_ALL_INDIVIDUALS = True

''' We used the NSGAII algorithm, but with only one iteration.
    I.e. individuals are only evaluated which have been genereated using a sampling strategy
'''
class PureSampling(Optimizer):
    
    algorithm_name = ALGORITHM_NAME

    def __init__(self,
                problem: Problem,
                config: SearchConfiguration):

        self.config = config
        self.problem = problem
        self.res = None
        self.algorithm = NSGA2(
            pop_size=config.population_size,                 
            sampling=CartesianSampling(),        # specify the rs sampling method here
            eliminate_duplicates=True)
        self.parameters = {
           "number_of_samples" : pow(config.population_size, problem.n_var)
        }
        # only one iteration means we only evaluate on time pop size solutions
        self.termination = get_termination("n_gen", 1)
        self.save_history = True
        
        log.info(f"Initialized algorithm with config: {config.__dict__}")
