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
from experiment.search_configuration import SearchConfiguration
from visualization import output
import quality_indicators.metrics.spread as qi

from model_ga.result import *

ALGORITHM_NAME = "NSGA-II"
RESULTS_FOLDER = os.sep + "results" + os.sep
WRITE_ALL_INDIVIDUALS = True

class NSGAII_SIM(object):

    algorithm_name = ALGORITHM_NAME

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

    def write_results(self, results_folder = RESULTS_FOLDER):
        algorithm_name = self.algorithm_name
        if self.res is None:
            print("Result object is None. Execute algorithm first, before writing results.")
            return
        print(f"=====[{ALGORITHM_NAME}] Writing results...")
        config = self.config
        res = self.res
        algorithm_parameters = {
            "Population size" : str(config.population_size),
            "Number of generations" : str(config.n_generations),
            "Number of offsprings": str(config.num_offsprings),
            "Crossover probability" : str(config.prob_crossover),
            "Crossover eta" : str(config.eta_crossover),
            "Mutation probability" : str(config.prob_mutation),
            "Mutation eta" : str(config.eta_mutation)
        }

        save_folder = output.create_save_folder(res.problem, results_folder, algorithm_name)

        output.convergence_analysis(res, save_folder)
        output.hypervolume_analysis(res, save_folder)
        output.spread_analysis(res, save_folder)
        output.write_calculation_properties(res,save_folder,algorithm_name,algorithm_parameters)
        output.design_space(res, save_folder)
        output.objective_space(res, save_folder)
        output.optimal_individuals(res, save_folder)
        output.write_summary_results(res, save_folder)
        output.write_simulation_output(res,save_folder)
        output.simulations(res, save_folder)

        if WRITE_ALL_INDIVIDUALS:
            output.all_individuals(res, save_folder)
