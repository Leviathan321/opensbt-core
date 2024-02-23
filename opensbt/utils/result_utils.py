import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import opensbt.algorithm.classification.decision_tree.decision_tree as decision_tree
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerPatch
from opensbt.visualization import scenario_plotter, combined
from pymoo.indicators.igd import IGD
from pymoo.indicators.hv import Hypervolume
from pymoo.core.population import Population
from opensbt.visualization.configuration import *
from opensbt.utils.sorting import *
from opensbt.algorithm.classification.classifier import ClassificationType
from opensbt.analysis.quality_indicators.quality import Quality
from opensbt.model_ga.problem import *
from opensbt.model_ga.result import *
from typing import Dict
from opensbt.utils.duplicates import duplicate_free
import logging as log
import uuid
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
import copy
from opensbt.config import BACKUP_FOLDER, CONSIDER_HIGH_VAL_OS_PLOT,  \
                            PENALTY_MAX, PENALTY_MIN, WRITE_ALL_INDIVIDUALS, \
                                METRIC_PLOTS_FOLDER, LAST_ITERATION_ONLY_DEFAULT, COVERAGE_METRIC_NAME


def create_result(problem, hist_holder, inner_algorithm, execution_time):
        # TODO calculate res.opt
        I = 0
        for algo in hist_holder:
            I += len(algo.pop)
            algo.evaluator.n_eval = I
            algo.start_time = 0
            algo.problem = problem
            algo.result()

        res_holder = SimulationResult()
        res_holder.algorithm = inner_algorithm
        res_holder.algorithm.evaluator.n_eval = I
        res_holder.problem = problem
        res_holder.algorithm.problem = problem
        res_holder.history = hist_holder
        res_holder.exec_time = execution_time

        # calculate total optimal population using individuals from all iterations
        opt_all = Population()
        for algo in hist_holder:
            opt_all = Population.merge(opt_all, algo.pop)
        # log.info(f"opt_all: {opt_all}")
        opt_all_nds = get_nondominated_population(opt_all)
        res_holder.opt = opt_all_nds

        return res_holder

def create_result_from_generations(path_generations, problem):

    n_generations = len(os.listdir(path_generations))

    # iterate over each generation file, cre
    inner_algorithm = NSGA2(
        pop_size=None,
        n_offsprings=None,
        sampling=None,
        crossover=SBX(),
        mutation=PM(),
        eliminate_duplicates=True)
    
    hist_holder = [copy.deepcopy(inner_algorithm) for i in range(n_generations)]

    for i in range(n_generations):
        path_gen = path_generations + f'gen_{i+1}.csv'
        pop_gen = combined.read_pf_single(filename=path_gen)
        hist_holder[i].pop = pop_gen
        opt_pop = Population(individuals=calc_nondominated_individuals(pop_gen))
        hist_holder[i].opt = opt_pop

    return create_result(problem=problem,
                        hist_holder=hist_holder,
                        inner_algorithm=inner_algorithm,
                        execution_time=0)