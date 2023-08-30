import os
from evaluation.fitness import *
from problem.adas_problem import ADASProblem
from problem.pymoo_test_problem import PymooTestProblem
from experiment.experiment_store import *
from algorithm.algorithm import *
from evaluation.critical import *

'''
    BNH Problem

    Pareto solutions:
    x∗1=x∗2∈[0,3]  and x∗1∈[3,5], x∗2=3
'''

def getExp1() -> Experiment:
    problem = PymooTestProblem(
        'BNH',
        critical_function=CriticalBnhDivided())

    config = DefaultSearchConfiguration()
    config.population_size = 10
    config.n_generations = 10
    config.maximal_execution_time = "00:00:02"
    experiment = Experiment(name="1",
                            problem=problem,
                            algorithm=AlgorithmType.NSGAII,
                            search_configuration=config)

    return experiment

experiments_store.register(getExp1())
