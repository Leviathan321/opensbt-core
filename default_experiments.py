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

############
''' Dummy Simulator with linear motion'''
def getExp2() -> Experiment:
    from simulation.dummy_simulation import DummySimulator

    problem = ADASProblem(
                          problem_name="DummySimulatorProblem",
                          scenario_path="",
                          xl=[0, 1, 0, 1],
                          xu=[360, 10,360, 5],
                          simulation_variables=[
                              "orientation_ego",
                              "velocity_ego",
                              "orientation_ped",
                              "velocity_ped"],
                          fitness_function=FitnessAdaptedDistanceSpeed(),
                          critical_function=CriticalAdasDistanceVelocity(),
                          simulate_function=DummySimulator.simulate,
                          simulation_time=5,
                          sampling_time=0.25)
    config = DefaultSearchConfiguration()
    config.population_size = 2
    config.n_generations = 5
    experiment = Experiment(
                            name="2",
                            problem=problem,
                            algorithm=AlgorithmType.NSGAII,
                            search_configuration=config)
    return experiment


experiments_store.register(getExp2())
