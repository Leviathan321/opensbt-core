import os
from evaluation.fitness import *
from problem.adas_problem import ADASProblem
from problem.pymoo_test_problem import PymooTestProblem
from experiment.experiment_store import *
from algorithm.algorithm import *
from evaluation.critical import *
from simulation.carla_simulation import CarlaSimulator

'''
EXAMPLE CARLA SIMULATOR
ego speed is in km/h
'''

def getExp1() -> Experiment:
    problem = ADASProblem(
                          problem_name="PedestrianCrossingStartWalk",
                          scenario_path=os.getcwd() + "/scenarios/PedestrianCrossing.xosc",
                          xl=[0.5, 1, 0],
                          xu=[3, 22, 60],
                          simulation_variables=[
                              "PedSpeed",
                              "EgoSpeed",
                              "PedDist"],
                          fitness_function=FitnessMinDistanceVelocityFrontOnly(),  
                          critical_function=CriticalAdasFrontCollisions(),
                          simulate_function=CarlaSimulator.simulate,
                          simulation_time=10,
                          sampling_time=100,
                          approx_eval_time=10,
                          do_visualize = True
                          )
    experiment = Experiment(name="1",
                            problem=problem,
                            algorithm=AlgorithmType.NSGAII,
                            search_configuration=DefaultSearchConfiguration())
    return experiment

experiments_store.register(getExp1())

'''
    BNH Problem

    Pareto solutions:
    x∗1=x∗2∈[0,3]  and x∗1∈[3,5], x∗2=3
'''

def getExp2() -> Experiment:
    problem = PymooTestProblem(
        'BNH',
        critical_function=CriticalBnhDivided())

    config = DefaultSearchConfiguration()
    config.maximal_execution_time = "00:00:01"
    experiment = Experiment(name="2",
                            problem=problem,
                            algorithm=AlgorithmType.NSGAII,
                            search_configuration=config)

    return experiment

experiments_store.register(getExp2())

def getExp3() -> Experiment:
    problem = ADASProblem(
                          problem_name="TwoPedestriansCrossing",
                          scenario_path=os.getcwd() + "/scenarios/PedestrianCrossingSecond.xosc",
                          xl=[0.5, 1, 0],
                          xu=[3, 80, 60],
                          simulation_variables=[
                              "PedestrianSpeed",
                              "FinalHostSpeed",
                              "PedestrianEgoDistanceStartWalk"],
                          fitness_function=FitnessMinDistanceVelocityFrontOnly(),  
                          critical_function=CriticalAdasFrontCollisions(),
                          simulate_function=CarlaSimulator.simulate,
                          simulation_time=10,
                          sampling_time=100,
                          approx_eval_time=10,
                          do_visualize=True
                          )
    config = DefaultSearchConfiguration()
    experiment = Experiment(name="3",
                            problem=problem,
                            algorithm=AlgorithmType.NSGAII,
                            search_configuration=config)

    return experiment
    
experiments_store.register(getExp3())

'''
Rastrigin SOO problem (n_var = 2, n_obj = 1), test for PSO
'''
def getExp4() -> Experiment:
    problem = PymooTestProblem(
        'rastrigin',
        critical_function=CriticalRastrigin())

    config = DefaultSearchConfiguration()
    config.maximal_execution_time = "00:00:01"
    experiment = Experiment(name="4",
                            problem=problem,
                            algorithm=AlgorithmType.PSO,
                            search_configuration=config)

    return experiment
    
experiments_store.register(getExp4())
