import os
from evaluation.fitness import *
from problem.adas_problem import ADASProblem
from problem.pymoo_test_problem import PymooTestProblem
from experiment.experiment_store import *
from algorithm.algorithm import *
from evaluation.critical import *

'''
Rastrigin SOO problem (n_var = 2, n_obj = 1), test for PSO
'''
def getExp5() -> Experiment:
    problem = PymooTestProblem(
        'rastrigin',
        critical_function=CriticalRastrigin())

    config = DefaultSearchConfiguration()
    config.maximal_execution_time = "00:00:01"
    experiment = Experiment(name="1",
                            problem=problem,
                            algorithm=AlgorithmType.PSO,
                            search_configuration=config)

    return experiment
    
experiments_store.register(getExp5())

''' FOCETA Prescan experiment for search with NSGAII'''
def getExp6() -> Experiment:
    from simulation.prescan_simulation import PrescanSimulator

    problem = ADASProblem(
                          scenario_path=os.getcwd() + "/../FOCETA/experiments/ASE_Leuven_PedestrianCrossing/Leuven_AVP_ori/Demo_AVP.pb",
                          simulation_variables=[
                              "Ego_HostVelGain",  # in m/s
                              "Other_Velocity_mps",  # , # in m/s
                              "Other_Time_s",  # in s,
                          ],
                          xl=[0, 0.5, 0],
                          xu=[5, 2, 5],
                          fitness_function=FitnessMinDistanceVelocity(),
                          critical_function=CriticalAdasDistanceVelocity(),
                          simulate_function=PrescanSimulator.simulate,
                          problem_name="PedestrianCrossingPrescan")

    config = DefaultSearchConfiguration()
    config.n_generations = 5
    config.population_size = 2
    experiment = Experiment(name="2",
                            problem=problem,
                            algorithm=AlgorithmType.NSGAII,
                            search_configuration=config)

    return experiment

experiments_store.register(getExp6())
