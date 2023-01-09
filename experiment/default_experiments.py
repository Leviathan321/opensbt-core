import os
from evaluation.fitness import *
from problem.adas_problem import ADASProblem
from problem.pymoo_test_problem import PymooTestProblem
from experiment.experiment import *
from algorithm.algorithm import *
from evaluation.critical import *

class DefaultExperiments(object):
    store: Dict = {}
    def register(self, exp: Experiment):
        if not exp.name in self.store:
            self.store[exp.name] = exp
            return 0
        else:
            print("Experiment with the given name is already registered.")
            return 1

    def load(self, experiment_name: str) -> Experiment:
        if experiment_name in self.store:
            return self.store[experiment_name]
        else:
            print("Experiment with the given name does not exist.")
            return 1

    def get_store(self):
        return self.store


experiments_store = DefaultExperiments()

'''
EXAMPLE CARLA SIMULATOR
ego speed is in km/h
'''

def getExp1() -> Experiment:
    from simulation.carla_simulation import CarlaSimulator

    problem = ADASProblem(
                          scenario_path=os.getcwd() + "/scenarios/PedestrianCrossing.xosc",
                          xl=[0.5, 1, 0],
                          xu=[3, 80, 60],
                          simulation_variables=[
                              "PedestrianSpeed",
                              "FinalHostSpeed",
                              "PedestrianEgoDistanceStartWalk"],
                          fitness_function=FitnessMinDistanceVelocity(),
                          simulate_function=CarlaSimulator.simulate,
                          critical_function=CriticalAdasFrontCollisions(),
                          simulation_time=10,
                          sampling_time=100,
                          problem_name="PedestrianCrossingStartWalk",
                          approx_eval_time=10,
                          do_visualize = False
                          )
    config = DefaultSearchConfiguration()
    experiment = Experiment(name="1",
                            problem=problem,
                            algorithm=AlgorithmType.NSGAII,
                            search_configuration=config)
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