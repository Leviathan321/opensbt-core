import os
from evaluation.critical import CriticalAdasDistanceVelocity
from evaluation.fitness import FitnessMinDistanceVelocityFrontOnly
from experiment.experiment import Experiment
from experiment.experiment_store import experiments_store
from experiment.search_configuration import DefaultSearchConfiguration
from problem.adas_problem import ADASProblem
from simulation.carla_simulation import CarlaSimulator
from algorithm.algorithm import *

''' Use this script to define experiment and run search without using CLI '''

# 1. Defining the problem
problem = ADASProblem(
                        scenario_path=os.getcwd() + "/scenarios/PedestrianCrossing.xosc",
                        xl=[0.5, 1, 0],
                        xu=[3, 80, 60],
                        simulation_variables=[
                            "PedestrianSpeed",
                            "FinalHostSpeed",
                            "PedestrianEgoDistanceStartWalk"],
                        fitness_function=FitnessMinDistanceVelocityFrontOnly(),
                        simulate_function=CarlaSimulator.simulate,
                        critical_function=CriticalAdasDistanceVelocity(),
                        simulation_time=10,
                        sampling_time=100,
                        problem_name="PedestrianCrossingStartWalk",
                        approx_eval_time=10,
                        do_visualize = False
                        )

# 2. Defining the algorithm and search configuration
# # TODO algorithm = "NSGA"

config = DefaultSearchConfiguration()
config.maximal_execution_time = "00:00:30"

# 3. Defining the experiment
experiment = Experiment(name="1",
                        problem=problem,
                        algorithm=AlgorithmType.NSGAII,
                        search_configuration=config)

experiments_store.register(experiment)

# 4. Run experiment

## a) via code

## experiment.run()

## b) via cli

## python run.py -e 1