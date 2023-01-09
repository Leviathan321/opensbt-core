import pymoo

from model_ga.individual import IndividualSimulated
pymoo.core.individual.Individual = IndividualSimulated

from model_ga.population import PopulationExtended
pymoo.core.population.Population = PopulationExtended

from model_ga.result  import ResultExtended
pymoo.core.result.Result = ResultExtended

from model_ga.problem import ProblemExtended
pymoo.core.problem.Problem = ProblemExtended

import argparse
import logging
import os
import re
import sys

from algorithm.nsga2_dt_sim import *
from algorithm.nsga2_sim import *
from simulation.simulator import SimulationType
from experiment.default_experiments import *

os.chmod(os.getcwd(), 0o777)
logging.basicConfig(filename="log.txt", filemode='w', level=logging.ERROR)

from pymoo.config import Config

Config.warnings['not_compiled'] = False

results_folder = '/results/single/'

algorithm = None
problem = None
experiment = None

########

parser = argparse.ArgumentParser(description="Pass parameters for search.")
parser.add_argument('-e', dest='exp_number', type=str, action='store',
                    help='Hardcoded example scenario to use [2 to 6].')
parser.add_argument('-i', dest='n_generations', type=int, default=None, action='store',
                    help='Number generations to perform.')
parser.add_argument('-n', dest='size_population', type=int, default=None, action='store',
                    help='The size of the initial population of scenario candidates.')
parser.add_argument('-a', dest='algorithm', type=int, default=None, action='store',
                    help='The algorithm to use for search, 1 for NSGA2, 2 for NSGA2-DT.')
parser.add_argument('-t', dest='maximal_execution_time', type=str, default=None, action='store',
                    help='The time to use for search with nsga2-DT (actual search time can be above the threshold, since algorithm might perform nsga2 iterations, when time limit is already reached.')
parser.add_argument('-f', dest='scenario_path', type=str, action='store',
                    help='The path to the scenario description file/experiment.')
parser.add_argument('-min', dest='var_min', nargs="+", type=float, action='store',
                    help='The lower bound of each parameter.')
parser.add_argument('-max', dest='var_max', nargs="+", type=float, action='store',
                    help='The upper bound of each parameter.')
parser.add_argument('-m', dest='design_names', nargs="+", type=str, action='store',
                    help='The names of the variables to modify.')
parser.add_argument('-dt', dest='max_tree_iterations', type=int, action='store',
                    help='The maximum number of total decision tree generations (when using NSGA2-DT algoritm).')
parser.add_argument('-o', dest='results_folder', type=str, action='store', default=os.sep + "results" + os.sep,
                    help='The name of the folder where the results of the search are stored (default: \\results\\single\\)')
parser.add_argument('-v', dest='do_visualize', action='store_true',
                    help='Whether to use the simuator\'s visualization. This feature is useful for debugging and demonstrations, however it reduces the search performance.')
parser.add_argument('-info', dest='show_info', action='store_true',
                    help='List name of all defined experiments')

args = parser.parse_args()

#######

# list all experiments


if args.show_info:
    print("Experiments with the following names are defined:")
    store = experiments.get_store()
    for name in store.keys():
        print(name)
    
    sys.exit(0)

parser.add_argument('-list', dest='show_info', action='store_false',
                    help='List name of all defined experiments')

if args.exp_number and args.scenario_path:
    print("Flags set not correctly: Experiment file and example experiment cannot be set at the same time")
    sys.exit()
elif not (args.exp_number or args.scenario_path):
    print("Flags set not correctly: No file is provided or no example experiment selected.")
    sys.exit()

###### set experiment
####### have indiviualized imports
if args.exp_number:
    # exp_number provided
    # selExpNumber = re.findall("[1-9]+", args.exp_number)[0]
    # print(f"Selected experiment number: {selExpNumber}")
    #experiment = experiment_switcher.get(int(selExpNumber))()
    experiment = experiments.load(experiment_name=args.exp_number)
    config = experiment.search_configuration
    problem = experiment.problem
    algorithm = experiment.algorithm

elif (args.scenario_path):
    scenario_path = args.scenario_path
    var_min = []
    var_max = []

    #TODO create an experiment from user input
    #TODO create an ADASProblem from user input

    print("-- Experiment provided by file")

    if args.var_min is None:
        print("-- Minimal bounds for search are not set.")
        sys.exit()

    if args.var_max is None:
        print("-- Maximal bounds for search are not set.")
        sys.exit()

    print("Creating an experiment from user input not yet supported. Use default_experiments.py to create experiment")
    sys.exit()

    # # set design names
    # if  args.design_names is None:
    #     design_names = ["feature_" + str(i) for i in range(len(var_min))]

    # if scenario_path.endswith('.pb'):
    #     fitnessFcn = fitness.fitness_min_distance_two_actors_prescan
    #     simulateFcn = PrescanSimulator.simulateBatch_compiled_csv
    # elif scenario_path.endswith('.scenario_path'):
    #     fitnessFcn = fitness.fitness_min_distance_two_actors_carla
    #     simulateFcn = CarlaSimulator.simulateBatch
    # else:
    #     print("-- File is not supported.")
    #     sys.exit()
    # experiment = Experiment()
else:
    print("-- No file provided and no experiment selected")
    sys.exit()

'''
override params if set by user
'''

if not args.size_population is None:
    config.population_size = args.size_population
if not args.n_generations is None:
    config.n_generations = args.n_generations
    config.inner_num_gen = args.n_generations #for NSGAII-DT
if not args.algorithm is None:
    algorithm = AlgorithmType(args.algorithm)
if not args.maximal_execution_time is None:
    config.maximal_execution_time = args.maximal_execution_time
if not args.max_tree_iterations is None:
    config.max_tree_iterations = args.max_tree_iterations
if not args.max_tree_iterations is None:
    results_folder = args.results_folder

if not args.var_max is None:
    problem.var_max = args.var_max
if not args.var_min is None:
    problem.var_min = args.var_min
if not args.design_names is None:
    problem.design_names = args.design_names
if not args.do_visualize is None:
    problem.do_visualize = args.do_visualize

####### Run algorithm

if __name__ == "__main__":
    execTime = None
    algo = None
    if algorithm == AlgorithmType.NSGAII:
        print("pymoo NSGA-II algorithm is used.")
        algo = NSGAII_SIM(
                              problem=problem,
                              config=config)
    elif algorithm == AlgorithmType.NSGAIIDT:
        print("pymoo NSGA-II-DT algorithm is used.")
        algo = NSGAII_DT_SIM(
                              problem=problem,
                              config=config)
    else:
        raise ValueError("Error: No algorithm with the given code: " + str(algorithm))

    res = algo.run()
    algo.write_results(results_folder=results_folder)

    print("====== Algorithm search time: " + str("%.2f" % res.exec_time) + " sec")
