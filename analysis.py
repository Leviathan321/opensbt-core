
import pymoo

from model_ga.individual import IndividualSimulated
pymoo.core.individual.Individual = IndividualSimulated

from model_ga.population import PopulationExtended
pymoo.core.population.Population = PopulationExtended

from model_ga.result  import ResultExtended
pymoo.core.result.Result = ResultExtended

from model_ga.problem import ProblemExtended
pymoo.core.problem.Problem = ProblemExtended

from algorithm.nsga2dte_optimizer import NSGAII_DT_SIM
from algorithm.nsga2dt_optimizer import NSGAII_SIM
from experiment.default_experiments import getExp6
from visualization.combined import *
import os
from datetime import datetime
from pathlib import Path
import shutil
from visualization import output
from experiment.search_configuration import *
import sys
import argparse
import re
from experiment.default_experiments import *
from utils.path import get_subfolders_from_folder

parser = argparse.ArgumentParser(description="Pass parameters for analysis.")
parser.add_argument('-r', dest='n_runs', type=int, default=None, action='store',
                    help='Number runs to perform each algorithm for statistical analysis.')
parser.add_argument('-p', dest='folder_runs', type=str, default=None, action='store',
                    help='The folder of the results written after executed runs of both algorithms. Path needs to end with "/".')

args = parser.parse_args()

####

BACKUP_FOLDER =  "backup" + os.sep

#### number of runs
n_runs_default = 10

if args.n_runs is None:
    n_runs = n_runs_default
else:
    n_runs = args.n_runs

run_paths_all =  { "NSGAII" : [], "NSGAII-DT": []}

if args.folder_runs:
    ''' provide folder of finished runs to do analysis '''

    run_paths_all["NSGAII"] = get_subfolders_from_folder(
                                            args.folder_runs + "NSGA-II" + os.sep )
    run_paths_all["NSGAII-DT"] = get_subfolders_from_folder(
                                            args.folder_runs  + "NSGA-II-DT" + os.sep) 

    analysis_folder = args.folder_runs
else:
    ''' run the algorithms first and then do analysis '''
    exp = getExp6()
    problem = exp.problem
    problem_name = problem.problem_name

    ##### search configurations
    config_1 = DefaultSearchConfiguration()
    config_1.population_size = 5
    config_1.n_generations = 10
    config_1.maximal_execution_time = "00:00:01"

    config_2 = DefaultSearchConfiguration()
    config_2.population_size = 5
    config_2.maximal_execution_time = "00:00:01"
    config_2.inner_num_gen = 4

    ##### results are written in results/analysis/<problem>/<n_runs>/<date>/

    analysis_folder = str(os.getcwd()) + os.sep + "results" + os.sep + "analysis" + os.sep + problem_name + os.sep +  str(n_runs) + "_runs" + os.sep + datetime.now().strftime(
            "%d-%m-%Y_%H-%M-%S") + os.sep

    def create_run_folder(analysis_folder, algorithm, run_num):
        i = run_num
        run_folder = analysis_folder + str(algorithm) + os.sep + str(f"run_{i}") + os.sep
        Path(run_folder).mkdir(parents=True, exist_ok=True)
        return run_folder

    ##### run search

    def write_results_reduced(res, results_folder):
        output.hypervolume_analysis(res, results_folder)
        output.spread_analysis(res, results_folder)
        #output.design_space(res, run_folder)
        output.objective_space(res, run_folder)
        output.optimal_individuals(res, results_folder)
        output.write_summary_results(res, results_folder)
        #output.write_simulation_output(res,run_folder)
        #output.simulations(res, run_folder)

    for i in range(1,n_runs+1):
        print(f"Running run {i} from {n_runs} with NSGA-II")
        run_folder = create_run_folder(analysis_folder,"NSGA-II",i)
        algo = NSGAII_SIM(
                            problem=problem,
                            config=config_1)
        res = algo.run()

        print("----- Storing result object ------")
        res.persist(run_folder + BACKUP_FOLDER)
        
        print("----- Reduced writing of results ------")
        write_results_reduced(res, run_folder)

        run_paths_all["NSGAII"].append(run_folder)
        #res_nsga2.append(res)
        print(f"---- Evaluating run {i} from {n_runs} with NSGA-II completed ----")

    for i in range(1,n_runs+1):
        print(f"Running run {i} from {n_runs} with NSGA-II-DT")
        run_folder = create_run_folder(analysis_folder,"NSGA-II-DT", i)

        algo = NSGAII_DT_SIM(
                                problem=problem,
                                config=config_2)
        res = algo.run()
        
        print("----- Storing result object ------")
        res.persist(run_folder + BACKUP_FOLDER)
        
        print("----- Reduced writing of results ------")
        write_results_reduced(res, run_folder)

        run_paths_all["NSGAII-DT"].append(run_folder)
        #res_nsga2_dt.append(res)
        print(f"---- Evaluating run {i} from {n_runs} with NSGA-II-DT completed ----")

    print("---- All runs completed. ----")
    
print("---- Calculating estimated pareto front. ----")

# calculate estimated pareto front

pf_estimated = calculate_combined_pf(run_paths_all["NSGAII"] + run_paths_all["NSGAII-DT"])
# print(f"estimated pf: {pf_estimated}")

# perform igd analysis/create plots 

result_runs_all = { "NSGAII" : [], 
                    "NSGAII-DT": []} 

for run_path in run_paths_all["NSGAII"]:
    backup_path = run_path + BACKUP_FOLDER
    # load result
    res = ResultExtended.load(backup_path)

    result_runs_all["NSGAII"].append(res)
    output.convergence_analysis(res, run_path, input_pf=pf_estimated)

for run_path in run_paths_all["NSGAII-DT"]:
    backup_path = run_path + BACKUP_FOLDER
    # load result
    res = ResultExtended.load(backup_path)

    result_runs_all["NSGAII-DT"].append(res)
    output.convergence_analysis(res, run_path, input_pf=pf_estimated)

# create combined hv/igd/sp plots

plot_combined_hypervolume_analysis(run_paths_all, analysis_folder)

plot_combined_spread_analysis(run_paths_all, analysis_folder)

plot_combined_igd_analysis(run_paths_all, analysis_folder)

print("---- Analysis plots generated. ----")

write_analysis_results(result_runs_all, analysis_folder)
#TODO add more information into written common analysis results

print("---- Analysis summary written to file. ")
