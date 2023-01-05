import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from model_ga.individual import *
from model_ga.population import *
from utils.sorting import get_nondominated_population
import os
import csv
from visualization.output import *
from quality_indicators.quality import EvaluationResult
from matplotlib import pyplot as plt
from math import floor
from numpy import random
import scipy
from scipy.interpolate import interp1d
import warnings
import sys
import scipy
from scipy.interpolate import interp1d
import warnings
import sys

BACKUP_FOLDER = "backup" + os.sep

# TODO add information on the deviation of the values wrt. to differen runs in the plots

def calculate_combined_pf(run_paths):
    pf_pop = Population()
    for run_path in run_paths:
        pf_run = read_pf_single(run_path + os.sep + "optimal_individuals.csv")
        pf_pop = Population.merge(pf_pop,pf_run)
    pf_pop = get_nondominated_population(pf_pop)
    inds = [ind.get("F").tolist() for ind in pf_pop]
    pf = np.array(inds, dtype=float)
    # print(pf)
    return pf

def plot_combined_spread_analysis(run_paths_array, save_folder):
    number_of_fitting_points = 7

    for key, (algo, run_paths) in enumerate(run_paths_array.items()):
        num_evals_limit = []
        for run_path in run_paths:
            eval_res = EvaluationResult.load(run_path + os.sep + BACKUP_FOLDER,"spread")
            n_evals, hv = eval_res.steps, eval_res.values
            num_evals_limit.append(n_evals[-1])

        max_num_evals = max(num_evals_limit)
        min_num_evals = min(num_evals_limit)

        step = min_num_evals // number_of_fitting_points
        x = np.arange(step, max_num_evals, step=step)
        y_stacked = np.zeros((len(run_paths), len(x)))

        n_evals_combined = []
        hv_combined = []

        f = plt.figure(figsize=(7, 5))
        for key_run, run_path in enumerate(run_paths):
            eval_res = EvaluationResult.load(run_path + os.sep + BACKUP_FOLDER,"spread")
            n_evals, hv = eval_res.steps, eval_res.values
            spl = scipy.interpolate.interp1d(np.array(n_evals), np.array(hv))
            x_run = np.arange(step, n_evals[-1], step)
            y = spl(x_run)
            y_stacked[key_run, 0:len(y)] = y

            plt.plot(n_evals, hv, marker='.', linestyle='--', label='run ' + str(key_run + 1))

        y_mean = np.sum(y_stacked, axis=0) / np.count_nonzero(y_stacked, axis=0)
        y_error = []
        for i in range(len(y_mean)):
            square_sum = 0
            n = 0
            for j in range(len(run_paths)):
                if np.all(y_stacked[j, i]):
                    n += 1
                    deviation = y_stacked[j, i] - y_mean[i]
                    square_sum += deviation ** 2
            if n <= 1:
                variance = 0
            else:
                variance = square_sum / (n - 1)
            standart_deviation = np.sqrt(variance)
            y_error.append(standart_deviation)

        x_plot = np.arange(step, min_num_evals, step=step)
        plt.plot(x_plot, y_mean[0:len(x_plot)], color='black', marker='o', lw=2, label='combined')
        plt.errorbar(x_plot, y_mean[0:len(x_plot)], y_error[0:len(x_plot)], fmt='.k', capsize=5)
        plt.legend(loc='best')

        plt.title(f"Spread Analysis ({algo})")
        plt.xlabel("Generation")
        plt.ylabel("SP")
        plt.savefig(save_folder + 'spread_combined_' + str(algo) + '.png')
        plt.clf()
        plt.close(f)


def plot_combined_igd_analysis(run_paths_array, save_folder):
    number_of_fitting_points = 7

    for key, (algo, run_paths) in enumerate(run_paths_array.items()):
        num_evals_limit = []
        for run_path in run_paths:
            eval_res = EvaluationResult.load(run_path + os.sep + BACKUP_FOLDER,"igd")
            n_evals, hv = eval_res.steps, eval_res.values
            num_evals_limit.append(n_evals[-1])

        max_num_evals = max(num_evals_limit)
        min_num_evals = min(num_evals_limit)

        step = min_num_evals // number_of_fitting_points
        x = np.arange(step, max_num_evals, step=step)
        y_stacked = np.zeros((len(run_paths), len(x)))

        n_evals_combined = []
        hv_combined = []

        f = plt.figure(figsize=(7, 5))
        for key_run, run_path in enumerate(run_paths):
            eval_res = EvaluationResult.load(run_path + os.sep + BACKUP_FOLDER,"igd")
            n_evals, hv = eval_res.steps, eval_res.values
            spl = scipy.interpolate.interp1d(np.array(n_evals), np.array(hv))
            x_run = np.arange(step, n_evals[-1], step)
            y = spl(x_run)
            y_stacked[key_run, 0:len(y)] = y

            plt.plot(n_evals, hv, marker='.', linestyle='--', label='run ' + str(key_run + 1))

        y_mean = np.sum(y_stacked, axis=0) / np.count_nonzero(y_stacked, axis=0)
        y_error = []
        for i in range(len(y_mean)):
            square_sum = 0
            n = 0
            for j in range(len(run_paths)):
                if np.all(y_stacked[j, i]):
                    n += 1
                    deviation = y_stacked[j, i] - y_mean[i]
                    square_sum += deviation ** 2
            if n <= 1:
                variance = 0
            else:
                variance = square_sum / (n - 1)
            standart_deviation = np.sqrt(variance)
            y_error.append(standart_deviation)

        x_plot = np.arange(step, min_num_evals, step=step)
        plt.plot(x_plot, y_mean[0:len(x_plot)], color='black', marker='o', lw=2, label='combined')
        plt.errorbar(x_plot, y_mean[0:len(x_plot)], y_error[0:len(x_plot)], fmt='.k', capsize=5)
        plt.legend(loc='best')

        plt.title(f"IGD Analysis ({algo})")
        plt.xlabel("Generation")
        plt.ylabel("IGD")
        plt.savefig(save_folder + 'igd_combined_' + str(algo) + '.png')
        plt.clf()
        plt.close(f)

def plot_combined_hypervolume_lin_analysis(run_paths_array, save_folder):
    for key, (algo,run_paths) in enumerate(run_paths_array.items()):
            if len(run_paths) == 0:
                print("Path list is empty")
                return

            f = plt.figure(figsize=(7, 5))    
            plt.title(f"Performance Analysis ({algo})")
            plt.xlabel("Evaluations")
            plt.ylabel("Hypervolume")

            n_runs = len(run_paths)
            hv_run = []
            evals_run = []

            cmap = plt.get_cmap('gnuplot')
            colors = [cmap(i) for i in np.linspace(0, 1,n_runs)]

            for ind, run_path in enumerate(run_paths):      
                eval_res = EvaluationResult.load(run_path + os.sep + BACKUP_FOLDER,"hv")
                n_evals, hv = eval_res.steps, eval_res.values
                plt.plot(n_evals, hv, marker='o', linestyle=":", linewidth=0.5, markersize=3 , color=colors[ind], label=f"run: " + str(ind + 1))
                hv_run.append(hv)
                evals_run.append(n_evals)

            def get_interpol_value(pos, n_evals, hv):
                #print(hv)
                #print(n_evals)
                for ind, eval in enumerate(n_evals):
                    if n_evals[ind] > pos:
                        if ind == 0:
                            value = (hv[ind]) /2
                        else:
                            diff = pos - n_evals[ind-1]
                            grad = (hv[ind] - hv[ind-1])/ (n_evals[ind] - n_evals[ind -1])
                            value = hv[ind-1] + grad*diff
                        return value
                        
            step = 10
            last_n_evals = [n_evals[-1] for n_evals in evals_run]
            #n_steps = floor(min(last_n_evals)/step)
            last_n_eval = min(last_n_evals)
            n_evals_comb = np.arange(0,last_n_eval, step)

            for ind in range(0, n_runs):
                n_evals = evals_run[ind] 
                hv = hv_run[ind]
                hv_run[ind] = [get_interpol_value(val, n_evals, hv) for val in n_evals_comb]
                #print(hv_comb_all)
                
            hv_comb_all = np.sum(hv_run,axis=0)
            hv_comb = np.asarray(hv_comb_all) / n_runs

            plt.plot(n_evals_comb, hv_comb, marker='o', linewidth=1, markersize=3 , color='black', label=f"combined")
          
            plt.legend(loc='best')
            #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.savefig(save_folder + 'hypervolume_combined_' + str(algo) + '.png')
            plt.clf()
            plt.close(f)


def plot_combined_hypervolume_analysis(run_paths_array, save_folder):
    number_of_fitting_points = 7

    for key, (algo, run_paths) in enumerate(run_paths_array.items()):
        num_evals_limit = []
        for run_path in run_paths:
            eval_res = EvaluationResult.load(run_path + os.sep + BACKUP_FOLDER,"hv")
            n_evals, hv = eval_res.steps, eval_res.values
            num_evals_limit.append(n_evals[-1])

        max_num_evals = max(num_evals_limit)
        min_num_evals = min(num_evals_limit)

        step = min_num_evals // number_of_fitting_points
        x = np.arange(step, max_num_evals, step=step)
        y_stacked = np.zeros((len(run_paths), len(x)))

        n_evals_combined = []
        hv_combined = []

        f = plt.figure(figsize=(7, 5))
        for key_run, run_path in enumerate(run_paths):
            eval_res = EvaluationResult.load(run_path + os.sep + BACKUP_FOLDER,"hv")
            n_evals, hv = eval_res.steps, eval_res.values
            spl = scipy.interpolate.interp1d(np.array(n_evals), np.array(hv))
            x_run = np.arange(step, n_evals[-1], step)
            y = spl(x_run)
            y_stacked[key_run, 0:len(y)] = y

            plt.plot(n_evals, hv, marker='.', linestyle='--', label='run ' + str(key_run + 1))

        y_mean = np.sum(y_stacked, axis=0) / np.count_nonzero(y_stacked, axis=0)
        y_error = []
        for i in range(len(y_mean)):
            square_sum = 0
            n = 0
            for j in range(len(run_paths)):
                if np.all(y_stacked[j, i]):
                    n += 1
                    deviation = y_stacked[j, i] - y_mean[i]
                    square_sum += deviation ** 2
            if n <= 1:
                variance = 0
            else:
                variance = square_sum / (n - 1)
            standart_deviation = np.sqrt(variance)
            y_error.append(standart_deviation)

        x_plot = np.arange(step, min_num_evals, step=step)
        plt.plot(x_plot, y_mean[0:len(x_plot)], color='black', marker='o', lw=2, label='combined')
        plt.errorbar(x_plot, y_mean[0:len(x_plot)], y_error[0:len(x_plot)], fmt='.k', capsize=5)
        plt.legend(loc='best')

        plt.title(f"Performance Analysis ({algo})")
        plt.xlabel("Number of evaluations")
        plt.ylabel("Hypervolume")
        plt.savefig(save_folder + 'hypervolume_combined_' + str(algo) + '.png')
        plt.clf()
        plt.close(f)


def write_analysis_results(result_runs_all, save_folder):

    evaluations_all = {}
    n_critical_all = {}
    mean_critical_all = {}
    mean_evaluations_all = {}

    #temporary
    algo1 = "NSGAII"
    algo2 = "NSGAII-DT"

    for algo, result_runs in result_runs_all.items():
        if len(result_runs) == 0:
            print("Result list is empty")
            return

        # n evaluations analysis
        n_evals = [ res.algorithm.evaluator.n_eval for res in result_runs ]

        min_n_evals = np.min(n_evals)
        max_n_evals = np.max(n_evals)
        mean_n_evals = np.sum(n_evals) / len(result_runs)

        # time analysis
        exec_time = [res.exec_time for res in result_runs]
        min_exec_time = np.min(exec_time)
        max_exec_time = np.max(exec_time)
        mean_exec_time = np.sum(exec_time) / len(result_runs)

        # # criticality analysis
        pop_all = [res.obtain_all_population() for res in result_runs]
        evaluations_all[algo] = np.asarray(pop_all, dtype=object)
        n_critical_all[algo] = np.asarray([len(evals.divide_critical_non_critical()[0]) for evals in evaluations_all[algo]],dtype=object)

        mean_critical_all[algo] = np.sum(n_critical_all[algo])#/len(result_runs)
        mean_evaluations_all[algo] = np.sum(len(evals) for evals in evaluations_all[algo])#/len(result_runs)

        '''Output of summery of the performance'''
        with open(save_folder + f'analysis_results_{algo}.csv', 'w', encoding='UTF8', newline='') as f:
            write_to = csv.writer(f)

            header = ['Attribute', 'Value']
            write_to.writerow(header)
            write_to.writerow(['min_n_evals', min_n_evals])
            write_to.writerow(['max_n_evals', max_n_evals])
            write_to.writerow(['mean_n_evals', mean_n_evals])

            write_to.writerow(['min_exec_time [s]', min_exec_time])
            write_to.writerow(['max_exec_time [s]', max_exec_time])
            write_to.writerow(['mean_exec_time [s]', mean_exec_time])

            # write_to.writerow(['Mean number critical Scenarios', len(mean_critical)])
            # write_to.writerow(['Mean evaluations all scenarios', len(mean_evaluations_all)])
            # write_to.writerow(['Mean ratio critical/all scenarios', '{0:.2f}'.format(len(critical_all) / len(all_population))])
            # write_to.writerow(['Mean ratio best Critical/best Scenarios', '{0:.2f}'.format(len(critical_best) / len(best_population))])

            f.close()

    ratio_critical_both = np.sum(n_critical_all[algo2]) / np.sum(n_critical_all[algo1])

    '''Output of summery of the performance'''
    with open(save_folder + f'analysis_combined.csv', 'w', encoding='UTF8', newline='') as f:
        write_to = csv.writer(f)

        write_to.writerow([f'Critical Scenarios {algo1}', np.sum(n_critical_all[algo1])])
        write_to.writerow([f'Critical Scenarios {algo2}', np.sum(n_critical_all[algo2])])
        write_to.writerow([f'Ratio Critical Scenarios {algo2}/{algo1}', '{0:.2f}'.format(ratio_critical_both)])

        # write_to.writerow([f'Mean evaluations all scenarios {algo1}', mean_evaluations_all[algo1]])
        # write_to.writerow([f'Mean evaluations all scenarios {algo2}', mean_evaluations_all[algo2]])

        # write_to.writerow([f'Mean critical all {algo1}', '{0:.2f}'.format(mean_critical_all[algo1])])
        # write_to.writerow([f'Mean critical all {algo2}', '{0:.2f}'.format(mean_critical_all[algo2])])

        #write_to.writerow(['Mean ratio best Critical/best Scenarios', '{0:.2f}'.format(len(critical_best) / len(best_population))])

        f.close()


def read_metric_single(filename, metric_name):
    table = pd.read_csv(filename, names=["n_eval", metric_name])
    n_evals = np.array(table["n_eval"][1:].values.tolist(), dtype=float)
    hv = np.array(table[metric_name][1:].values.tolist(), dtype=float)
    return n_evals, hv

def read_pf_single(filename):
    individuals = []
    table = pd.read_csv(filename)
    n_var = -1
    k = 0
    #identify number of objectives
    for col in table.columns[1:]:
        if col.startswith("Fitness_"):
            n_var = k
            break
        k = k + 1
    for i in range(len(table)):
        X = table.iloc[i, 1:n_var + 1].to_numpy()
        F = table.iloc[i, n_var + 1:].to_numpy()
        ind = Individual()
        ind.set("X", X)
        ind.set("F", F)
        individuals.append(ind)
    return Population(individuals=individuals)
