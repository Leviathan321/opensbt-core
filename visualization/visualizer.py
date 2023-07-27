import csv
import math
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import algorithm.classification.decision_tree.decision_tree as decision_tree
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerPatch
from utils.sampling import CartesianSampling
from visualization import scenario_plotter as plotter
from pymoo.indicators.igd import IGD
from pymoo.indicators.hv import Hypervolume
from pymoo.core.population import Population
from visualization.configuration import *
from utils.sorting import *
from algorithm.classification.classifier import ClassificationType
from quality_indicators.quality import Quality
from model_ga.problem import *
from model_ga.result import *
from typing import Dict
from utils.duplicates import duplicate_free
from utils.sorting import get_nondominated_population

WRITE_ALL_INDIVIDUALS = True
BACKUP_FOLDER =  "backup" + os.sep

def create_save_folder(problem: Problem, results_folder: str, algorithm_name: str, is_experimental=False):
    problem_name = problem.problem_name
    # algorithm_name = type(res.algorithm).__name__
    # if results folder is already a valid folder, do not create it in parent, use it relative

    if os.path.isdir(results_folder):
        save_folder = results_folder 
        #+ problem_name + os.sep + algorithm_name + os.sep + datetime.now().strftime(
        #   "%d-%m-%Y_%H-%M-%S") + os.sep
    elif is_experimental:
        save_folder = str(
            os.getcwd()) + results_folder + problem_name + os.sep + algorithm_name + os.sep + "temp" + os.sep
    else:
        save_folder = str(
            os.getcwd()) + results_folder + problem_name + os.sep + algorithm_name + os.sep + datetime.now().strftime(
            "%d-%m-%Y_%H-%M-%S") + os.sep
    print(f"save_folder created: {save_folder}")
    Path(save_folder).mkdir(parents=True, exist_ok=True)
    return save_folder

def write_calculation_properties(res: Result, save_folder: str, algorithm_name: str, algorithm_parameters: Dict, **kwargs):
    problem = res.problem
    # algorithm_name = type(res.algorithm).__name__
    is_simulation = problem.is_simulation()
    with open(save_folder + 'calculation_properties.csv', 'w', encoding='UTF8', newline='') as f:
        write_to = csv.writer(f)

        header = ['Attribute', 'Value']
        write_to.writerow(header)
        write_to.writerow(['Problem', problem.problem_name])
        write_to.writerow(['Algorithm', algorithm_name])
        write_to.writerow(['Search variables', problem.design_names])        
        write_to.writerow(['Search space', [v for v in zip(problem.xl,problem.xu)]])

        if is_simulation:
            write_to.writerow(['Fitness function', str(problem.fitness_function.name)])
        else:
            write_to.writerow(['Fitness function', "<No name available>"])

        write_to.writerow(['Critical function', str(problem.critical_function.name)])
        # write_to.writerow(['Number of maximal tree generations', str(max_tree_iterations)])
        write_to.writerow(['Search time', str("%.2f" % res.exec_time + " sec")])

        for item,value in algorithm_parameters.items():
            write_to.writerow([item, value])

        _additional_descritption(res, save_folder, algorithm_name, **kwargs)

        f.close()

    _calc_properties(res, save_folder, algorithm_name, **kwargs)


def _calc_properties(res, save_folder, algorithm_name, **kwargs):
    pass

def _additional_descritption(res, save_folder, algorithm_name, **kwargs):
    pass

'''Output of the simulation data for all solutions (for the moment only partial data)'''
def write_simulation_output(res: Result, save_folder: str):
    
    problem = res.problem

    if not problem.is_simulation():
        return

    all_population = res.obtain_all_population()

    with open(save_folder + 'simulation_output.csv', 'w', encoding='UTF8', newline='') as f:
        write_to = csv.writer(f)
        header = ['Index']
        other_params = all_population.get("SO")[0].otherParams

        # write header
        for item, value in other_params.items():
            if isinstance(value,float) or isinstance(value,int) or isinstance(value,bool):
                header.append(item)
        write_to.writerow(header)

        # write values
        for index in range(len(all_population)):
            row = [index]
            other_params = all_population.get("SO")[index].otherParams
            for item, value in other_params.items():
                if isinstance(value,float):
                    row.extend(["%.2f" % value])
                if isinstance(value,int) or isinstance(value,bool):
                    row.extend([value])
            write_to.writerow(row)
        f.close()

def digd_analysis(res: Result, save_folder: str, input_crit=None, filename='digd'):
    # print("------ Performing igd analysis ------")

    eval_result = Quality.calculate_digd(res,input_crit=input_crit)
    if eval_result is None:
        return

    n_evals, gd = eval_result.steps, eval_result.values

    # store
    eval_result.persist(save_folder + BACKUP_FOLDER)
    write_metric_history(n_evals, gd,'digd_all',save_folder)

    # plot
    f = plt.figure()
    plt.plot(n_evals, gd, color='black', lw=0.7)
    plt.scatter(n_evals, gd, facecolor='none', edgecolor='black', marker='o')
    plt.title("Design Space Convergence Analysis")
    plt.xlabel("Function Evaluations")
    plt.ylabel("dIGD")
    # plt.yscale("log")
    plt.savefig(save_folder + filename + '.png')
    plt.clf()
    plt.close(f)

    # output to console
    print(f"Final dIGD value: {gd[-1]}")

def gd_analysis(res: Result, save_folder: str, input_pf=None, filename='gd', mode='default'):
    # print("------ Performing igd analysis ------")

    eval_result = Quality.calculate_gd(res, input_pf=input_pf, mode=mode)
    if eval_result is None:
        return

    n_evals, gd = eval_result.steps, eval_result.values

    # store
    eval_result.persist(save_folder + BACKUP_FOLDER)
    write_metric_history(n_evals, gd,'gd_all' + '_' + mode,save_folder)

    # plot
    f = plt.figure()
    plt.plot(n_evals, gd, color='black', lw=0.7)
    plt.scatter(n_evals, gd, facecolor='none', edgecolor='black', marker='o')
    plt.title("Convergence Analysis")
    plt.xlabel("Function Evaluations")
    plt.ylabel("GD")
    # plt.yscale("log")
    plt.savefig(save_folder + filename + '_' + mode + '.png')
    plt.clf()
    plt.close(f)

    # output to console
    print(f"Final GD value: {gd[-1]}")

def gd_analysis_hitherto(res: Result, save_folder: str, input_pf=None, filename='gd_global', mode='default'):
    print("------ Performing igd analysis ------")

    eval_result = Quality.calculate_gd_hitherto(res, input_pf=input_pf, mode=mode)
    if eval_result is None:
        return

    n_evals, gd = eval_result.steps, eval_result.values

    # store
    eval_result.persist(save_folder + BACKUP_FOLDER)
    write_metric_history(n_evals, gd,'gd_global' + '_' + mode,save_folder)

    # plot
    f = plt.figure()
    plt.plot(n_evals, gd, color='black', lw=0.7)
    plt.scatter(n_evals, gd, facecolor='none', edgecolor='black', marker='o')
    plt.title("Convergence Analysis")
    plt.xlabel("Function Evaluations")
    plt.ylabel("GD")
    # plt.yscale("log")
    plt.savefig(save_folder + filename + '_' + mode + '.png')
    plt.clf()
    plt.close(f)

def igd_analysis(res: Result, save_folder: str, input_pf=None, filename='igd'):
    # print("------ Performing igd analysis ------")

    eval_result = Quality.calculate_igd(res, input_pf=input_pf)
    if eval_result is None:
        return

    n_evals, igd = eval_result.steps, eval_result.values

    # store 
    eval_result.persist(save_folder + BACKUP_FOLDER)
    write_metric_history(n_evals, igd,'igd_all',save_folder)

    # plot
    f = plt.figure()
    plt.plot(n_evals, igd, color='black', lw=0.7)
    plt.scatter(n_evals, igd, facecolor='none', edgecolor='black', marker='o')
    plt.title("Convergence Analysis")
    plt.xlabel("Function Evaluations")
    plt.ylabel("IGD")
    # plt.yscale("log")
    plt.savefig(save_folder + filename + '.png')
    plt.clf()
    plt.close(f)

    # output to console
    print(f"Final IGD value: {igd[-1]}")


def igd_analysis_hitherto(res: Result, save_folder: str, input_pf=None, filename='igd_global'):
    # print("------ Performing igd analysis ------")

    eval_result = Quality.calculate_igd_hitherto(res, input_pf=input_pf)
    if eval_result is None:
        return

    n_evals, igd = eval_result.steps, eval_result.values

    # store 
    eval_result.persist(save_folder + BACKUP_FOLDER)
    write_metric_history(n_evals, igd,'igd_global',save_folder)

    # plot
    f = plt.figure()
    plt.plot(n_evals, igd, color='black', lw=0.7)
    plt.scatter(n_evals, igd, facecolor='none', edgecolor='black', marker='o')
    plt.title("Convergence Analysis")
    plt.xlabel("Function Evaluations")
    plt.ylabel("IGD")
    # plt.yscale("log")
    plt.savefig(save_folder + filename + '.png')
    plt.clf()
    plt.close(f)


def write_metric_history(n_evals, hist_F, metric_name, save_folder):
    history_folder = save_folder + "history" + os.sep
    Path(history_folder).mkdir(parents=True, exist_ok=True)
    with open(history_folder+ '' + metric_name + '.csv', 'w', encoding='UTF8', newline='') as f:
        write_to = csv.writer(f)
        header = ['n_evals', metric_name]
        write_to.writerow(header)
        for i,val in enumerate(n_evals):
            write_to.writerow([n_evals[i], hist_F[i]])
        f.close()

def hypervolume_analysis(res, save_folder):
    # print("------ Performing hv analysis ------")
   
    eval_result = Quality.calculate_hv_hitherto(res)
 
    if eval_result is None:
        return
    n_evals, hv = eval_result.steps, eval_result.values

    # store
    eval_result.persist(save_folder + BACKUP_FOLDER)
    write_metric_history(n_evals, hv, 'hv_all', save_folder)

    # plot
    plt.figure(figsize=(7, 5))
    plt.plot(n_evals, hv, color='black', lw=0.7)
    plt.scatter(n_evals, hv, facecolor="none", edgecolor='black', marker='o')
    plt.title("Performance Analysis")
    plt.xlabel("Function Evaluations")
    plt.ylabel("Hypervolume")
    plt.savefig(save_folder + 'hypervolume.png')

    # output to console
    print(f"Final HV value: {hv[-1]}")

def hypervolume_analysis_local(res, save_folder):
    print("------ Performing hv analysis ------")

    eval_result = Quality.calculate_hv(res)

    if eval_result is None:
        return

    n_evals, hv = eval_result.steps, eval_result.values

    # store
    eval_result.persist(save_folder + BACKUP_FOLDER)
    write_metric_history(n_evals, hv,'hv_local_all',save_folder)    

    # plot
    plt.figure(figsize=(7, 5))
    plt.plot(n_evals, hv, color='black', lw=0.7)
    plt.scatter(n_evals, hv, facecolor="none", edgecolor='black', marker='o')
    plt.title("Performance Analysis")
    plt.xlabel("Function Evaluations")
    plt.ylabel("Hypervolume")
    plt.savefig(save_folder + 'hypervolume_local.png')


def spread_analysis(res, save_folder):
    # print("------ Performing sp analysis ------")

    eval_result = Quality.calculate_sp(res)
    if eval_result is None:
        return
    
    n_evals, uniformity = eval_result.steps, eval_result.values
    
    # store
    eval_result.persist(save_folder + BACKUP_FOLDER)
    write_metric_history(n_evals,uniformity,'sp',save_folder)

    # plot
    plt.figure(figsize=(7, 5))
    plt.plot(n_evals, uniformity, color='black', lw=0.7)
    plt.scatter(n_evals, uniformity, facecolor="none", edgecolor='black', marker='o')
    plt.title("Spreadness/Uniformity Analysis")
    plt.xlabel("Function Evaluations")
    plt.ylabel("SP")
    plt.savefig(save_folder + 'spread.png')

    # output to console
    print(f"Final SP value: {uniformity[-1]}")

def spread_analysis_hitherto(res, save_folder, hitherto = False):
    print("------ Performing sp analysis ------")

    eval_result = Quality.calculate_sp_hitherto(res)
    if eval_result is None:
        return
    
    n_evals, uniformity = eval_result.steps, eval_result.values
    
    # store
    eval_result.persist(save_folder + BACKUP_FOLDER)
    write_metric_history(n_evals,uniformity,'sp_global',save_folder)

    # plot
    plt.figure(figsize=(7, 5))
    plt.plot(n_evals, uniformity, color='black', lw=0.7)
    plt.scatter(n_evals, uniformity, facecolor="none", edgecolor='black', marker='o')
    plt.title("Spreadness/Uniformity Analysis")
    plt.xlabel("Function Evaluations")
    plt.ylabel("SP")
    plt.savefig(save_folder + 'spread_global.png')

class HandlerCircle(HandlerPatch):
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        center = 0.5 * width - 0.5 * xdescent, 0.5 * height - 0.5 * ydescent
        p = mpatches.Ellipse(xy=center, width=min(width, height),
                             height=min(width, height))
        self.update_prop(p, orig_handle, legend)
        p.set_transform(trans)
        return [p]

def create_markers():
    patch_not_critical_region = mpatches.Patch(color=color_not_critical, label='Not critical regions',
                                               alpha=0.05)
    patch_critical_region = mpatches.Patch(color=color_critical, label='Critical regions', alpha=0.05)

    circle_critical = mpatches.Circle((0.5, 0.5), radius=2, facecolor='none',
                                      edgecolor=color_critical, linewidth=1, label='Critical testcases')

    circle_not_critical = mpatches.Circle((0.5, 0.5), radius=2, facecolor='none',
                                          edgecolor=color_not_critical, linewidth=1, label='Not critical testcases')

    circle_optimal = mpatches.Circle((0.5, 0.5), radius=2, facecolor=color_optimal,
                                     edgecolor='none', linewidth=1, label='Optimal testcases')

    circle_not_optimal = mpatches.Circle((0.5, 0.5), radius=2, facecolor=color_not_optimal,
                                         edgecolor='none', linewidth=1, label='Not optimal testcases')

    line_pareto = Line2D([0], [0], label='Pareto front', color='blue')

    marker_list = [patch_not_critical_region, patch_critical_region, circle_critical, circle_not_critical,
                   circle_optimal, circle_not_optimal, line_pareto]

    return marker_list

def write_summary_results(res, save_folder):
    all_population = res.obtain_all_population()
    best_population = res.opt

    critical_best,_ = best_population.divide_critical_non_critical()
    critical_all,_ = all_population.divide_critical_non_critical()
    
    n_crit_all_dup_free = len(duplicate_free(critical_all))
    n_all_dup_free = len(duplicate_free(all_population))
    n_crit_best_dup_free = len(duplicate_free(critical_best))
    n_best_dup_free = len(duplicate_free(best_population))

    '''Output of summery of the performance'''
    with open(save_folder + 'summary_results.csv', 'w', encoding='UTF8', newline='') as f:
        write_to = csv.writer(f)

        header = ['Attribute', 'Value']
        write_to.writerow(header)
        write_to.writerow(['Number Critical Scenarios', len(critical_all)])
        write_to.writerow(['Number Critical Scenarios (duplicate free)', n_crit_all_dup_free])

        write_to.writerow(['Number All Scenarios', len(all_population)])
        write_to.writerow(['Number All Scenarios (duplicate free)', n_all_dup_free])

        write_to.writerow(['Number Best Critical Scenarios', len(critical_best)])
        write_to.writerow(['Number Best Critical Scenarios (duplicate free)', n_crit_best_dup_free])

        write_to.writerow(['Number Best Scenarios', len(best_population)])
        write_to.writerow(['Number Best Scenarios (duplicate free)',n_best_dup_free])

        write_to.writerow(['Ratio Critical/All scenarios', '{0:.2f}'.format(len(critical_all) / len(all_population))])
        write_to.writerow(['Ratio Critical/All scenarios (duplicate free)', '{0:.2f}'.format(n_crit_all_dup_free/n_all_dup_free)])

        write_to.writerow(['Ratio Best Critical/Best Scenarios', '{0:.2f}'.format(len(critical_best) / len(best_population))])
        write_to.writerow(['Ratio Best Critical/Best Scenarios (duplicate free)', '{0:.2f}'.format(n_crit_best_dup_free/n_best_dup_free)])

        f.close()

    print(['Number Critical Scenarios (duplicate free)', n_crit_all_dup_free])
    print(['Number All Scenarios (duplicate free)', n_all_dup_free])
    print(['Ratio Critical/All scenarios (duplicate free)', '{0:.2f}'.format(n_crit_all_dup_free/n_all_dup_free)])

def design_space(res, save_folder, classification_type=ClassificationType.DT, iteration=None):
    save_folder_design = save_folder + "design_space" + os.sep
    Path(save_folder_design).mkdir(parents=True, exist_ok=True)
    save_folder_plot = save_folder_design

    if iteration is not None:
        save_folder_design_iteration = save_folder_design + 'TI_' + str(iteration) + os.sep
        Path(save_folder_design_iteration).mkdir(parents=True, exist_ok=True)
        save_folder_plot = save_folder_design_iteration
 
    hist = res.history
    problem = res.problem
    design_names = problem.design_names
    n_var = problem.n_var
    xl = problem.xl
    xu = problem.xu

    all_population = res.obtain_all_population()
    critical_all, non_critical_all = all_population.divide_critical_non_critical()

    if classification_type == ClassificationType.DT:
        save_folder_classification = save_folder + "classification" + os.sep
        Path(save_folder_classification).mkdir(parents=True, exist_ok=True)
        regions = decision_tree.generate_critical_regions(all_population, problem, save_folder=save_folder_classification)
    
    f = plt.figure(figsize=(12, 10))
    for axis_x in range(n_var - 1):
        for axis_y in range(axis_x + 1, n_var):
            if classification_type == ClassificationType.DT:
                for region in regions:
                    x_rectangle = region.xl[axis_x]
                    y_rectangle = region.xl[axis_y]
                    width_rectangle = region.xu[axis_x] - region.xl[axis_x]
                    height_rectangle = region.xu[axis_y] - region.xl[axis_y]
                    region_color = color_not_critical

                    if region.is_critical:
                        region_color = color_critical
                    plt.gca().add_patch(Rectangle((x_rectangle, y_rectangle), width_rectangle, height_rectangle,
                                                  edgecolor=region_color, lw=1.5, ls='-',
                                                  facecolor='none', alpha=0.2))
                    plt.gca().add_patch(Rectangle((x_rectangle, y_rectangle), width_rectangle, height_rectangle,
                                                  edgecolor='none',
                                                  facecolor=region_color, alpha=0.05))

            ax = plt.subplot(111)
            plt.title("Design Space" + " (" + str(len(all_population)) + " testcases, " + str(len(critical_all)) + " of which are critical)")

            if classification_type == ClassificationType.DT:
                critical, not_critical = all_population.divide_critical_non_critical()
                if len(not_critical) != 0:
                    ax.scatter(not_critical.get("X")[:, axis_x], not_critical.get("X")[:, axis_y],
                                s=40,
                                facecolors=color_not_optimal,
                                edgecolors=color_not_critical, marker='o')
                if len(critical) != 0:
                    ax.scatter(critical.get("X")[:, axis_x], critical.get("X")[:, axis_y], s=40,
                                facecolors=color_not_optimal,
                                edgecolors=color_critical, marker='o')

                
                opt = get_nondominated_population(all_population)
                critical_opt, not_critical_opt = opt.divide_critical_non_critical()

                if len(critical_opt) != 0:
                    ax.scatter(critical_opt.get("X")[:, axis_x], critical_opt.get("X")[:, axis_y], s=40,
                               facecolors=color_optimal,
                               edgecolors=color_critical, marker='o')
                                
                if len(not_critical_opt) != 0:
                    ax.scatter(not_critical_opt.get("X")[:, axis_x], not_critical_opt.get("X")[:, axis_y], s=40,
                               facecolors=color_optimal,
                               edgecolors=color_not_critical, marker='o')


            eta_x = (xu[axis_x] - xl[axis_x]) / 10
            eta_y = (xu[axis_y] - xl[axis_y]) / 10
            plt.xlim(xl[axis_x] - eta_x, xu[axis_x] + eta_x)
            plt.ylim(xl[axis_y] - eta_y, xu[axis_y] + eta_y)
            plt.xlabel(design_names[axis_x])
            plt.ylabel(design_names[axis_y])
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            ax.set_aspect(1.0 / ax.get_data_ratio(), adjustable='box')

            marker_list = create_markers()
            markers = marker_list[:-1]

            plt.legend(handles=markers,
                       loc='center left', bbox_to_anchor=(1, 0.5), handler_map={mpatches.Circle: HandlerCircle()})

            plt.savefig(save_folder_plot + design_names[axis_x] + '_' + design_names[axis_y] + '.png')
            plt.clf()

    plt.close(f)


def objective_space(res, save_folder, iteration=None):
    save_folder_objective = save_folder + "objective_space" + os.sep
    Path(save_folder_objective).mkdir(parents=True, exist_ok=True)   
    save_folder_plot = save_folder_objective

    if iteration is not None:
        save_folder_iteration = save_folder_objective + 'TI_' + str(iteration) + os.sep
        Path(save_folder_iteration).mkdir(parents=True, exist_ok=True)
        save_folder_plot = save_folder_iteration
 
    hist = res.history
    problem = res.problem
    pf = problem.pareto_front()
    n_obj = problem.n_obj
    objective_names = problem.objective_names

    all_population = res.obtain_all_population()
    critical_all, non_critical_all = all_population.divide_critical_non_critical()

    f = plt.figure(figsize=(12, 10))
    for axis_x in range(n_obj - 1):
        for axis_y in range(axis_x + 1, n_obj):
            ax = plt.subplot(111)
            plt.title("Objective Space" + " (" + str(len(all_population)) + " testcases, " + str(len(critical_all)) + " of which are critical)")

            if True: #classification_type == ClassificationType.DT:
                critical, not_critical = all_population.divide_critical_non_critical()

                critical_clean = duplicate_free(critical)
                not_critical_clean = duplicate_free(not_critical)
                
                if len(not_critical_clean) != 0:
                    ax.scatter(not_critical_clean.get("F")[:, axis_x], not_critical_clean.get("F")[:, axis_y], s=40,
                                facecolors=color_not_optimal,
                                edgecolors=color_not_critical, marker='o')
                if len(critical_clean) != 0:
                    ax.scatter(critical_clean.get("F")[:, axis_x], critical_clean.get("F")[:, axis_y], s=40,
                                facecolors=color_not_optimal, edgecolors=color_critical, marker='o')

            if pf is not None:
                ax.plot(pf[:, axis_x], pf[:, axis_y], color='blue', lw=0.7, zorder=1)

            if True: #classification_type == ClassificationType.DT:
                optimal_pop = get_nondominated_population(all_population)
                critical, not_critical = optimal_pop.divide_critical_non_critical()
                critical_clean = duplicate_free(critical)
                not_critical_clean = duplicate_free(not_critical)
                
                if len(not_critical_clean) != 0:
                    ax.scatter(not_critical_clean.get("F")[:, axis_x], not_critical_clean.get("F")[:, axis_y], s=40,
                               facecolors=color_optimal, edgecolors=color_not_critical, marker='o')
                if len(critical_clean) != 0:
                    ax.scatter(critical_clean.get("F")[:, axis_x], critical_clean.get("F")[:, axis_y], s=40,
                               facecolors=color_optimal, edgecolors=color_critical, marker='o')

            #limit axes bounds, since we do not want to show fitness values as 1000 or int.max, 
            # that assign bad quality to worse scenarios
            CONSIDER_HIGH_VAL = True
            if CONSIDER_HIGH_VAL:
                MAX_VALUE = 1000
                MIN_VALUE = -1000
                
                pop_f_x = all_population.get("F")[:,axis_x]
                clean_pop_x = np.delete(pop_f_x, np.where(pop_f_x == MAX_VALUE))
                max_x_f_ind = max(clean_pop_x)
                clean_pop_x = np.delete(pop_f_x, np.where(pop_f_x == MIN_VALUE))
                min_x_f_ind = min(clean_pop_x)

                pop_f_y = all_population.get("F")[:,axis_y]
                clean_pop_y = np.delete(pop_f_y, np.where(pop_f_y == MAX_VALUE))
                max_y_f_ind = max(clean_pop_y)
                clean_pop_y = np.delete(pop_f_y, np.where(pop_f_y == MIN_VALUE))
                min_y_f_ind = min(clean_pop_y)

                eta_x = abs(max_x_f_ind - min_x_f_ind) / 10
                eta_y = abs(max_y_f_ind- min_y_f_ind) / 10
                
                plt.xlim(min_x_f_ind - eta_x, max_x_f_ind  + eta_x)
                plt.ylim(min_y_f_ind - eta_y, max_y_f_ind  + eta_y)
                
            plt.xlabel(objective_names[axis_x])
            plt.ylabel(objective_names[axis_y])

            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

            marker_list = create_markers()
            if pf is not None:
                markers = marker_list[2:]
            else:
                markers = marker_list[2:-1]

            plt.legend(handles=markers,
                    loc='center left', bbox_to_anchor=(1, 0.5), handler_map={mpatches.Circle: HandlerCircle()})

            plt.savefig(save_folder_plot + objective_names[axis_x] + '_' + objective_names[axis_y] + '.png')
            plt.clf()            
    plt.close(f)

def optimal_individuals(res, save_folder):
    """Output of optimal individuals (duplicate free)"""
    problem = res.problem
    design_names = problem.design_names
    objective_names = problem.objective_names

    with open(save_folder + 'optimal_testcases.csv', 'w', encoding='UTF8', newline='') as f:
        write_to = csv.writer(f)

        header = ['Index']
        for i in range(problem.n_var):
            header.append(design_names[i])
        for i in range(problem.n_obj):
            header.append(f"Fitness_"+ objective_names[i])

        # column to indicate wheter individual is critical or not 
        header.append(f"Critical")

        write_to.writerow(header)

        clean_pop = duplicate_free(res.opt)

        for index in range(len(clean_pop)):
            row = [index]
            row.extend(["%.6f" % X_i for X_i in clean_pop.get("X")[index]])
            row.extend(["%.6f" % F_i for F_i in clean_pop.get("F")[index]])
            row.extend(["%i" % clean_pop.get("CB")[index]])
            write_to.writerow(row)
        f.close()

def all_individuals(res, save_folder):
    """Output of all evaluated individuals"""
    problem = res.problem
    hist = res.history
    design_names = problem.design_names
    objective_names = problem.objective_names

    with open(save_folder + 'all_testcases.csv', 'w', encoding='UTF8', newline='') as f:
        write_to = csv.writer(f)

        header = ['Index']
        for i in range(problem.n_var):
            header.append(design_names[i])
        for i in range(problem.n_obj):
            header.append(objective_names[i])
        # column to indicate wheter individual is critical or not 
        header.append(f"Critical")

        write_to.writerow(header)

        index = 0
        for algo in hist:
            for i in range(len(algo.pop)):
                row = [index]
                row.extend(["%.6f" % X_i for X_i in algo.pop.get("X")[i]])
                row.extend(["%.6f" % F_i for F_i in algo.pop.get("F")[i]])
                row.extend(["%i" % algo.pop.get("CB")[i]])
                write_to.writerow(row)
                index += 1
        f.close()

def all_critical_individuals(res, save_folder):
    """Output of all critical individuals"""
    problem = res.problem
    hist = res.history    # TODO check why when iterating over the algo in the history set is different
    design_names = problem.design_names
    objective_names = problem.objective_names

    all = res.obtain_all_population()
    critical = all.divide_critical_non_critical()[0]

    with open(save_folder + 'all_critical_testcases.csv', 'w', encoding='UTF8', newline='') as f:
        write_to = csv.writer(f)

        header = ['Index']
        for i in range(problem.n_var):
            header.append(design_names[i])
        for i in range(problem.n_obj):
                header.append(f"Fitness_"+ objective_names[i])
                
        write_to.writerow(header)

        index = 0
        # for algo in hist:
        for i in range(len(critical)):
            row = [index]
            row.extend(["%.6f" % X_i for X_i in critical.get("X")[i]])
            row.extend(["%.6f" % F_i for F_i in critical.get("F")[i]])
            write_to.writerow(row)
            index += 1
        f.close()

def simulations(res, save_folder):
    '''Visualization of the results of simulations'''
    ''' Plots scenarios only once when duplicates available'''

    problem = res.problem
    is_simulation = problem.is_simulation()
    if is_simulation:
        save_folder_gif = save_folder + "gif" + os.sep
        Path(save_folder_gif).mkdir(parents=True, exist_ok=True)
        clean_pop = duplicate_free(res.opt)
        for index, simout in enumerate(clean_pop.get("SO")):
            file_name = str(index) + str("_trajectory")
            param_values = clean_pop.get("X")[index]
            plotter.plot_gif(param_values, simout, save_folder_gif, file_name)
    else:
        print("No simulation visualization available. The experiment is not a simulation.")
