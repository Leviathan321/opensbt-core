import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import opensbt.algorithm.classification.decision_tree.decision_tree as decision_tree
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerPatch
from opensbt.visualization import scenario_plotter
from pymoo.indicators.igd import IGD
from pymoo.indicators.hv import Hypervolume
from pymoo.core.population import Population
from opensbt.visualization.configuration import *
from opensbt.utils.sorting import *
from opensbt.algorithm.classification.classifier import ClassificationType
from opensbt.analysis.quality_indicators.quality import Quality
from opensbt.model_ga.problem import *
from opensbt.model_ga.result import *
from typing import Dict
from opensbt.utils.duplicates import duplicate_free
import logging as log

from opensbt.config import BACKUP_FOLDER, CONSIDER_HIGH_VAL_OS_PLOT,  \
                            PENALTY_MAX, PENALTY_MIN, WRITE_ALL_INDIVIDUALS, \
                                METRIC_PLOTS_FOLDER, LAST_ITERATION_ONLY_DEFAULT, COVERAGE_METRIC_NAME, \
                                N_CELLS

def cid_analysis_hitherto(res: Result, save_folder: str, reference_set=None, n_evals_by_axis=None):
    log.info("------ Performing CID analysis ------")
    save_folder_plot =  save_folder + METRIC_PLOTS_FOLDER
    Path(save_folder_plot).mkdir(parents=True, exist_ok=True)

    eval_result = Quality.calculate_cid(res, reference_set=reference_set, n_evals_by_axis=n_evals_by_axis)

    if eval_result is None:
        log.info("No IDGE values computed")
        return
    
    n_evals, cid = eval_result.steps, eval_result.values
    
    # store
    eval_result.persist(save_folder + BACKUP_FOLDER)
    write_metric_history(n_evals, cid, 'cid',save_folder)

    f = plt.figure()
    plt.plot(n_evals, cid, color='black', lw=0.7)
    plt.scatter(n_evals, cid, facecolor="none", edgecolor='black', marker="o")
    plt.title("Coverage Analysis")
    plt.xlabel("Function Evaluations")
    plt.ylabel(COVERAGE_METRIC_NAME)
    plt.savefig(save_folder_plot + COVERAGE_METRIC_NAME.lower() + '_global.png')
    plt.clf()
    plt.close(f)

    # output to console
    log.info(f"Final {COVERAGE_METRIC_NAME} value: {cid[-1]}")

def calculate_n_crit_distinct(res: Result, 
                             save_folder: str, 
                             bound_min = None, 
                             bound_max = None, 
                             n_cells=N_CELLS,
                             optimal=False,
                             var = "F"):

    log.info("------ Performing number critical analysis ------")
    log.info(f"------ Optimal: {optimal}------")
    log.info(f"------ N_cells: {n_cells}------")

    save_folder_plot =  save_folder + METRIC_PLOTS_FOLDER
    Path(save_folder_plot).mkdir(parents=True, exist_ok=True)

    eval_result = Quality.calculate_n_crit_distinct(res, 
                                            bound_min, 
                                            bound_max, 
                                            n_cells=n_cells,
                                            optimal=optimal,
                                            var = var
                                            )

    if eval_result is None:
        log.info("No number distinct criticals computed")
        return
    
    n_evals, cid = eval_result.steps, eval_result.values
    
    # store
    eval_result.persist(save_folder + BACKUP_FOLDER)
    write_metric_history(n_evals, cid, f"n_crit{'_opt' if optimal else ''}_{var}",save_folder)

    f = plt.figure()
    plt.plot(n_evals, cid, color='black', lw=0.7)
    plt.scatter(n_evals, cid, facecolor="none", edgecolor='black', marker="o")
    plt.title("Failure Analysis")
    plt.xlabel("Function Evaluations")
    plt.ylabel(f"Number Critical (Cell Size = {n_cells})")
    plt.savefig(save_folder_plot + f'n_crit{"_opt" if optimal else ""}_{var}.png')
    plt.close()
    plt.clf()
    plt.close(f)

    # output to console
    log.info(f"Final n_critical value: {cid[-1]}")

def gd_analysis(res: Result, save_folder: str, input_pf=None, filename='gd', mode='default', critical_only = False):
    log.info("------ Performing gd analysis ------")
    save_folder_plot =  save_folder + METRIC_PLOTS_FOLDER
    Path(save_folder_plot).mkdir(parents=True, exist_ok=True)

    eval_result = Quality.calculate_gd(res, input_pf=input_pf, critical_only=critical_only, mode=mode)
    if eval_result is None:
        log.info("No GD values computed")
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
    log.info(f"Final GD value: {gd[-1]}")


def gd_analysis_hitherto(res: Result, save_folder: str, input_pf=None, filename='gd_global', mode='default'):
    log.info("------ Performing igd analysis ------")
    save_folder_plot =  save_folder + METRIC_PLOTS_FOLDER
    Path(save_folder_plot).mkdir(parents=True, exist_ok=True)

    eval_result = Quality.calculate_gd_hitherto(res, input_pf=input_pf, mode=mode)
    if eval_result is None:
        log.info("Eval result is none.")
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
    plt.savefig(save_folder_plot + filename + '_' + mode + '.png')
    plt.clf()
    plt.close(f)

def igd_analysis(res: Result, save_folder: str, critical_only = False, input_pf=None, filename='igd'):
    # log.info("------ Performing igd analysis ------")
    save_folder_plot =  save_folder + METRIC_PLOTS_FOLDER
    Path(save_folder_plot).mkdir(parents=True, exist_ok=True)

    eval_result = Quality.calculate_igd(res, critical_only, input_pf=input_pf)
    if eval_result is None:
        log.info("Eval result is none.")
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
    plt.savefig(save_folder_plot + filename + '.png')
    plt.clf()
    plt.close(f)

    # output to console
    log.info(f"Final IGD value: {igd[-1]}")

def igd_analysis_hitherto(res: Result, save_folder: str, input_pf=None, filename='igd_global'):
    # log.info("------ Performing igd analysis ------")
    save_folder_plot =  save_folder + METRIC_PLOTS_FOLDER
    Path(save_folder_plot).mkdir(parents=True, exist_ok=True)

    eval_result = Quality.calculate_igd_hitherto(res, input_pf=input_pf)
    if eval_result is None:
        log.info("Eval result is none.")
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
    plt.savefig(save_folder_plot + filename + '.png')
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

def hypervolume_analysis(res, save_folder, critical_only=False, ref_point_hv=None, ideal=None, nadir=None):
    # log.info("------ Performing hv analysis ------")
    save_folder_plot =  save_folder + METRIC_PLOTS_FOLDER
    Path(save_folder_plot).mkdir(parents=True, exist_ok=True)

    eval_result = Quality.calculate_hv_hitherto(res, critical_only, ref_point_hv, ideal, nadir)
 
    if eval_result is None:
        log.info("Eval result is none.")
        return
    
    n_evals, hv = eval_result.steps, eval_result.values

    # store
    eval_result.persist(save_folder + BACKUP_FOLDER)
    write_metric_history(n_evals, hv, 'hv_all', save_folder)

    # plot
    f = plt.figure(figsize=(7, 5))
    plt.plot(n_evals, hv, color='black', lw=0.7)
    plt.scatter(n_evals, hv, facecolor="none", edgecolor='black', marker='o')
    plt.title("Performance Analysis")
    plt.xlabel("Function Evaluations")
    plt.ylabel("Hypervolume")
    plt.savefig(save_folder_plot + 'hypervolume.png')
    plt.clf()
    plt.close(f)

    # output to console
    log.info(f"Final HV value: {hv[-1]}")

def hypervolume_analysis_local(res, save_folder):
    log.info("------ Performing hv analysis ------")
    
    save_folder_plot =  save_folder + METRIC_PLOTS_FOLDER
    Path(save_folder_plot).mkdir(parents=True, exist_ok=True)

    eval_result = Quality.calculate_hv(res)

    if eval_result is None:
        log.info("Eval result is none.")
        return
    

    n_evals, hv = eval_result.steps, eval_result.values

    # store
    eval_result.persist(save_folder + BACKUP_FOLDER)
    write_metric_history(n_evals, hv,'hv_local_all',save_folder)    

    # plot
    f = plt.figure(figsize=(7, 5))
    plt.plot(n_evals, hv, color='black', lw=0.7)
    plt.scatter(n_evals, hv, facecolor="none", edgecolor='black', marker='o')
    plt.title("Performance Analysis")
    plt.xlabel("Function Evaluations")
    plt.ylabel("Hypervolume")
    plt.savefig(save_folder_plot + 'hypervolume_local.png')
    plt.clf()
    plt.close(f)

def si_analysis(res, 
                save_folder,
                input_pf,
                critical_only=False,
                ideal=None,
                nadir=None):
    # log.info("------ Performing sp analysis ------")
    save_folder_plot =  save_folder + METRIC_PLOTS_FOLDER
    Path(save_folder_plot).mkdir(parents=True, exist_ok=True)

    eval_result = Quality.calculate_si(res,
                                       input_pf=input_pf,
                                       critical_only=critical_only,
                                       ideal=None,
                                       nadir=None)
    if eval_result is None:
        log.info("Eval result is none.")
        return
    
    n_evals, uniformity = eval_result.steps, eval_result.values
    
    # store
    eval_result.persist(save_folder + BACKUP_FOLDER)
    write_metric_history(n_evals,uniformity,'si',save_folder)

    # plot
    f = plt.figure(figsize=(7, 5))
    plt.plot(n_evals, uniformity, color='black', lw=0.7)
    plt.scatter(n_evals, uniformity, facecolor="none", edgecolor='black', marker='o')
    plt.title("Uniformity Analysis")
    plt.xlabel("Function Evaluations")
    plt.ylabel("SI")
    plt.savefig(save_folder_plot + 'si.png')
    plt.clf()
    plt.close(f)
    # output to console
    log.info(f"Final SI value: {uniformity[-1]}")

def spread_analysis(res, save_folder, critical_only=False):
    # log.info("------ Performing sp analysis ------")
    save_folder_plot =  save_folder + METRIC_PLOTS_FOLDER
    Path(save_folder_plot).mkdir(parents=True, exist_ok=True)

    eval_result = Quality.calculate_sp(res, critical_only=critical_only)
    if eval_result is None:
        log.info("Eval result is none.")
        return
    
    n_evals, uniformity = eval_result.steps, eval_result.values
    
    # store
    eval_result.persist(save_folder + BACKUP_FOLDER)
    write_metric_history(n_evals,uniformity,'sp',save_folder)

    # plot
    f = plt.figure(figsize=(7, 5))
    plt.plot(n_evals, uniformity, color='black', lw=0.7)
    plt.scatter(n_evals, uniformity, facecolor="none", edgecolor='black', marker='o')
    plt.title("Spreadness/Uniformity Analysis")
    plt.xlabel("Function Evaluations")
    plt.ylabel("SP")
    plt.savefig(save_folder_plot + 'spread.png')
    plt.clf()
    plt.close(f)

    # output to console
    log.info(f"Final SP value: {uniformity[-1]}")


def spread_analysis_hitherto(res, save_folder, hitherto = False):
    log.info("------ Performing sp analysis ------")
    save_folder_plot =  save_folder + METRIC_PLOTS_FOLDER
    Path(save_folder_plot).mkdir(parents=True, exist_ok=True)

    eval_result = Quality.calculate_sp_hitherto(res)
    if eval_result is None:
        log.info("Evalt result is none.")
        return
    
    n_evals, uniformity = eval_result.steps, eval_result.values
    
    # store
    eval_result.persist(save_folder + BACKUP_FOLDER)
    write_metric_history(n_evals,uniformity,'sp_global',save_folder)

    # plot
    f  = plt.figure(figsize=(7, 5))
    plt.plot(n_evals, uniformity, color='black', lw=0.7)
    plt.scatter(n_evals, uniformity, facecolor="none", edgecolor='black', marker='o')
    plt.title("Spreadness/Uniformity Analysis")
    plt.xlabel("Function Evaluations")
    plt.ylabel("SP")
    plt.clf()
    plt.close(f)
    
    plt.savefig(save_folder_plot  + 'spread_global.png')
