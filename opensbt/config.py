import os
import numpy as np

# core framework
DEBUG = False
SHOW_PLOT = False
RESULTS_FOLDER = os.sep + "results" + os.sep
WRITE_ALL_INDIVIDUALS = True
LOG_FILE = "." + os.sep + "log.txt"
BACKUP_FOLDER = "backup"
EXPERIMENTAL_MODE = True

# analysis module
N_CELLS = 10
CONSIDER_HIGH_VAL_OS_PLOT = False
PENALTY_MAX = 1000
PENALTY_MIN = -1000 
WRITE_ALL_INDIVIDUALS = True
LAST_ITERATION_ONLY_DEFAULT = True
LAST_ITERATION_ONLY = True
METRIC_PLOTS_FOLDER = "metrics" + os.sep
COVERAGE_METRIC_NAME = "CID"
LOAD_FROM_GENERATIONS = True

metric_config = {}
metric_config["DUMMY"] = dict(
    ref_point_hv = np.asarray([20,0]),
    ideal = np.asarray([0,-20])
)
