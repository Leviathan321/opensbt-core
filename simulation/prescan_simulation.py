from simulation.simulator import SimulationOutput, Simulator
import json
import logging
import time 
import sys
import os
import subprocess
from typing import List
from model_ga.individual import *
from prescan_runner import runner
from pathlib import Path
DEBUG = False
TIME_STEP = 1
DO_VISUALIZE = False
SIM_TIME = 10

OUTPUT_FILENAME = "results.csv"
TRACES_FILENAME = "trace_online.csv"
INPUT_FILENAME = "input.json"
EXP_EXECUTABLE = "Demo_AVP_cs"
PATH_KILL_SCRIPT = os.getcwd() + "\\..\\FOCETA\\experiments\\PrescanHangKill.bat"

class PrescanSimulator(Simulator):

    ''' Examplary produced input.json:
            {   
                "HostVelGain": 1.4026142683114282,   //adresses egos'velocity
                "Other":                             //adresses parameters of the other actor
                    {   
                        "Velocity_mps": 2.7790308908407315, 
                        "Time_s": 0.4617562491805899, 
                        "Accel_mpss": 1.0067168680742395
                    }
            }
    '''
    @staticmethod
    def get_ind_as_json(ind,features):
        resJson = {
            }
        for i in range(len(features)):
            actor = features[i].split('_',1)[0] 
            featureName = features[i].split('_',1)[1] 
            if actor != "Ego" :
                if actor not in resJson:
                    resJson[actor] = {}
                resJson[actor][featureName] = ind[i]
            else:
                resJson[featureName] = ind[i]

        if DEBUG:
            logging.info(resJson)
        return resJson

    @staticmethod
    def delete_traces(path):
        if Path(path).exists():
            os.remove(path)
            logging.info(f'Traces files trace_online.csv removed')

    @staticmethod
    def simulate(list_individuals: List[Individual], 
                 variable_names: List[str], 
                 scenario_path: str, 
                 sim_time :float = SIM_TIME, 
                 time_step : float = TIME_STEP,
                 do_visualize : bool = DO_VISUALIZE):
        
        parent_dir = os.path.dirname(scenario_path)
        traces_path = os.path.join("", parent_dir + os.sep + TRACES_FILENAME)
        try:
            results = []
            for ind in list_individuals:
                # Write to input.json individual
                json_input = PrescanSimulator.get_ind_as_json(ind, variable_names) 
                with open(parent_dir + os.sep + INPUT_FILENAME, "w") as outfile:
                    outfile.write(json.dumps(json_input))  
                logging.info(f"++ Prescan Experiment Created for {ind} ++")
                logging.info("++ Running scenario with Prescan ++ ")
                
                start_time_simulation = time.time()

                ouput_runner =  runner.run_scenario(input_json_name = INPUT_FILENAME,
                                             exp_file = scenario_path,
                                             name_executable = EXP_EXECUTABLE,
                                             sim_time=sim_time,
                                             do_visualize=do_visualize,
                                             output_filename= OUTPUT_FILENAME,
                                             traces_filename= TRACES_FILENAME)
                simout = SimulationOutput.from_json(json.dumps(ouput_runner))

                end_time_simulation = time.time()

                logging.info(f"Simulation Time is: {end_time_simulation - start_time_simulation}")
                results.append(simout)  
                
                if DEBUG:
                    check_if_continue_by_user()
     
                # delete file where traces are stored from simulation
                PrescanSimulator.delete_traces(traces_path)
        except Exception as e:
            raise e
        finally:
            PrescanSimulator.delete_traces(traces_path)
        return results  
    
    @staticmethod    
    def kill():
        import subprocess
        filepath=PATH_KILL_SCRIPT
        p = subprocess.Popen(filepath, shell=True, stdout = subprocess.PIPE)

def check_if_continue_by_user():
    doContinue = input("Continue search? press Y for yes, else N.")
    logging.info(f"++ Input was {doContinue}")
    if doContinue == 'N' or doContinue == 'n':
        logging.info("Terminating search after user input")
        sys.exit()
    else:
        logging.info("Continuing search")       
        