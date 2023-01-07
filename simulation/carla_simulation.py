from pathlib import Path
from typing import List, Dict
from carla_simulation import balancer
from simulation.simulator import Simulator, SimulationOutput
import logging
import json
import os

SCENARIO_DIR = "/tmp/scenarios"

class CarlaSimulator(Simulator):

    ''' Simulates a set of scenarios and returns the output '''
    @staticmethod
    def simulate(
        list_individuals,
        variable_names,
        scenario_path: str,
        sim_time: float,
        time_step:float,
        do_visualize:bool = False
    ) -> List[SimulationOutput]:
        xosc = scenario_path
        try:
            for ind in list_individuals:
                logging.info("provided following values:")
                instance_values = [v for v in zip(variable_names,ind)]
                logging.info(instance_values)
                CarlaSimulator.create_scenario_instance_xosc(xosc, dict(instance_values), outfolder=SCENARIO_DIR)
            logging.info("++ running scenarios with carla ++ ")
            outs = balancer.run_scenarios(scenario_dir=SCENARIO_DIR, visualization_flag=do_visualize)
            results = []
            for out in outs:
                simout = SimulationOutput.from_json(json.dumps(out))
                #print(f"len(simout.collisions): {len(simout.collisions)}")
                simout.otherParams["isCollision"] = (len(simout.collisions) != 0)
                results.append(simout)
        except Exception as e:
            raise e
        finally:
            logging.info("++ removing temporary scenarios ++")
            file_list = [ f for f in os.listdir(SCENARIO_DIR) if f.endswith(".xosc") ]
            for f in file_list:
                os.remove(os.path.join(SCENARIO_DIR, f))
        return results

    @staticmethod
    def create_scenario_instance_xosc(filename: str, values_dict: Dict, outfolder=None):
        Path(outfolder).mkdir(parents=True, exist_ok=True)
        parameterFlag = "$"

        print(values_dict)
        # Read in the file
        with open(filename, 'r') as file :
            filedata = file.read()

        suffix = ""
        # Replace the target string
        for k,v in values_dict.items():
            filedata = filedata.replace(parameterFlag + k, str(v))
            suffix = suffix + "_" + str(v)

        if outfolder is not None:
            if not os.path.exists(outfolder):
                os.mkdir(outfolder)
            filename = outfolder + os.sep + os.path.split(filename)[1]

        splitFilename =  os.path.splitext(filename)
        newPathPrefix = splitFilename[0]

        # Write the file out again
        newFileName  = newPathPrefix + suffix + splitFilename[1]
        with open(newFileName, 'w') as file:
            file.write(filedata)

        return newFileName