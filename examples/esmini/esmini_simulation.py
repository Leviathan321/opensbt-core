from pathlib import Path
import subprocess
from typing import List, Dict
import logging as log
import json
import os
import xml.etree.ElementTree as ET
from examples.esmini.parser import EsminiParser
from opensbt.model_ga.individual import Individual
from opensbt.simulation.simulator import Simulator, SimulationOutput
import examples.esmini
from pathlib import Path

SCENARIO_DIR = os.getcwd() + "tmp" + os.sep
ESMINI_PATH = "C:\\Program Files\\esmini-bin_Windows\\esmini\\bin\\esmini"

class EsminiSimulator(Simulator):

    ''' Simulates a set of scenarios and returns the output '''
    @staticmethod
    def simulate(
        list_individuals: List[Individual],
        variable_names: List[str],
        scenario_path: str,
        sim_time: float,
        time_step: float,
        do_visualize: bool = False,
    ) -> List[SimulationOutput]:
        xosc = scenario_path

        try:
            results = []
            for ind in list_individuals:
                log.info("provided following values:")
                instance_values = [v for v in zip(variable_names,ind)]
                log.info(instance_values)
                sim_parameters = [i[0] for i in instance_values]
                if len(sim_parameters) != len(set(sim_parameters)):
                    raise Exception("Duplicate simulation varibles names, please name every variable different.")
                scenario_file = EsminiSimulator.create_scenario_instance_xosc(xosc, dict(instance_values), outfolder=SCENARIO_DIR)
                log.info("++ running scenarios with carla ++ ")

                # run esmini on scenario
                suffix = os.path.splitext(os.path.basename(scenario_file))[0]                
                output_file = f"{SCENARIO_DIR}\log_{suffix}.csv"

                Path(SCENARIO_DIR).mkdir(exist_ok=True,mode=0o777)
                
                if do_visualize:
                    flags_visualize =  ["--window",
                                "60",
                                "60",
                                "800",
                                "600"]
                else:
                    flags_visualize = []
                subprocess.run([ESMINI_PATH] + 
                               flags_visualize + 
                               [
                                "--osc",
                                scenario_file,
                                "--fixed_timestep",
                                "0.05",
                                "--csv_logger",
                                output_file,
                                "--collision"], 
                                stdout=subprocess.PIPE, 
                                stderr=subprocess.PIPE, text=True)

                out = EsminiParser.parse_log(output_file)

                # create simout instance
                simout = SimulationOutput.from_json(json.dumps(out))
                results.append(simout)

        except Exception as e:
            raise e
        finally:
            log.info("++ removing temporary scenarios ++")
            file_list = [ f for f in os.listdir(SCENARIO_DIR) if f.endswith(".xosc") ]
            for f in file_list:
                os.remove(os.path.join(SCENARIO_DIR, f))
        return results

    ''' Replace parameter values in parameter declaration section by provided parameters '''
    @staticmethod
    def create_scenario_instance_xosc(filename: str, values_dict: Dict, outfolder=None):
        xml_tree = ET.parse(filename)
        parameters = xml_tree.find('ParameterDeclarations')
        for name, value in values_dict.items():
            for parameter in parameters:
                if parameter.attrib.get("name") == name:
                    parameter.attrib["value"] = str(value)
        # # Write the file out again
        if outfolder is not None:
            Path(outfolder).mkdir(parents=True, exist_ok=True)
            filename = outfolder + os.sep + os.path.split(filename)[1]
        split_filename =  os.path.splitext(filename)
        new_path_prefix = split_filename[0]
        ending = split_filename[1]

        suffix = ""
        for k,v in values_dict.items():
            suffix = suffix + "_" + str(v)

        new_file_name  = new_path_prefix + suffix + ending
        xml_tree.write(new_file_name)

        return new_file_name
    


if __name__ == "__main__":
    scenario_path = os.getcwd() + r"\examples\esmini\scenarios\lanechange_scenario.xosc"
    results = EsminiSimulator.simulate(
                            list_individuals=[["30","10"]],
                             variable_names=[
                                 "EgoSpeed",
                                 "SimDuration"
                             ],
                             scenario_path=scenario_path,
                            do_visualize=True,
                            sim_time=10,
                            time_step=1)
