from pathlib import Path
from typing import List, Dict
import logging as log
import json
import os
import xml.etree.ElementTree as ET
from opensbt.model_ga.individual import Individual
from opensbt.simulation.simulator import Simulator, SimulationOutput

try:
    import yaml
    from carla_simulation.balancer import Balancer
except Exception:
    log.info("Carla Simulation adapter could not have been imported.")

SCENARIO_DIR = "/tmp/scenarios"
FAULT_DIR = "/tmp/faults"

class CarlaSimulator(Simulator):

    _balancer = None

    ''' Simulates a set of scenarios and returns the output '''
    @staticmethod
    def simulate(
        list_individuals: List[Individual],
        variable_names: List[str],
        scenario_path: str,
        sim_time: float,
        time_step: float,
        do_visualize: bool = False,
        fault: str = None,
        terminate: bool = False
    ) -> List[SimulationOutput]:
        xosc = scenario_path

        try:

            for ind in list_individuals:
                log.info("provided following values:")
                instance_values = [v for v in zip(variable_names,ind)]
                log.info(instance_values)
                sim_parameters = [i[0] for i in instance_values]
                if len(sim_parameters) != len(set(sim_parameters)):
                    raise Exception("Duplicate simulation varibles names, please name every variable different.")
                pattern = CarlaSimulator.create_scenario_instance_xosc(xosc, dict(instance_values), outfolder=SCENARIO_DIR)
                if fault is not None:
                    CarlaSimulator.create_fault_instance_yaml(fault, dict(instance_values),pattern)


            log.info("++ running scenarios with carla ++ ")

            if CarlaSimulator._balancer is None:
                CarlaSimulator._balancer = Balancer(
                    scenarios_dir = SCENARIO_DIR,
                    jobs = 1,
                    visualization = do_visualize,
                    agent = 'FMIAgent',
                    faults_dir = FAULT_DIR
                )
                CarlaSimulator._balancer.start()

            outs = CarlaSimulator._balancer.run()

            results = []
            for out in outs:
                simout = SimulationOutput.from_json(json.dumps(out))
                results.append(simout)

        except Exception as e:

            raise e

        finally:
            if terminate:
                CarlaSimulator._balancer.stop()

            log.info("++ removing temporary scenarios ++")

            file_list = [ f for f in os.listdir(SCENARIO_DIR) if f.endswith(".xosc") ]
            for f in file_list:
                os.remove(os.path.join(SCENARIO_DIR, f))
                if fault is not None:
                    os.remove(os.path.join(FAULT_DIR,f))

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


    @staticmethod
    def create_fault_instance_yaml(filename: str, values_dict: Dict, pattern: str):
        with open(filename, 'r') as stream:
            try:
                loaded = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        if "starttime" not in loaded['faultInjection']:
            raise Exception("Starttime parameter missing in fault yaml.")
        if "starttime" not in values_dict.keys() and loaded['faultInjection'].get('starttime') is None:
            raise Exception("Starttime needs to be a simulation variable or its value has to be set in the fault.yaml")
        for name, value in values_dict.items():
            # starttime & endtime
            if(name in loaded['faultInjection']):
                loaded['faultInjection'][name] = str(value)
            # other fault specific parameters, e.g. drift rate
            if name in loaded['faultInjection']['parameters']:
                json_string = (loaded['faultInjection']['parameters']).encode().decode('unicode_escape')
                json_dict = json.loads(json_string)
                json_dict[name] = value
                loaded['faultInjection']['parameters'] = json.dumps(json_dict)

        filename = os.path.split(pattern)
        path = FAULT_DIR + "/" + filename[1]
        if(os.path.isdir(FAULT_DIR + "/")) is False:
            os.mkdir(FAULT_DIR + "/")
        f = open(path, "w")
        f.write(yaml.safe_dump(loaded))
        f.close()

        return path
