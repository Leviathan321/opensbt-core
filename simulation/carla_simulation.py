import numpy as np
from carlaSimulation import runner
from utils.text_operations import createScenarioInstanceXOSC
from simulation.simulator import SimulationOutput
import json
import os

SCENARIO_DIR = str(os.getcwd()) + os.sep + "carlaSimulation" + os.sep + "temp"

class CarlaSimulator(object):
    # TODO consider sampling time; check how to set sampling time via matlab
    samplingTime = 1

    ## Simulates a set of scenarios and returns the output
    @staticmethod
    def simulateBatch(listIndividuals, featureNames, xosc: str, simTime: float,samplingTime = samplingTime):
        try:
            for ind in listIndividuals:
                instanceValues = CarlaSimulator.getParameterDict(featureNames,ind)
                createScenarioInstanceXOSC(xosc,instanceValues,outfolder=SCENARIO_DIR)

            print("++ running scenarios with carla ++ ")

            outs = runner.run_scenarios(scenario_dir=SCENARIO_DIR)
            results = []
            
            for out in outs:
                # TODO decide to do evaluation in optimizer oder directly by carla 
                #print(f"Simulation output: {out}")

                simout = SimulationOutput.from_json(json.dumps(out))
                if len(simout.collisions) != 0:
                    simout.otherParams["isCollision"] = True
                else:
                    simout.otherParams["isCollision"] = False
                results.append(simout)
        except Exception as e:
            raise e
        finally:
            print("++ removing temporary scenarios ++")
            filelist = [ f for f in os.listdir(SCENARIO_DIR) if f.endswith(".xosc") ]
            for f in filelist:
                os.remove(os.path.join(SCENARIO_DIR, f))
        return results
    
    @staticmethod
    def getParameterDict(featureNames, values):
        print("provided following values:")
        print(featureNames)
        print(values)
        instanceValues = {}
        i = 0
        for name in featureNames:
            instanceValues[name] = "{:.3f}".format(values[i]) # HACK temporary pruning digits TODO prune digits after individuum generation
            i = i+1
        return instanceValues