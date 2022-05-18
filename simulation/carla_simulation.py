from curses import flash
import numpy as np
from pathlib import Path
from carlaSimulation.scenario import Scenario
from utils.text_operations import createScenarioInstanceXOSC
from carlaSimulation import runScenarioStack, runSingle
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
                
            outs = runScenarioStack.run(scenario_dir=SCENARIO_DIR)

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
    
    ## Simulates one scenario and returns the output
    @staticmethod
    def simulate(individual, featureNames, xosc: str, simTime: float,samplingTime = samplingTime):
        
        instanceValues = CarlaSimulator.getParameterDict(featureNames,individual)
        scenarioInstancePath = createScenarioInstanceXOSC(xosc,instanceValues,outfolder=SCENARIO_DIR)

        out = runSingle.run(scenarioInstancePath)
        
        # remove scenarioinstance
        os.remove(scenarioInstancePath)

        simout = SimulationOutput.from_json(json.dumps(out))
        if len(simout.collisions) != 0:
            simout.otherParams["isCollision"] = True
        else:
            simout.otherParams["isCollision"] = False
        return simout
            
    @staticmethod
    def getParameterDict(featureNames, values):
        print("provided following values:")
        print(featureNames)
        print(values)
        instanceValues = {}
        i = 0
        for name in featureNames:
            instanceValues[name] = "{:.3f}".format(values[i]) # HACK temporary pruning digits
            i = i+1
        return instanceValues