import numpy as np
from utils.text_operations import createScenarioInstanceXOSC
from carlaSimulation import runner
from simulation.simulator import SimulationOutput
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

            # TODO consider output from carla stored in dictionary or class

            for out in outs:
                # TODO get real metrics calculation
                # TODO decide to do evaluation in optimizer oder directly by carla
                # put here the recording results of carla, or the postprocessed results
                # use dummy values for now
                otherParams = {}

                otherParams["samples"] = out[0]
                otherParams["positionEgo"] = out[1]
                otherParams["positionAdv"] = out[2]
                otherParams["velocityEgo"] = out[3]
                otherParams["velocitiyAdv"] = out[4]
                otherParams["distanceEgo"] = out[5]

                steps =  len(otherParams["samples"])

                #collisions = out[6]

                # if otherParams["collisions"] != []:
                #     otherParams["isCollision"] = True
                # else:
                #     otherParams["isCollision"] = False

                # TODO use real values

                otherParams["isCollision"] = True
                egoTrajectory = np.ones((4,steps))
                objectTrajectory = np.ones((4,steps))

                simout = SimulationOutput(simTime,egoTrajectory,objectTrajectory,otherParams=otherParams)
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
            instanceValues[name] = "{:.3f}".format(values[i]) # HACK temporary pruning digits
            i = i+1
        return instanceValues