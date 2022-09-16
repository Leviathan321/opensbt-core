from ast import parse
from pickle import TRUE
import matlab.engine
import numpy as np
from prescanSimulation.matutils.matlib import *
from simulation.simulator import SimulationOutput
import json
from  prescanSimulation.parser import parser_compiled_csv 
from  prescanSimulation.parser import parser_workspace

import time 
import sys
import os
import subprocess

DEBUG = True
OUTPUT_FILENAME = "results.csv"
TRACES_FILENAME = "trace_online.csv"
INPUT_FILENAME = "input.json"

class PrescanSimulator(object):
    # TODO consider sampling time; check how to set sampling time via matlab
    samplingTime = 1

    @staticmethod
    def simulateOSC(var,filepath: str, simTime: float):
        # TODO
        pass

    @staticmethod
    def runExp(eng, expPath: str, simTime: float, doRegenerate=False):
        simOutput = eng.runExperimentFromPb(expPath,doRegenerate, simTime, nargout=1)
        return simOutput

    @staticmethod
    def runCompiledExp(expPath: str, expName):
        path = expPath + os.sep + expName + ".exe"
        print("calling " + path)
        subprocess.run(expName + ".exe", cwd=expPath, shell=True)
        
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
    def getIndividualAsJson(ind,features):
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
            print(resJson)
        return resJson

    @staticmethod
    def delete_traces(path):
        print(f"traces path: {path}")
        if Path(path).exists():
            os.remove(path)
            print(f'Traces files trace_online.csv removed')

    @staticmethod
    def simulateBatch_compiled_csv(listIndividuals, featureNames, xosc: str, simTime: float,samplingTime = samplingTime):
        filepath = xosc
        try:
            # Connect to matlab and add matlab libs to path
            eng = connectToPrescanMatlab()     
            p = Path(filepath)
            parentDir = str(p.parent)
            cd(eng,parentDir)
            addPath(eng,filepath,recursive=True)
            results = []
            print("++ Matlab libraries added") 

            for ind in listIndividuals:
                # Write to input.json individual
                jsonInput = PrescanSimulator.getIndividualAsJson(ind, featureNames) 

                with open(str(p.parent) + os.sep + INPUT_FILENAME, "w") as outfile:
                    outfile.write(json.dumps(jsonInput))  
                print("++ Input.json created") 
                print(f"++ Prescan Experiment Created for {ind} ++")

                # Modify Experiment by running matlab script named 'ChangeModel' that reads from input.json  
                eng.ChangeModel(nargout=0) 
     
                # Run scenario
                print("++ Running scenario with Prescan ++ ")
                
                start_time_simulation = time.time()


                PrescanSimulator.runCompiledExp(str(parentDir),"Demo_AVP_cs")
                end_time_simulation = time.time()
                
                print(f"++ Simulation time of scenario is {end_time_simulation - start_time_simulation} s")
    
                # parse simulation output
                parsed = parser_compiled_csv.parseOutput(
                    parentDir + os.sep + OUTPUT_FILENAME, 
                    parentDir + os.sep + TRACES_FILENAME
                )
                simout = SimulationOutput.from_json(json.dumps(parsed))

                results.append(simout)  
                
         
                if DEBUG:
                    doContinue = input("Continue search? press Y for yes, else N.")
                    print(f"++ Input was {doContinue}")
                    if doContinue == 'N' or doContinue == 'n':
                        print("Terminating search after user input")
                        sys.exit()
                    else:
                        print("Continuing search")       
                        
                # delete file where traces are stored from simulation
                traces_path = os.path.join("",parentDir + os.sep + TRACES_FILENAME)
                PrescanSimulator.delete_traces(traces_path)
        except Exception as e:
            raise e
        finally:
            PrescanSimulator.delete_traces(traces_path)
        return results  


     ## Simulates a set of scenarios and return the simulation output after reading it from the matlab workspace
    @staticmethod
    def simulateBatch_workspace_output(listIndividuals, featureNames, xosc: str, simTime: float,samplingTime = samplingTime):
        filepath = xosc
        try:
            eng = connectToPrescanMatlab()     
            p = Path(filepath)
            parentDir = str(p.parent)
            cd(eng,parentDir)
            addPath(eng,filepath,recursive=True)
            results = []

            for ind in listIndividuals:
                # Write to input.json individual
                jsonInput = PrescanSimulator.getIndividualAsJson(ind, featureNames) 

                with open(str(p.parent) + os.sep + INPUT_FILENAME, "w") as outfile:
                    outfile.write(json.dumps(jsonInput))  
                print("Input.json created") 
                print(f"++ Prescan Experiment Created for {ind} ++")

                # Modify Experiment by running script that reads from input.json  
                eng.ChangeModel(nargout=0) 

                print("++ Running scenario with Prescan ++ ")
     
                PrescanSimulator.runExp(eng,filepath,simTime)
                
                out = eng.workspace["trace_container"]
                
                # parsed = parser_workspace.parseOutput(eng,out)

                # simout = SimulationOutput.from_json(json.dumps(parsed))
                # results.append(simout)
        except Exception as e:
            raise e
        finally:
            print("++ Removing temporary scenarios ++")
            # for f in experiments:
            #     os.remove(os.path.join("",f))
        return results  
                
     ## Simulates a set of scenarios and returns the output for a compiled experiment
    @staticmethod
    def simulateBatch_compiled(listIndividuals, featureNames, xosc: str, simTime: float,samplingTime = samplingTime):
        filepath = xosc
        experiments = []
        try:
            eng = connectToPrescanMatlab()     
            p = Path(filepath)
            parentDir = str(p.parent)
            cd(eng,parentDir)
            addPath(eng,filepath,recursive=True)
            results = []

            for ind in listIndividuals:
                # Write to input.json individual
                jsonInput = PrescanSimulator.getIndividualAsJson(ind, featureNames) 

                with open(str(p.parent) + os.sep + INPUT_FILENAME, "w") as outfile:
                    outfile.write(json.dumps(jsonInput))  
                print("Input.json created") 
                print(f"++ Prescan Experiment Created for {ind} ++")

                # Modify Experiment by running script that reads from input.json  
                expPath = eng.ChangeModel(nargout=0) 

                experiments.append(expPath)
           
                print("++ Running scenario with Prescan ++ ")
     
                # run scenario
                print(f"++ Running Prescan Experiment {filepath}... ++")
                PrescanSimulator.runCompiledExp(str(parentDir),"Demo_AVP_cs")

                parsed = parser_compiled_csv.parseOutput(parentDir + os.sep + OUTPUT_FILENAME, parentDir + os.sep + TRACES_FILENAME)
                simout = SimulationOutput.from_json(json.dumps(parsed))
                results.append(simout)
        except Exception as e:
            raise e
        finally:
            print("++ Removing temporary scenarios ++")
            # for f in experiments:
            #     os.remove(os.path.join("",f))
        return results  

    ## Simulates a set of scenarios and returns the output
    @staticmethod
    def simulateBatch(listIndividuals, featureNames, xosc: str, simTime: float,samplingTime = samplingTime):
        filepath = xosc
        experiments = []
        try:
            eng = connectToPrescanMatlab()     
            p = Path(filepath)
            cd(eng,str(p.parent))
            addPath(eng,filepath,recursive=True)
            
            for ind in listIndividuals:
                # TODO create scenario instances
                # TODO check how to dynamically instantiate scenarion without creating experiments every time 

                # Uncomment to create example experiment        
                expPath = eng.createExperiment(*ind)
                experiments.append(expPath)
                print(f"++ Prescan Experiment Created for {ind} ++")
            # DUMMY
            # experiments = [filepath] * len(listIndividuals)
            print("++ Running scenarios with Prescan ++ ")
            
            results = []
            for filepath in experiments:        
                # run scenario
                print(f"++ Running Prescan Experimen {filepath}... ++")
                #output = eng.runExperiment(filepath)
                output = PrescanSimulator.runExp(eng,filepath,simTime, doRegenerate=False)
                parsed = parseOutput(eng,output)
                simout = SimulationOutput.from_json(json.dumps(parsed))
                results.append(simout)
        except Exception as e:
            raise e
        finally:
            print("++ Removing temporary scenarios ++")
            for f in experiments:
                os.remove(os.path.join("",f))
        return results