import prescan.api.experiment
import matlab.engine
import sys
import os 

print(os.environ['PYTHONPATH'])

eng = matlab.engine.start_matlab()
algorithmPath = "C:\\Users\\sorokin\\Documents\\Projects\\FOCETA\\matlab_algorithm"
prescanMatlabInitPath = "C:\\Program Files\\Simcenter Prescan\\Prescan_2021.3\\matlab\\MatlabInitialisation"

#sys.path.append("C:\\Program Files\\Simcenter Prescan\\Prescan_2021.3\\python")
#sys.path.append("C:\\Program Files\\MATLAB\\R2019b\\extern\\engines\\python\\build\\lib\\matlab")

eng.addpath(eng.genpath(algorithmPath))


def runExp():
    # path = "C:\\users\\sorokin\\fortiss\FOCETA - InternalDrive\models\FOCETA_PrescanModels_v1.0\Leuven_AVP_ori\Demo_AVP.pex"
    path = "C:\\Users\\sorokin\\Documents\\Projects\\FOCETA\\Experiments\\Experiment_1\\Experiment_1.pb"
    experiment = prescan.api.experiment.loadExperimentFromFile(path)

    print(experiment)

    experiment.weather.fog.enabled = True
    experiment.weather.fog.visibility = 300

    experiment.saveToFile(path)

    eng.runExperiment(experiment,10.0)



# execute matlab command

def runMatlabScript():
    res = eng.main()
    return res

def runMatlabSqrt():
    res = []
    for i in range(1,10):
        res.append(eng.sqrt(float(i)))
    return res

def runMatlabPrescanSim():
    return eng.prescanTest(nargout=0)

def createExperiment():
    return prescan.api.experiment.Experiment()

simOut = runExp()

# documentation
# e.g. 
# python3 -m pydoc prescan.api.air
#
# import prescan.api
# help (prescan.api.air)


# open matlab to edit
# eng.edit('triarea',nargout=0)