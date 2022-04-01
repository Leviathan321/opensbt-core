import prescan.api.experiment
import matlab.engine

libPath = "C:\\Users\\sorokin\Documents\\Projects\\FOCETA\\matlab_algorithm\\generic"

def runExp():
    # path = "C:\\users\\sorokin\\fortiss\FOCETA - InternalDrive\models\FOCETA_PrescanModels_v1.0\Leuven_AVP_ori\Demo_AVP.pex"
    path = "C:\\Users\\sorokin\\Documents\\Projects\\FOCETA\\Experiments\\Experiment_3\\child_at_road.pb"
    eng.addpath(eng.genpath(libPath))
    eng.addpath(path)
    experiment = prescan.api.experiment.loadExperimentFromFile(path)
    # print(experiment)
    
    # modify
    experiment.weather.fog.enabled = True
    experiment.weather.fog.visibility = 300
    experiment.saveToFile(path)

    simTime = 30

    # run experiment
    simOutput = eng.runExperimentFromPb(path,'on', simTime, nargout=0)

    return simOutput

sessions = matlab.engine.find_matlab()

if sessions != ():
    print(str(sessions[0]))
    print("++ connecting to session " + str(sessions[0]))
    
    eng = matlab.engine.connect_matlab(sessions[0])
    eng.addpath(libPath)

    print(eng.workspace)

    simOut = runExp()

    # save in workspace variable
    eng.workspace['simOut'] = simOut

    # check output values
    print(eng.eval('simOut.tout',nargout=1))
    print(eng.eval('simOut.state',nargout=1))
    print(eng.eval('simOut.SimulationMetadata',nargout=1))
else:
    print("++ no matlab session is shared")