from genericpath import isdir
from tokenize import String
from typing import Dict
import os

def createScenarioInstanceXOSC(filename: String, dict: Dict, outfolder=None):
    parameterFlag = "$"

    # Read in the file
    with open(filename, 'r') as file :
        filedata = file.read()
    
    suffix = ""
    # Replace the target string
    for k,v in dict.items():
        filedata = filedata.replace(parameterFlag + k,v)
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


# test
# instanceValues = {
#     "EgoStartS" : "2",
#     "EgoLaneChangeStart": "10" ,
#     "EgoTargetSpeed": "90"
# }

# path ="../scenarios/2-lanechange-ego-left_carla_generic.xosc"
# outfolder = os.getcwd() + os.sep + "temp"
# createScenarioInstanceXOSC(path,instanceValues,outfolder)