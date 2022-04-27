from tokenize import String
from typing import Dict
import os

def substitute(filename: String, dict: Dict):
    # Read in the file
    with open(filename, 'r') as file :
        filedata = file.read()

    suffix = ""
    # Replace the target string
    for k,v in dict.items():
        filedata = filedata.replace(k,v)
        suffix = suffix + "_" + str(v)

    # Write the file out again
    splitFilename =  os.path.splitext(filename)
    newFilename = splitFilename[0] + suffix + "." + splitFilename[1]
    with open(newFilename, 'w') as file:
        file.write(filedata)

    return newFilename

# test
instanceValues = {
    "$EgoStartS" : "2",
    "$EgoLaneChangeStart": "10" ,
    "$EgoTargetSpeed": "90"
}

path ="..\\scenarios\\2-lanechange-ego-left_carla_generic.xosc"

substitute(path,instanceValues)