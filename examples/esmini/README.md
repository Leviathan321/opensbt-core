# Esmini Example

This example illustrated test case generation with OpenSBT using the simulator [esmini](https://github.com/esmini/esmini).

To make it run first you need to [download](https://drive.google.com/drive/folders/12vtq7LeRfa90CVjLcedv8LS0Yn14lBZg) esmini binary and place it in this folder. Then you have to update the path to the esmini executable in the config file in this folder.
Make sure that the Catalogs folder is also placed in the binary. You can find the Catalogs folder in **resources/xosc**. Update the catalogs path in the openscenario file in the corresponding scenario in the folder [/exmaples/esmini/scenarios/](/examples/esmini/scenarios/).

You can run two examples (a lane change and a cut in scenario with an LKAS controller) by running:

`python -m examples.esmini.run_lanechange`

`python -m examples.esmini.run_cutin`


