# Esmini Example

This example illustrated test case generation with OpenSBT using the simulator esmini.

To make it run first you need to download esmini binary. Then you have to update the path to the esmini executable in the config file in this folder.
Make sure that the Catalogs folder is also placed in the binary. You can find the Catalogs folder in resources/xosc. Update the catalogs path in the openscenario file in scenario of this folder.

You can run two examples (a lane change and a cut in scenario with an LKAS controller) by running:

`python -m examples.esmini.run_lanechange`

`python -m examples.esmini.run_cutin`


