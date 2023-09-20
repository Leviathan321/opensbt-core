# OpenSBT - A Modular Framework for Search-based Testing of Automated Driving Systems


## Intro

OpenSBT provides a modular and extandable code base for the search-based testing of automated driving systems. It provides interfaces to integrate search algorithms, fitness/criticality functions and simulation environments in a modular way. That means, one of this components can be replaced by a another component without the need to change the connectivity to other components or adapt the result analysis/visualization. Further it visualizes the test outcome and analysis the critical behaviour of the ADS. 

A demo video of OpenSBT can be found here: https://www.youtube.com/watch?v=oOyug8rwAB8.


## Architecture

[<img src="doc/figures/OpenSBT_architecture.png" width="500"/>]()

OpenSBT builds upon [Pymoo](https://pymoo.org/). It extends internal models as Individual, Result to apply SBT of ADS.
Further it provides three interfaces/abstractions to integrate 
SBT component in a modular way.
OpenSBT provides already extensions for the simulation of test cases in the [Prescan](https://git.fortiss.org/opensbt/prescan_runner) and [CARLA Simulator](https://git.fortiss.org/opensbt/carla-runner). 

In this branch is an example of SBT using the CARLA Simulator provided. An examplary integration of PRESCAN with the Prescan runner is provided here:
https://git.fortiss.org/opensbt/opensbt-core/-/tree/ASE_prescan_experiment.

## Installation

OpenSBT is available as a [PIP-package](TODO). OpenSBT requires python to be installed. Compatibility has been tested with python 3.7 and 3.8. 
If OpenSBT is downloaded directly from this repo, it is recommented to create a virtual environment and install all dependencies in this environment by executing:

```bash
bash install.sh
```

*Note: For windows use as the second command 'source venv/Scripts/activate' instead.*

## Getting Started
After installing OpenSBT, you can try following examples which show how to use OpenSBT for testing. We have also provided a virtual machine
where you can execute these example (TODO).

- CARLA PID Agent Testing in CARLA Simulator 
- AEB Agent Testing in CARLA Simulator
- Simulink-based Agent Testing in PreScan Simulator (Can we call it example if we cannot share code)
- Simple Agent Testing in Dummy Simulator (Check whether this is the right place)

Also we have provided [jupyter notebooks](doc/jupyter/) which explain step-by-step 
a) (How to run the implemented testing examples)[]
b) (How to integrate/implement a new testing algorithm in OpenSBT)[]
c) (How to define a testing experiment)[]

OpenSBT can be run as a standalone application or can be imported as a library. To import it as a library you just need to install the correspnding pip package. To use it standalone you need to run `python run.py` with different flags (TODO) from the OpenSBTs main folder.

This will invoke experiments which are registered in OpenSBT. The benefit of using the standalone mode is that we can by setting flags vary search parameters, the search space, the algorithm or the experiment without modifying the python code.

In the following we describe how to execute the examples implemented in OpenSBT using the standalone mode.

### PID Agent (CARLA)

1. Goto CarlaRunner repo, follow the instructions to install the CarlaRunner 
(download the moduls, build, pip install, adjust paths should be fine)

2. Run the experiment Y in default experiments (need to check whether we put the experiments in an extra folder, since this should not be part of the "basic" opensbt files)

3. Observe that output files are generated

### Rover Agent (CARLA)

1. Goto CarlaRunner repo, follow the instructions to install the CarlaRunner 
(download the moduls, build, pip install, adjust paths should be fine)

(+ *in CarlaSimulation the agent flag should be set.@Tiziano: can we pass this directly when we define the experiment*)

2. Run the experiment X in default experiments (need to check whether we put the experiments in an extra folder, since this should not be part of the "basic" opensbt files)

3. Observe that output files are generated

### Simulink-based Agent (PreScan) 
(*We can only provide a demo, if we an share some example Prescan experiment?*)

1. Goto PrescanRunner repo, follow the instructions to install the PrescanRunner 
(download the moduls, build, pip install)

2. Run the experiment X in default experiments (need to check whether we put the experiments in an extra folder, since this should not be part of the "basic" opensbt files)

3. Observe that output files are generated


## Usage


### Optional Parameters

All flags that can be set are (get options by -h flag):

```
 -h, --help            show this help message and exit
  -e EXP_NUMBER         Name of existing experiment to be used. (show all experiments via -info)].
  -i N_GENERATIONS      Number generations to perform.
  -n SIZE_POPULATION    The size of the initial population of scenario candidates.
  -a ALGORITHM          The algorithm to use for search. (Currently only 1: NSGAII supported.)
  -t MAXIMAL_EXECUTION_TIME
                        The time to use for search.
  -f SCENARIO_PATH      The path to the scenario description file.
  -min VAR_MIN [VAR_MIN ...]
                        The lower bound of each search parameter.
  -max VAR_MAX [VAR_MAX ...]
                        The upper bound of each search parameter.
  -m DESIGN_NAMES [DESIGN_NAMES ...]
                        The names of the variables to modify.
  -o RESULTS_FOLDER     The name of the folder where the results of the search are stored (default: /results/single/)
  -v                    Whether to use the simuator's visualization. This feature is useful for debugging and demonstrations, however it reduces the search performance.
  -info                 Names of all defined experiments.
```

## Features to be implemented

- [ ] Improve the architecture to define algorithms
- [ ] Implement graphical user interface

## FAQs

- [] TODO (check last review, feedback from students)

## License

OpenSBT is licensed under the [Apache License, Version 2.0](LICENSE).

## Authors

Lev Sorokin (sorokin@fortiss.org) \
Tiziano Munaro (munaro@fortiss.org) \
Damir Safin (safin@fortiss.org) 

