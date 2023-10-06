# OpenSBT - A Modular Framework for Search-based Testing of Automated Driving Systems


## Intro

OpenSBT provides a modular and extandable code base for the search-based testing of automated driving systems. It provides interfaces to integrate search algorithms, fitness/criticality functions and simulation environments in a modular way. That means, one of this components can be replaced by a another component without the need to change the connectivity to other components or adapt the result analysis/visualization. Further it visualizes the test outcome and analysis the critical behaviour of the ADS. 

A demo video of OpenSBT can be found here: https://www.youtube.com/watch?v=oOyug8rwAB8.


## Architecture

[<img src="doc/figures/OpenSBT_architecture.png" width="500"/>]()

OpenSBT builds upon [Pymoo](https://pymoo.org/) and extends internal optimization related models to tailor heuristic search algorithms for testing ADS systems.

## Installation

OpenSBT requires python to be installed and its compatibality has been tested with python 3.7 and 3.8. OpenSBT can be run as a standalone application or can be imported as a library. To use it in the standalone mode follow the installation instructions [here](/doc/jupyter/01_Installation.ipynb). To import it as a library you need to install the correspnding pip package [PIP-package](TODO). 

The benefit of using the standalone mode is that we can modify and execute existing testing experiments by using command line flags/operations.

## Usage

After having installed OpenSBT, you can follow the tutorials provided as [jupyter notebooks](doc/jupyter/) which explain step-by-step of how to use OpenSBT.
In these tutorials, we have integrated 

- [1] A simplified SUT simulated in very simplistic simulator (linear motion planning) 
- [2] A real AEB agent simulated in [CARLA](https://carla.org/) using the simulator adapter [CARLA Runner Extension](https://git.fortiss.org/opensbt/carla-runner).

Note: We have also implemented a [simulator adapter](https://git.fortiss.org/opensbt/prescan_runner) for the execution of Prescan experiments.

We have also provided a [virtual machine]() where you can execute the jupyter notebooks with the dummy simulator.

## Flags

Following flags can be set when running OpenSBT in the standalone mode (via python run.py):

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

## FAQs

- [] TODO (check last review, feedback from students)

## License

OpenSBT is licensed under the [Apache License, Version 2.0](LICENSE).

## Authors

Lev Sorokin (sorokin@fortiss.org) \
Tiziano Munaro (munaro@fortiss.org) \
Damir Safin (safin@fortiss.org) 

