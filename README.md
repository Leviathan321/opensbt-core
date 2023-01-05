# OpenSBT - An open-source framework for the application of search-based testing approaches for Automated and Autonomous Driving Systems


## Intro

The framework provides a modular and extandable code base for the application of search-based testing approach on AD/ADAS systems.

The tool provides the following features:

## Preliminaries

The tool requires python to be installed.
To install all dependencies run:

```
python -m pip install -r requirements.txt
```

### Preliminaries using CARLA Simulator

Follow the steps desribed  [here](https://git.fortiss.org/fortissimo/ff1_testing/ff1_carla) to integrate CARLA.

### Usage


The results are written in the *results* folder.

### Example 


### Testing a SUT in Carla Simulator 

To run search with a scenario in carla we have implemented two examples  where a pedestrian crosses the lane of the ego. In one the environment is modified, and the other only the pedestrians speed and host speed is modified.
To
```
python run.py -e 1
```
### Running search on MOO Problem

Run the following to execute search with a mathematical multobjective problem.

```
python run.py -e 2
```

### Optional Parameters

All flags that can be set are (get options by -h flag):

```
  -e EXP_NUMBER         Hardcoded example scenario to use [2 to 6].
  -i N_ITERATIONS       Number iterations to perform.
  -n SIZE_POPULATION    The size of the initial population of scenario candidates.
  -a ALGORITHM          The algorithm to use for search, 1 for NSGA2, 2 for NSGA2-DT.
  -t MAXIMAL_EXECUTION_TIME
                        The time to use for search with nsga2-DT (actual search time can be above the threshold, since algorithm might
                        perform nsga2 iterations, when time limit is already reached.
  -f XOSC               The path to the scenario description file/experiment.
  -min VAR_MIN [VAR_MIN ...]
                        The lower bound of each parameter.
  -max VAR_MAX [VAR_MAX ...]
                        The upper bound of each parameter.
  -m DESIGN_NAMES [DESIGN_NAMES ...]
                        The names of the variables to modify.
  -dt MAX_TREE_ITERATIONS
                        The maximum number of total decision tree generations (when using NSGA2-DT algoritm).
```

## Limitations


## Authors

Lev Sorokin (sorokin@fortiss.org)
Tiziano Munaro (munaro@fortiss.org)
Damir Safin (safin@fortiss.org)
