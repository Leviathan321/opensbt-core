# Search-based Test Case Generation
## Intro


The tool implements search-based critical test case generation based on this [approach](https://orbilu.uni.lu/bitstream/10993/33706/1/ICSE-Main-24.pdf).
It supports pure search with NSGA2 as well additionally clustering based search with DT to accelerate critical scenario generation.

## Preliminaries

The tool can be used together with the Carla Simulator. 
The search algorithm requires that Python (>= 3.7) is installed.
Usage

Install dependencies by executing:

```
pip install -r requirements.txt
```

### Preliminaries using Carla

Follow the steps desribed  [s. here](/carlaSimulation) to integrate Carla.

### Usage

### Simplified Example (No simulator)
Run the following to execute search with a uniform motion of an ego and an other actor without any physics or environment simulation (The initial positions of ego and the other actor fixed, only velocity and orientation is varied):

```
py run.py -e 1
```

### Search with Carla

To run search with an example Carla run:

```
py run.py -e 2 
```
or

```
py run.py -e 3
```

### Optional Parameters

All flags that can be set are (get options by -h flag):

```
  -e EXPNUMBER          Hardcoded example scenario to use (possible 1, 9).
  -i NITERATIONS        Number iterations to perform.
  -n SIZEPOPULATION     The size of the initial population of scenario
                        candidates.
  -a ALGORITHM          The algorithm to use for search, 0 for nsga2, 1 for
                        nsga2dt.
  -t TIMESEARCH         The time to use for search with nsga2-DT (actual
                        search time can be above the threshold, since
                        algorithm might perform nsga2 iterations, when time
                        limit is already reached.
  -f XOSC               The path to the .xosc Carla-Experiment.
  -min VAR_MIN [VAR_MIN ...]
                        The lower bound of each parameter.
  -max VAR_MAX [VAR_MAX ...]
                        The upper bound of each parameter.
  -m FEATURE_NAMES [FEATURE_NAMES ...]
                        The names of the features to modify. Format: <actorname>_<feature>, i. e. "Ego_HostVelGain", "Other_Velocity_mps".

```

## Authors

Lev Sorokin (sorokin@fortiss.org)
