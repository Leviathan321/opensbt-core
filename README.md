# Search-based Test Case Generation
## Intro


The tool implements search-based critical test case generation based on this [approach](https://orbilu.uni.lu/bitstream/10993/33706/1/ICSE-Main-24.pdf).
It supports pure search with NSGA2 as well additionally clustering based search with DT.

## Usage


The tool can be used together with Prescan Simulator, but we are also working on the integration of the Carla Simulator. 

### Preliminaries


The search algorithm requires that Python (>= 3.7), and Carla is installed. 

Install dependencies by executing:

```
pip install -r requirements.txt
```

### Example

Run the following to execute search with uniform motion of an ego and an other actor:

```
py run.py -e 1
```

To run search with an example Carla scenario run:

```
py run.py -e 2
```

### New Scenario

Execute the followoring to run search for a scenario provided in OpenSCENARIO v1 format:

```
py run.py -f <scenario.xosc> -min 1 1 1 -max 10 20 10 -m "par1 "par2" "par3"
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
  -f XOSC               The path to the .pb file of the Prescan Experiment.
  -min VAR_MIN [VAR_MIN ...]
                        The upper bound of each parameter.
  -max VAR_MAX [VAR_MAX ...]
                        The lower bound of each parameter.
  -m FEATURE_NAMES [FEATURE_NAMES ...]
                        The names of the features to modify.
```

### Issues

There are prescan related imports in the run.py, which need to be removed when using a carla simulation setup. 

TODO import libs dependent on the scenario format provided
