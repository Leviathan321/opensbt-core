# Search-based Test Case Generation
## Intro


The tool implements search-based critical test case generation based on this [approach](https://orbilu.uni.lu/bitstream/10993/33706/1/ICSE-Main-24.pdf).
It supports pure search with NSGA2 as well additionally clustering based search with DT.

## Usage


The tool can be used together with Prescan Simulator, but we are also working on the integration of the Carla Simulator. 
Since this is not yet mature, we describe only the integration with Prescan.


### Preliminaries


The search algorithm requires that Python (>= 3.7), Matlab, and Prescan is installed. The matlab engine needs to be exported to python ([s. here](https://de.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html)).

Compatibility has been tested with Prescan 2019.b and MATLAB R2019.b (Versions should match.)

Install dependencies by executing:


```
pip install -r requirements.txt
```

### Search

Start MATLAB using the Prescan Process Manager and share the engine by executing in the terminal:

```
matlab.engine.shareEngine
```

### Example

To run search with an example experiment
make sure **PrescanHeedsExperiment** is downloaded in a folder **experiments** that is placed next to this.

Run the following to execute search:

```
py run.py -e 8
```

### New Experiment

Make sure to have a file named **UpdateModel.m** in the experiments folder that reads from a json file **input.json** parameter values and sets the values in the experiment model.
Consider as an example experiment **experiments/PrescanHeedsExperiment**

Run the tool by providing the path to the experiment file, the upper and lower bounds, as well the names of the parameters to vary (should match with the ones set by **UpdateModel.m**):

```
py run.py -f <experiment.pb> -min 1 1 1 -max 10 20 10 -m "par1 "par2" "par3"
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
                        The lower bound of each parameter.
  -max VAR_MAX [VAR_MAX ...]
                        The upper bound of each parameter.
  -m FEATURE_NAMES [FEATURE_NAMES ...]
                        The names of the features to modify.
```


## Limitations

Since OpenSCENARIO support of Prescan is not mature, Prescan experiment files have to be used.

## Authors

Lev Sorokin (sorokin@fortiss.org)
