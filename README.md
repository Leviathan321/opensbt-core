# Search-based Test Case Generation
## Intro


The tool implements search-based critical test case generation based on this [approach](https://orbilu.uni.lu/bitstream/10993/33706/1/ICSE-Main-24.pdf).
It supports pure search with NSGA2 as well additionally clustering based search with DT.

## Preliminaries

The tool can be used together with Prescan Simulator and the Carla Simulator. 
The search algorithm requires that Python (>= 3.7) is installed.

Install dependencies by executing:


```
pip install -r requirements.txt
```

### Preliminaries using Carla

Follow the steps desribed  [s. here](/carlaSimulation) to integrate Carla.

### Preliminaries using Prescan

To use Prescan matlab needs to be installed. 
Compatibility has been tested with Prescan 2021.3.0 and MATLAB R2019.b.

#### Matlab

Matlab R2019.b can be downloaded from <file://///fs01/Install/Mathworks> (VPN to fortiss network/local LAN connection required). Installation instruction is available here:  <https://one.fortiss.org/sites/workinghere/Wikipages/install%20some%20software.aspx>

The matlab engine needs to be exported to python ([s. here](https://de.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html)).

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



### Search with Prescan

Start MATLAB using the Prescan Process Manager and share the engine by executing in the terminal:

```
matlab.engine.shareEngine
```



#### Real Example

To run search with an example Prescan experiment
make sure **PrescanHeedsExperiment** is downloaded in a folder **experiments** that is placed next to this.

Run the following to execute search:

```
py run.py -e 8
```

#### New Experiment

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
                        The names of the features to modify. Format: <actorname>_<feature>, i. e. "Ego_HostVelGain", "Other_Velocity_mps".

```


## Limitations

Since OpenSCENARIO support of Prescan is not mature, Prescan experiment files have to be used.

## Authors

Lev Sorokin (sorokin@fortiss.org)
