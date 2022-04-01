# Search-based Test Case Generation
## Optimizer


Following test case generation algorithms are implemented:

- NSGAII (nsgaII_testcase.py)

- NSGAII with decision tree; only one iteration of critical region search (nsgaII_dt.py)

Following information needs to be passed to the algorithm:

- fitness functions
- crtiticality functions
- bounds of variables to be optimized
- simulator (currently DummySimulator)

Optional:

- optimizer specific parameters

### Example

Run a scenario where ego and other vehicle are moving with constant speed linearly.
The fitness function is defined by the smallest distance between ego and other vehicle.
The criticality function is defined arbitrarily to test the approach.

```
py nsgaII_testcase.py
```

### TODO

[] Simulate scenario provided as .xosc file in CARLA.
[] Simulate scenario provided as .peb file to in PRESCAN.
[] Create an interface for calling an optimizer.
[] Fix bugs in nsgaII_dt.py

## Pipecleaner

The `run.bash` script ...

1. ... starts an instance of the CARLA simulator, ...
2. ... loads an OpenSCENARIO specification via CARLA's _Scenario Runner_ (more on that [here](https://carla-scenariorunner.readthedocs.io/en/latest/getting_started/#running-scenarios-using-the-openscenario-format)),...
3. ... allows manual control of the ego vehicle via CARLA's `VehicleControl` API (more on that [here](https://carla.readthedocs.io/en/latest/python_api/#carla.VehicleControl)), ...
4. ... records the simulation (more on that [here](https://carla.readthedocs.io/en/latest/adv_recorder/#recording)), ...
5. ... and uses the Scenario Runner's _Metrics Manager_ to extract and evaluate the ego vehicle's distance to the vehicle in front (more on that [here](https://carla-scenariorunner.readthedocs.io/en/latest/metrics_module/#3-run-the-metrics-manager)).

### Known Issues

The metrics manager might cause a segmentation fault in the simulation process. Still looking into that ...
