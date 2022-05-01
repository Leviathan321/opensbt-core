# Search-based Test Case Generation
## Optimizer


Following test case generation algorithms are implemented:

- NSGAII (runNSG2.py)

- NSGAII with decision tree; only one iteration of critical region search (runNSG2DT.py)

Following information needs to be passed to the algorithm:

- fitness functions
- crtiticality functions
- bounds of variables to be optimized
- simulator (currently DummySimulator)
- xosc file (if OpenSCENARIO) is supported
- names of the features to be explored

Optional:

- optimizer specific parameters

### Example

a) 

To run a scenario where ego and other vehicle are moving with constant speed linearly,
uncomment *setExp1* in runNSG2.py.

The fitness function is defined by the smallest distance between ego and other vehicle.
The criticality function is defined arbitrarily to test the approach:

Execute

```
py runNSG2.py
```

b)

To run a carla scenario
uncomment *setExp2* in runNSG2.py.

Execute

```
py runNSG2.py
```


### TODO

- [ ] Create an interface for calling an optimizer.
