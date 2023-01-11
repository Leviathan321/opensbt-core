# OpenSBT - An open-source framework for the application of search-based testing approaches for Automated and Autonomous Driving Systems


## Intro

The framework provides a modular and extandable code base for the application of search-based testing approaches on AD/ADAS systems.

The tool provides the following features:

- [ ] Feature 1
- [ ] Feature 2
- [ ] Feature 3

Future work features

- [ ] Feature 1
- [ ] Feature 2
- [ ] Feature 3

## Preliminaries

The tool requires python to be installed.
To install all dependencies run:

```
python -m pip install -r requirements.txt
```

### Preliminaries using CARLA Simulator

Follow the steps desribed  [here](https://git.fortiss.org/fortissimo/ff1_testing/ff1_carla) to integrate CARLA.

### Example Usage: Testing a SUT in Carla Simulator

#### Creating the experiment

1. Integrating the Simulator/SUT

2. Defining a fitness function

3. Integrating a search method

4. Defining a testing experiment


#### Running the search

We demonstrate the search with a scenario in carla where a pedestrian crosses the lane of the ego car.

To run search with the predefined search configuration use:

```
python run.py -e 1
```

The results are written in the *results* folder.

To change the search space, the search method and termination creteria run the following.
```
python run.py -e 1 -a 1 -min 0 0 -max 10 2 -m "SpeedEgo" "SpeedPed" -t "01:00:00"
```

#### Analyzing the results

TODO

### Optional Parameters TODO update

All flags that can be set are (get options by -h flag):

```
  -e EXP_NUMBER         Implemented experiments to use (1: Carla AEB Experiment, 2: TBD).
  -i N_ITERATIONS       Number iterations to perform.
  -n SIZE_POPULATION    The size of the initial population of scenario candidates.
  -a ALGORITHM          The algorithm to use for search, 1 for NSGA2, 2 for NSGA2-DT.
  -t MAXIMAL_EXECUTION_TIME
                        The maximal time to be used for search.
  -f XOSC               The path to the scenario description file/experiment.
  -min VAR_MIN [VAR_MIN ...]
                        The lower bound of each search parameter.
  -max VAR_MAX [VAR_MAX ...]
                        The upper bound of each search parameter.
  -m DESIGN_NAMES [DESIGN_NAMES ...]
                        The names of the variables to modify.
```

## Visual Studio Code Integration

To reproduce the example setup included with the OpenSBT framework in [Microsoft Visual Studio Code](https://code.visualstudio.com/) copy the following `launch.json` and `tasks.json` files in the `.vscode` directory of your workspace. Make sure to replace all `/path/to/` paths according to your setup.

### launch.json

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "OpenSBT",
            "type": "python",
            "request": "launch",
            "program": "run.py",
            "console": "integratedTerminal",
            "justMyCode": true,
            "env": {
                "CARLA_ROOT": "/path/to/carla/repository",
                "PYTHONPATH": "/path/to/carla/repository/PythonAPI/carla/dist/carla-0.9.13-py3.7-linux-x86_64.egg:/path/to/carla/repository/PythonAPI/carla/agents:/path/to/carla/repository/PythonAPI/carla:/path/to/carla/scenario/runner/repository",
                "SCENARIO_RUNNER_ROOT": "/path/to/carla/scenario/runner/repository"
            },
            "args": [
                "-e", "1",
                "-n", "30",
                "-i", "50",
                "-t", "01:00:00",
                "-v"
            ],
            "preLaunchTask": "start",
            "postDebugTask": "stop"
        }
    ]
}
```

### tasks.json

```json
{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "make_directory",
            "type": "shell",
            "command": "mkdir",
            "args": [
                "-p",
                "/tmp/recordings"
            ]
        },
        {
            "label": "carla_start",
            "type": "shell",
            "command": "docker",
            "args": [
                "compose",
                "-f", "/path/to/opensbt/carla/runner/docker-compose.yml",
                "up",
                "-d",
                "--scale", "carla-server=2"
            ]
        },
        {
            "label": "carla_stop",
            "type": "shell",
            "command": "docker",
            "args": [
                "compose",
                "-f", "/path/to/opensbt/carla/runner/docker-compose.yml",
                "down"
            ]
        },
        {
            "label": "start",
            "dependsOn": [
                "make_directory",
                "carla_start"
            ]
        },
        {
            "label": "stop",
            "dependsOn": [
                "carla_stop"
            ]
        }
    ]
}

```


## License

OpenSBT is licensed under the [Apache License, Version 2.0](LICENSE).

## Authors

Lev Sorokin (sorokin@fortiss.org) \
Tiziano Munaro (munaro@fortiss.org) \
Damir Safin (safin@fortiss.org)
