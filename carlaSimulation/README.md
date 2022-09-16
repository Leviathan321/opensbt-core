# CARLA Interface for Scenario-based Testing

## Prerequisites

The CARLA server must already be running when the simulation is started. Due to its lightweight, ease of use, and modularity, a Docker-based setup is preferred. 

The CARLA interface will instruct the server to record the simulation runs and to store the data in the container's `/tmp/recordings` folder. In order to access these files, this directory is mounted by the host as well. However, make sure that the user with ID `1000` must be the directory's owner!

### Docker 
Instructions to install docker are available here [here](https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository).

### Carla
The CARLA repository can be cloned from [here](https://github.com/carla-simulator/carla).

### ScenarioRunner

To execute scenarios in Carla the application *ScenarioRunner* is required. The documentation of the tool can be found here:
[here](https://github.com/carla-simulator/scenario_runner/blob/master/Docs/getting_scenariorunner.md).

The repository can be cloned from [here](https://github.com/carla-simulator/scenario_runner).

## Getting Started

To start the container with default settings, run `docker compose up` in this repository's root folder. Instructions on Docker Compose can be found [here](https://docs.docker.com/compose/). 

Once CARLA is up and running, execute the `run.py` script. Make sure that the `PYTHONPATH` includes all of the dependencies required by the CARLA ScenarioRunner and the `CARLA_ROOT` and `SCENARIO_RUNNER_ROOT` point to the respective local repositories. 

### VS Code

If you use Visual Studio Code, the following launch file might be useful reference:

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "CARLA",
            "type": "python",
            "request": "launch",
            "program": "run.py",
            "console": "integratedTerminal",
            "justMyCode": true,
            "env": {
                "CARLA_ROOT": "~/Repositories/CARLA/Simulator",
                "PYTHONPATH": "~/Repositories/CARLA/Simulator/PythonAPI/carla/dist/carla-0.9.13-py3.7-linux-x86_64.egg:~/Repositories/CARLA/Simulator/PythonAPI/carla/agents:~/Repositories/CARLA/Simulator/PythonAPI/carla:~/Repositories/CARLA/ScenarioRunner",
                "SCENARIO_RUNNER_ROOT": "~/Repositories/CARLA/ScenarioRunner"
            }
        }
    ]
}
```
### Running scenarios

Consider to change the map in Carla to the map used in the scenario before running a scenario.

```
cd pathToCarla/carla/PythonAPI/utils
python3 config.py --map Town01
```
