import os

import matplotlib.pyplot as plt

from simulator import Simulator
from scenario import Scenario
from recorder import Recorder

from metrics.distance import DistanceBetweenVehicles
from controllers.human import HumanAgent


HOST_CARLA = 'localhost'
PORT_CARLA = 2000
TIMEOUT_CARLA = 10

RECORDING_DIR = '/tmp/recordings'
SCENARIO_DIR = 'scenarios'
METRICS_DIR = 'metrics'

def get_simulator(host, port, timeout):
    return Simulator(host, port, timeout)

def get_scenarios(directory):
    scenarios = None
    with os.scandir(directory) as entries:
        scenarios = [
            Scenario(entry)
                for entry in entries
                    if entry.name.endswith('.xosc') and entry.is_file()
        ]
    return scenarios

def get_evaluator():
    return DistanceBetweenVehicles()

def get_controller():
    return HumanAgent

def get_recorder(directory):
    return Recorder(directory)

simulator = get_simulator(HOST_CARLA, PORT_CARLA, TIMEOUT_CARLA)
scenarios = get_scenarios(SCENARIO_DIR)
recorder = get_recorder(RECORDING_DIR)
evaluator = get_evaluator()
agent = get_controller()

for scenario in scenarios:
    scenario.simulate(simulator, agent, recorder)

recordings = recorder.get_recordings()

evaluations = list()
for recording in recordings:
    evaluations.append(
        evaluator.evaluate(
            simulator,
            recording
        )
    )

for (frame, dist) in evaluations:
    plt.plot(frame, dist)
    plt.ylabel('Distance [m]')
    plt.xlabel('Frame number')
    plt.title('Distance')
    plt.show()
