import os

import matplotlib.pyplot as plt

from carlaSimulation.simulator import Simulator
from carlaSimulation.scenario import Scenario
from carlaSimulation.recorder import Recorder

from carlaSimulation.metrics.distance import DistanceBetweenVehicles
from carlaSimulation.controllers.human import HumanAgent


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


def run(xosc,recording_dir=RECORDING_DIR):

    simulator = get_simulator(HOST_CARLA, PORT_CARLA, TIMEOUT_CARLA)
    recorder = get_recorder(recording_dir)
    evaluator = get_evaluator()
    agent = get_controller()
    print(xosc)
    scenario = Scenario(xosc)
    scenario.simulate(simulator, agent, recorder)

    recordings = recorder.get_recordings()

    evaluation = evaluator.evaluate(
                simulator,
                recordings[0]
            )

    # for (frame, dist) in evaluations:
    #     plt.plot(frame, dist)
    #     plt.ylabel('Distance [m]')
    #     plt.xlabel('Frame number')
    #     plt.title('Distance')
    #     plt.show()

    #print(evaluation)
        
    return evaluation

