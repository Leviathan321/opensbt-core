import os

from carlaSimulation.controllers.npc import NpcAgent

from carlaSimulation.simulator import Simulator
from carlaSimulation.scenario import Scenario
from carlaSimulation.recorder import Recorder

from carlaSimulation.metrics.raw import RawData


HOST_CARLA = 'localhost'
PORT_CARLA = 2000
TIMEOUT_CARLA = 15
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
    return RawData()

def get_controller():
    return NpcAgent

def get_recorder(directory):
    return Recorder(directory)

def run_scenarios(scenario_dir=SCENARIO_DIR,recording_dir=RECORDING_DIR):

    simulator = get_simulator(HOST_CARLA, PORT_CARLA, TIMEOUT_CARLA)
    scenarios = get_scenarios(scenario_dir)
    recorder = get_recorder(recording_dir)
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

    return evaluations
