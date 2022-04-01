import importlib
import carla
import os
import sys
import importlib
import inspect

from srunner.scenarios.open_scenario import OpenScenario
from srunner.scenarioconfigs.openscenario_configuration import OpenScenarioConfiguration
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenario_manager import ScenarioManager
from srunner.metrics.tools.metrics_log import MetricsLog

HOST = 'localhost'
PORT = 2000
TIMEOUT = 10

RECORDINGS = '/tmp/recordings'
SCENARIOS = 'scenarios'
METRICS = 'metrics'

def get_metrics_class(file):
    module_name = os.path.basename(file).split('.')[0]
    sys.path.insert(0, os.path.dirname(file))
    metric_module = importlib.import_module(module_name)
    for member in inspect.getmembers(metric_module, inspect.isclass):
        member_parent = member[1].__bases__[0]
        if 'BasicMetric' in str(member_parent):
            return member[1]


client = carla.Client(HOST, PORT)
client.set_timeout(TIMEOUT)

world = client.get_world()

CarlaDataProvider.set_client(client)
CarlaDataProvider.set_world(world)

current_scenario = "test"
current_metric = "test"

scenario_file = "{}/{}.xosc".format(SCENARIOS, current_scenario)
recording_file = "{}/{}.log".format(RECORDINGS, current_scenario)
metric_file = "{}/{}.py".format(METRICS, current_metric)

config = OpenScenarioConfiguration(
    scenario_file,
    client,
    {}
)

vehicles = []
for vehicle in config.ego_vehicles:
    vehicles.append(
        CarlaDataProvider.request_new_actor(
            vehicle.model,
            vehicle.transform,
            vehicle.rolename,
            color=vehicle.color,
            actor_category=vehicle.category
        )
    )

CarlaDataProvider.set_traffic_manager_port(int(8000))

scenario = OpenScenario(world,
                        vehicles, config, scenario_file)

manager = ScenarioManager()
manager.load_scenario(scenario)
client.start_recorder(recording_file, True)
manager.run_scenario()
client.stop_recorder()

recording_info = client.show_recorder_file_info(recording_file, True)
metric_log = MetricsLog(recording_info)

metrics_class = get_metrics_class(metric_file)
metrics_class(client.get_world().get_map(), metric_log, None)
