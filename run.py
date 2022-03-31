import carla

from srunner.scenarios.open_scenario import OpenScenario
from srunner.scenarioconfigs.openscenario_configuration import OpenScenarioConfiguration
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenario_manager import ScenarioManager

HOST = 'localhost'
PORT = 2000
TIMEOUT = 10
SCENARIO = "scenarios/test.xosc"

client = carla.Client(HOST, PORT)
client.set_timeout(TIMEOUT)

CarlaDataProvider.set_client(client)
CarlaDataProvider.set_world(client.get_world())

config = OpenScenarioConfiguration(
    SCENARIO, CarlaDataProvider.get_client(), {})

vehicles = []
for vehicle in config.ego_vehicles:
    vehicles.append(CarlaDataProvider.request_new_actor(
        vehicle.model,
        vehicle.transform,
        vehicle.rolename,
        color=vehicle.color,
        actor_category=vehicle.category
    ))

CarlaDataProvider.set_traffic_manager_port(int(8000))

scenario = OpenScenario(CarlaDataProvider.get_world(), vehicles, config, SCENARIO)

manager = ScenarioManager()
manager.load_scenario(scenario)

client.start_recorder("test.log", True)

manager.run_scenario()

client.stop_recorder()
