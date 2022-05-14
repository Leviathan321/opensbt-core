from pathlib import Path

from srunner.scenarios.open_scenario import OpenScenario
from srunner.scenarioconfigs.openscenario_configuration import OpenScenarioConfiguration
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenario_manager import ScenarioManager


class Scenario:

    def __init__(self, xosc):
        self.xosc = xosc

    def simulate(self, simulator, controller, recorder):
        client = simulator.get_client()

        world = client.get_world()

        settings = world.get_settings()
        settings.no_rendering_mode = True
        world.apply_settings(settings)

        CarlaDataProvider.set_client(client)
        CarlaDataProvider.set_world(world)

        config = OpenScenarioConfiguration(
            self.xosc,
            client,
            {}
        )

        CarlaDataProvider.set_traffic_manager_port(int(8000))

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

        agent = controller("")

        scenario = OpenScenario(
            world,
            vehicles,
            config,
            self.xosc
        )

        recording = recorder.add_recording(
            Path(self.xosc.path).stem
        )

        manager = ScenarioManager()
        manager.load_scenario(scenario, agent)
        client.start_recorder(
            recording,
            True
        )
        manager.run_scenario()
        client.stop_recorder()
