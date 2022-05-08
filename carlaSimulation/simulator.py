import carla


class Simulator:

    def __init__(self, host, port, timeout):
        self.client = carla.Client(host, port)
        self.client.set_timeout(timeout)
        self.apply_settings()

    def get_client(self):
        return self.client

    def apply_settings(self):
        world = self.client.get_world()
        settings = world.get_settings()

        settings.fixed_delta_seconds = 0.1
        settings.synchronous_mode = True

        world.apply_settings(settings)

        traffic_manager = self.client.get_trafficmanager(int(8000))
        traffic_manager.set_synchronous_mode(True)
