import carla

class Simulator:

    def __init__(self, host, port, timeout):
        self.client = carla.Client(host, port)
        self.client.set_timeout(timeout)

    def get_client(self):
        return self.client
