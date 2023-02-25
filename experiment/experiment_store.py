from experiment.experiment import *

from utils.singleton import Singleton

class DefaultExperiments(Singleton):
    def init(self):
        self.store = {}

    def register(self, exp: Experiment):
        if not exp.name in self.store:
            self.store[exp.name] = exp
            return 0
        else:
            print("Experiment with the given name is already registered.")
            return 1

    def load(self, experiment_name: str) -> Experiment:
        if experiment_name in self.store:
            return self.store[experiment_name]
        else:
            print("Experiment with the given name does not exist.")
            return 1

    def get_store(self):
        return self.store

experiments_store = DefaultExperiments()