from abc import ABC, abstractmethod

import gym
import numpy as np


class Agent(ABC):

    def __init__(self, env: gym.Env = None):
        self.env = env

    @abstractmethod
    def predict(self, obs: np.ndarray, speed: float = 0.0) -> np.ndarray:
        raise NotImplementedError('Not implemented')

