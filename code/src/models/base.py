from abc import ABC, abstractmethod
import numpy as np

class SimulationModel(ABC):
    @property
    @abstractmethod
    def param_names(self):
        pass

    @abstractmethod
    def sample_thetas(self, n_samples):
        pass

    @abstractmethod
    def simulate(self, theta_values, time_size, space_size, deepset_size):
        pass