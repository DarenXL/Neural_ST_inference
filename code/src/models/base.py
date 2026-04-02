from abc import ABC, abstractmethod
import numpy as np

class SimulationModel(ABC):

    @property
    @abstractmethod
    def param_names(self):
        pass

    def get_param_names(self, alt_format=False):
        return self.param_names

    @property
    @abstractmethod
    def nb_params(self):
        pass

    @abstractmethod
    def to_title(self, vals):
        pass

    @abstractmethod
    def sample_thetas(self, n_samples):
        pass

    @abstractmethod
    def simulate(self, theta_values, time_size, space_dim, space_size, deepset_size):
        pass

    @abstractmethod
    def kriging(self, coords_in, coords_out, y, space_dim, theta_values):
        pass

    @abstractmethod
    def get_likelihood(self, coords, y, theta):
        pass
