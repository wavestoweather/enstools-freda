from .algorithm import Algorithm
import numpy as np


class DebugDatatypes(Algorithm):

    @staticmethod
    def assimilate(state: np.ndarray, state_map: np.ndarray,
                   observations: np.ndarray, observation_type: np.ndarray, reports: np.ndarray,
                   points_in_radius: np.ndarray, weights: np.ndarray, updated: np.ndarray, det: int, rho: float):
        """
        only check the data types for debugging
        """
        assert state.dtype == np.float32
        assert state_map.dtype == np.int32
        assert observations.dtype == np.float32
        assert observation_type.dtype == np.int32
        assert reports.dtype == np.int32
        assert points_in_radius.dtype == np.int32
        assert weights.dtype == np.float32
        assert updated.dtype == np.int8
        assert type(det) == int
        assert type(rho) == float

    @staticmethod
    def weights_for_gridpoint(localization_radius: float, distance: np.ndarray):
        """
        only check the data types
        """
        assert type(localization_radius) == float
        assert distance.dtype == np.float32
        return np.empty_like(distance)
