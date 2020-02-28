from enstools.da.support.feedback_file import LevelType
from numba import jit, i4, f4
import numpy as np
from abc import ABC, abstractmethod


class Algorithm(ABC):
    """
    Base class for new algorithms. Every algorithm has to implement the assimilate method.
    """
    @staticmethod
    @abstractmethod
    def assimilate(state: np.ndarray, state_map: np.ndarray,
                   observations: np.ndarray, observation_type: np.ndarray, reports: np.ndarray,
                   points_in_radius: np.ndarray, weights: np.ndarray, updated: np.ndarray):
        """
        This function is called multiple times with different subsets of observation reports. All reports are
        processable without updating overlapping areas of the model domain. That means, the reports are guaranteed
        to have no overlap in the grid points that are affected by them. Or the the reports have a total overlap.

        Parameters
        ----------
        state:
                Complete state including all ensemble members. Shape: (grid-points, layers, member). All variables are
                stacked on top of each other. That means, variable one may use the layers 0 to 89 and variable two the
                layers 90 to 179. Type: float32

        state_map:
                Location of variables within the state array. Shape: (size of variable table, 2). Meaning of the two
                properties per variable: 0=first index of layers dimension in state, 1=number of indices in layer
                dimension in state. This array can be used like a dictionary. Example: temperature has the variable
                number 2 in in feedback files. A loop over all layers of temperature in the state would look like this:
                for l in range(state_map[2, 0], state_map[0] + state_map[1]). Variables that are not included in our
                state have values of -1 at the corresponding position of the state_map

        observations:
                This array will always include all observations. It is the responsibility of the algorithm to select
                the right ones based on the content of the reports array. Shape: (n-obs, 3). The three values per
                observation are: 0=value, 1=error, 2=level. Type: float32

        observation_type:
                Array with shape (n-obs, 2). The first value is the variable number for each observation. The second
                value the level type. 0=Model Level, 251=Pressure. See FeedbackFile enum LevelType.
                In future, additional information are possibly added as necessary.

        reports:
                This array contains a list of reports that should be processed within this call of the function. A
                report contains of a number of observation that belongs to the same grid points. Shape: (n-reports, 4).
                The four values per report are: 0=first index of an included observation in the observation array,
                1=number of consecutive observations in the observations array, 2=index in the state of the nearest
                grid point, 3=index in the points_in_radius array.

        points_in_radius:
                Array with shape (n-unique-reports, max points in radius). The first dimension is the index in
                reports[:,4]. the second dimension includes all points that are in the localization radius of a specific
                report. The number of points in the localization radius is not constant. The second dimension of the
                array is large enough to hold all values for the largest number of grid points within the radius. In
                smaller cases, -1 ist used as a fill values.

        weights:
                Array with the same shape as points_in_radius. For each point in each localization radius of each
                observation a weigth between 0 and 1 is provided.

        updated:
                in order to update the overlapping regions, the calling run function must know which indices of the grid
                have been updated. Only those are communicated with the other processors. Shape: (state.shape[0]).
                Valid values: 0=not updated, 1=updated. This array is initialized with zeros, here we only need to write
                ones.
        """

    @staticmethod
    def weights_for_gridpoint(localization_radius: float, distance: np.ndarray):
        """
        This function is used to calculate a weight for each grid points in the localization radius. It is called from
        DataAssimilation.run on all affected grid points and the result is forwarded to the assimilate method.

        Parameters
        ----------
        localization_radius:
                localization radius in m.

        distance:
                distance in m.

        Returns
        -------
        value between 0 and 1.
        """
        return np.exp(-0.5*(distance/localization_radius)**2)



@jit("i4(f4[:,:,:], i4[:,:], i4, i4, f4)", nopython=True, nogil=True)
def nearest_vertical_level(state: np.ndarray, state_map: np.ndarray, grid_index: int, i_ens: int, level: float):
    """
    Find the index of the closest vertical level for a given pressure level. The pressure field from the state
    variable is used.

    Parameters
    ----------
    state:
            state variable as in assimilate method.

    state_map:
            map of the state variable as in assimilate method.

    grid_index:
            index of the grid cell within the state variable

    i_ens:
            number of the ensemble member to look at.

    level:
            vertical level we want to find within the state.

    """
    # variable P has the ID 251 (see FeedbackFile class). Make sure, that it is part of the state
    p = LevelType.PRESSURE.value
    assert state_map[p, 0] != -1

    # start with the pressure value in the first level
    i_value = state_map[p, 0]
    p_diff_nearest = np.abs(state[grid_index, i_value, i_ens] - level)

    # loop over all vertical level to find the closest level
    for i_level in range(state_map[p, 0] + 1, state_map[p, 0] + state_map[p, 1]):
        p_diff_current = np.abs(state[grid_index, i_level, i_ens] - level)
        if p_diff_current < p_diff_nearest:
            p_diff_nearest = p_diff_current
            i_value = i_level

    return i_value - state_map[p, 0]


@jit("f4(f4[:,:,::1], i4[:,::1], i4, f4[:,::1], i4[:,::1], i4, f4[::1], f4[::1])",
     nopython=True, nogil=True, locals={"level": i4, "i_ens": i4, "mean": f4})
def model_equivalent(state: np.ndarray, state_map: np.ndarray, grid_index: int,
                     observations: np.ndarray, observation_type: np.ndarray, i_obs: int,
                     result: np.ndarray, result_deviation_from_mean: np.ndarray):
    """
    Extract the model equivalent from the state. In case of observations on model levels, the value of the corresponding
    grid cell from the state is taken. In case of pressure levels, we pick the value at the closest grid cell.

    Parameters
    ----------
    state:
            state variable as in assimilate method.

    state_map:
            mapping for the content of the state as in assimilate method.

    grid_index:
            horizontal index in the local part of the state.

    observations:
            observations array as in assimilate method.

    observation_type:
            observation_type variable as in assimilate method.

    i_obs:
            index of the observation to process.

    result:
            1-D array with the model equivalent of each ensemble member

    result_deviation_from_mean:
            result - mean

    Returns
    -------
    mean value of the result array
    """

    # loop over all ensemble members
    mean = 0.0
    for i_ens in range(state.shape[2]):
        # at first, find the vertical level of this observation, unless it is already given as an input
        if observation_type[i_obs, 1] == LevelType.PRESSURE.value:
            level = nearest_vertical_level(state, state_map, grid_index, i_ens, observations[i_obs, 2])
        elif observation_type[i_obs, 1] == LevelType.MODEL_LEVEL.value:
            level = int(observations[i_obs, 2])
        else:
            raise ValueError("unsupported observation level. Only model and pressure are allowed.")

        # write the value of the state into the result array
        result[i_ens] = state[grid_index, state_map[observation_type[i_obs, 0], 0] + level, i_ens]
        mean += result[i_ens]
    mean /= state.shape[2]

    # calculate deviation from mean
    for i_ens in range(state.shape[2]):
        result_deviation_from_mean[i_ens] = result[i_ens] - mean
    return mean


@jit("f4(f4[:,:,:], i4, i4)", nopython=True, nogil=True, locals={"sum": f4}, inline='always')
def ensemble_mean(state: np.ndarray, grid_index: int, level_index: int) -> np.float32:
    """
    Calculate the ensemble mean for a given location in the state.

    Parameters
    ----------
    state:
            state as in assimilate method.

    grid_index:
            horizontal grid index.

    level_index:
            vertical grid index (includes variables)

    Returns
    -------
    mean value at the given location
    """
    sum = 0.0
    for i_ens in range(state.shape[2]):
        sum += state[grid_index, level_index, i_ens]
    sum /= state.shape[2]
    return sum


@jit("f4(f4[:,:,:], i4,i4, f4[:])", nopython=True, nogil=True, locals={"cov": f4}, inline='always')
def covariance(state: np.ndarray, grid_index: int, level_index: int, deviation_from_mean: np.ndarray):
    """
    calculate the covariance between the model equivalent (given already as deviation from mean)
    and a given grid cell / variable in the state.

    Parameters
    ----------
    state:
            state variable as in assimilate method.

    grid_index:
            horizontal grid index.

    level_index:
            vertical grid index and variable.

    deviation_from_mean:
            1-D array with model equivalent minus mean or model equivalent for each ensemble member.

    Returns
    -------
    covariance.
    """
    # get the mean value of the given location in the state
    mean_state = ensemble_mean(state, grid_index, level_index)

    # compute covariance
    cov = 0.0
    for i_ens in range(state.shape[2]):
        cov += (state[grid_index, level_index, i_ens] - mean_state) * deviation_from_mean[i_ens]
    cov /= (state.shape[2] - 1)
    return cov
