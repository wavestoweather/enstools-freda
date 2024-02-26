from .algorithm import Algorithm, model_equivalent, covariance, lin_solve
from numba import jit, i4, f4, prange
import numpy as np
from numba.core import types

class QPEns(Algorithm):

    @staticmethod
    @jit("void(f4[:,:,::1], i4[:,::1], i4[:,::1], f4[:,::1], i4[:,::1], i4[:,::1], i4[:,::1], f4[:,::1], f4[:,::1], i1[::1], i4, f4[::1])",
         nopython=True, nogil=True, parallel=True,
         locals={"i_report": i4, "i_obs": i4, "i_radius": i4, "i_layer": i4, "i_points": i4, "i_cell": i4,
             "p_equivalent": f4, "denominator": f4, "p": f4, "i": i4, "n_inv": f4, "i_nens": i4, "j": i4})
    def assimilate(state: np.ndarray, state_map: np.ndarray, state_map_inverse: np.ndarray,
                   observations: np.ndarray, observation_type: np.ndarray, reports: np.ndarray,
                   points_in_radius: np.ndarray, weights_h: np.ndarray, weights_v: np.ndarray, 
                   updated: np.ndarray, det: int, rho: np.ndarray):
        """
        see Algorithm class for documentation of arguments.
        """
        # temporal variables
        det = 0
        print("################11111111111111111111111111111111111############################")
        n_varlayer = state.shape[1]
        n_ens = state.shape[2]
        n_inv = 1. / (n_ens - 1)
        equivalent = np.empty(n_ens, dtype=np.float32)
        deviation_equivalent_mean = np.empty(n_ens, dtype=np.float32)
        print("####################222222222222222222222222222222###############################")
        # observations are processed one by one in the order that they are listed in the reports array
        for i_report in range(reports.shape[0]):
            # all observations in this report are located at this index within the local part of the grid.
            grid_index = reports[i_report, 2]
            innovation = np.empty((reports[i_report, 1],n_ens), dtype=np.float32)
            Y = np.empty((reports[i_report, 1],n_ens), dtype=np.float32)
            R = np.empty(reports[i_report, 1], dtype=np.float32)
            print(i_report,"#####################333333333333333333333333#######################")
            assert grid_index != -1
            
            # loop over all observations in this report
            for i_obs in range(reports[i_report, 0], reports[i_report, 0] + reports[i_report, 1]):
                print(i_report,i_obs,"#####################looooooooooooooooooooooop######################")
                obs_layer = int(observations[i_obs,2])
                mlevel_obs = state_map_inverse[obs_layer,1]
                # get model equivalents for the given observation and the mean which is later used for covariances
                # for observation on model levels, model_equivalent returns just the corresponding gird cell.
                equivalent_mean = model_equivalent(state, state_map, grid_index, observations, observation_type, i_obs,
                                 equivalent, deviation_equivalent_mean)
                print(i_report,i_obs,"################################4444444444444444444444444444444444#################################")
                # calculate innovation from observation value[i_obs, 0] and observation error[i_obs, 0]
               #
                #print(observations[i_obs, 0].shape)
                #print(np.random.normal(0, observations[i_obs, 1], n_ens-det).shape,equivalent.shape,innovation[i_obs-reports[i_report, 0],:].shape)
                print(np.random.normal(0, observations[i_obs, 1], n_ens-det).shape)
                print(equivalent.shape)
                print(innovation[i_obs-reports[i_report, 0],:].shape)
                innovation[i_obs-reports[i_report, 0],:] = observations[i_obs, 0] + np.random.normal(0, observations[i_obs, 1], n_ens-det) - equivalent
                print(i_report,i_obs,"####################555555555555555555555555555555555##################")
                Y[i_obs-reports[i_report, 0],:] = deviation_equivalent_mean 
                print(i_report,i_obs,"#########################66666666666666666666666666666#########################")
                R[i_obs-reports[i_report, 0]] = observations[i_obs, 1]**2 
                print(i_report,i_obs,"#############################7777777777777777777777777#############################")
            
            # Solve Av=b
            print("#############SOLVING THE SYSTEM#####################")
            A = n_inv * rho[mlevel_obs] * np.dot(Y,Y.T) + np.diag(R)
            #assert innovation.dtype == np.dtype(np.float32)
            v = np.empty((reports[i_report, 1], n_ens), dtype=np.float32)
            v[:] = lin_solve(A,innovation)
            print("#############SYSTEM SOLVED###########################")
            #v = innovation #np.linalg.solve(A,innovation)
            # loop over all grid cells and all variables that are within the localization radius
            # This loop runs in parallel if NUMBA_NUM_THREADS is larger than 1.
            i_points = reports[i_report, 3]
            for i_radius in prange(points_in_radius.shape[1]):

                # the number of points for each observation is not constant. stop the loop as soon as we reach
                # a grid cell index of -1
                i_cell = points_in_radius[i_points, i_radius]
                if i_cell == -1:
                    continue

                # mark the current point as updated. This will cause updates of overlapping areas between processors
                updated[i_cell] = 1

                # loop over all layers of the state, this is also a loop over all variables as variables are stacked
                # on top of each other in the state variable.
                for i_layer in range(n_varlayer):
                    # calculate covariance between model equivalent and the current location in the state
                    mlevel_state = state_map_inverse[i_layer,1]
                    PH = np.empty((1,reports[i_report, 1]),dtype=np.float32)
                    for j in range(reports[i_report, 1]):
                        PH[0,j] =  covariance(state, i_cell, i_layer, Y[j,:]) * weights_h[i_points, i_radius]

                    state[i_cell, i_layer, :] += np.dot(PH,v)[0,:]

                
         
