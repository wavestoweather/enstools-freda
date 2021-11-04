from .algorithm import Algorithm, model_equivalent, covariance, weights_for_gridpoint
from numba import jit, prange, i4, f4
import numpy as np
import math

class EnSRF(Algorithm):

    @staticmethod
    @jit("void(f4[:,:,::1], i4[:,::1], i4[:,::1], f4[:,::1], i4[:,::1], i4[:,::1], i4[:,::1], f4[:,::1], i1[::1], i4, f4)",
         nopython=True, nogil=True, parallel=True,
         locals={"i_report": i4, "i_obs": i4, "i_radius": i4, "i_layer": i4, "i_points": i4, "i_cell": i4,
                 "p_equivalent": f4, "denominator": f4, "p": f4})
    def assimilate(state: np.ndarray, state_map: np.ndarray, state_map_inverse: np.ndarray,
                   observations: np.ndarray, observation_type: np.ndarray, reports: np.ndarray,
                   points_in_radius: np.ndarray, weights: np.ndarray, updated: np.ndarray, det: int, rho: float):
        """
        see Algorithm class for documentation of arguments.
        """
        # temporal variables
        n_varlayer = state.shape[1]
        n_ens = state.shape[2]
        n_inv = 1. / (n_ens - 1)
        equivalent = np.empty(n_ens, dtype=np.float32)
        deviation_equivalent_mean = np.empty(n_ens, dtype=np.float32)
        dist = np.empty(1,dtype=np.float32)
        print('')
        print('reports.shape[0] ',reports.shape[0])
        print('points_in_radius.shape[1] ',points_in_radius.shape[1])
        print('n_varlayer ',n_varlayer)
        # observations are processed one by one in the order that they are listed in the reports array
        for i_report in range(reports.shape[0]):
            # all observations in this report are located at this index within the local part of the grid.
            grid_index = reports[i_report, 2]
            assert grid_index != -1
            print('number of obs in report ',reports[i_report, 0], 'is ',reports[i_report, 1])
            # loop over all observations in this report
            for i_obs in range(reports[i_report, 0], reports[i_report, 0] + reports[i_report, 1]):
                obs_layer = int(observations[i_obs,2])
                plevel_obs = observations[i_obs,3]/100.0
                log_plevel_obs = math.log(plevel_obs)
                #print('obs_layer for iobs ',i_obs,' is ',obs_layer)
                mlevel_obs = state_map_inverse[obs_layer,1]
                var = state_map_inverse[obs_layer,0]
               # print('obs: ',[i_report,i_obs,var,mlevel_obs,plevel_obs])
                #print('test for ',i_obs,' is ',mlevel_obs)
                #mlevel_obs = state_map_inverse[obs_layer,1]
                #print('mlevel for iobs ',i_obs, ' is ',mlevel_obs)
                # get model equivalents for the given observation and the mean which is later used for covariances
                # for observation on model levels, model_equivalent returns just the corresponding gird cell.
                equivalent_mean = model_equivalent(state, state_map, grid_index, observations, observation_type, i_obs,
                                 equivalent, deviation_equivalent_mean)

                # calculate innovation from observation value[i_obs, 0] and observation error[i_obs, 0]
                innovation = observations[i_obs, 0] - equivalent_mean
               
                # calculate variance of model equivalent
                p_equivalent = rho*np.sum(deviation_equivalent_mean**2) * n_inv
                denominator = 1.0 / (p_equivalent + observations[i_obs, 1]**2)
                denominator_ens = 1.0 + (observations[i_obs, 1]**2*denominator)**0.5
                # loop over all grid cells and all variables that are within the localization radius
                # This loop runs in parallel if NUMBA_NUM_THREADS is larger than 1.
                i_points = reports[i_report, 3]
                for i_radius in prange(points_in_radius.shape[1]):
                 
                    # the number of points for each observation is not constant. stop the loop as soon as we reach
                    # a grid cell index of -1
                    i_cell = points_in_radius[i_points, i_radius]
                    if i_cell == -1:
                        #print('number of points in in radius for obs ',i_obs,' is: ',  i_radius)
                        continue

                    # mark the current point as updated. This will cause updates of overlapping areas between processors
                    updated[i_cell] = 1

                    # loop over all layers of the state, this is also a loop over all variables as variables are stacked
                    # on top of each other in the state variable.
                    for i_layer in range(n_varlayer):
                       # print('obs: ',[i_obs,var,mlevel_obs],'layer: ',[i_layer,state_map_inverse[i_layer,0],state_map_inverse[i_layer,1]])
               #         dist = state_map_inverse[i_layer,1]- mlevel_obs
               #         if dist < 0:
               #             dist = - dist
                        
               #         if dist > 0.1:
               #             continue
               #         print('observation ',i_obs,' with layer ',i_layer,' gone through')
                        mlevel_state = state_map_inverse[i_layer,1]
                       
                        # calculate covariance between model equivalent and the current location in the state
                        p = rho * covariance(state, i_cell, i_layer, deviation_equivalent_mean) * weights[i_points, i_radius] #* vertical_loc[dist]
                        #print('loc value for ',i_points,' is ',weights[i_points, i_radius])
                        # update the state at the current location
                        for i_ens in range(n_ens):
                            i_layer_p = state_map[251,0] + mlevel_state
                            plevel_state = state[i_cell, i_layer_p, i_ens]/100.0
                            log_plevel_state = math.log(plevel_state)
                            dist[0] = log_plevel_state - log_plevel_obs
                            if dist[0] < 0.0:
                                dist[0] = - dist[0]
                            if dist > 2*2.0:
                                continue
                            factor = weights_for_gridpoint(localization_radius = 2.0, distance = dist)[0] 
                            #print('obs: ',[var,mlevel_obs,plevel_obs],' state: ',[state_map_inverse[i_layer,0],state_map_inverse[i_layer,1],i_layer_p,plevel_state])
                            
                            state[i_cell, i_layer, i_ens] += factor * p * denominator * (innovation - deviation_equivalent_mean[i_ens] / denominator_ens)
