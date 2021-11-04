"""
Some support routines for feedback files in idealized experiments.
"""
from enstools.io import read
from enstools.interpolation import model2pressure
from enstools.misc import spherical2cartesian
from typing import Union, List, Dict
import xarray as xr
import numpy as np
import scipy.spatial
import logging
from enum import Enum
import pdb

# ---------------------------------------------- tables from documentation ------------------------------------------- #
# see http://www2.cosmo-model.org/content/model/documentation/core/cosmoFeedbackFileDefinition.pdf
# and https://gitlab.physik.uni-muenchen.de/Leonhard.Scheck/kendapy/blob/master/ekf.py

tables = {'obstypes': {1: 'SYNOP', 2: 'AIREP', 3: 'SATOB', 4: 'DRIBU', 5: 'TEMP', 6: 'PILOT', 7: 'SATEM', 8: 'PAOB',
                       9: 'SCATT', 10: 'RAD', 11: 'GPSRO', 12: 'GPSGB', 13: 'RADAR'},
          'codetypes': {0: 'UNKNOWN0', 145: 'UNKNOWN145', 11: 'SRSCD', 14: 'ATSCD', 21: 'AHSCD', 24: 'ATSHS',
                        140: 'METAR', 110: 'GPS', 141: 'AIRCD', 41: 'CODAR', 144: 'AMDAR', 87: 'CLPRD', 88: 'STBCD',
                        90: 'AMV', 165: 'DRBCD', 64: 'TESAC', 35: 'LDTCD', 36: 'SHTCD', 135: 'TDROP', 37: 'TMPMB',
                        32: 'LDPCD', 33: 'SHPCD', 38: 'PLTMB', 210: 'ATOVS', 132: 'WP_EU', 133: 'RA_EU', 134: 'WP_JP',
                        136: 'PR_US', 137: 'RAVAD', 218: 'SEVIR', 123: 'ASCAT', 122: 'QSCAT', 216: 'AIRS', 217: 'IASI'},
          'r_states': {0: 'ACCEPTED', 1: 'ACTIVE', 3: 'MERGED', 5: 'PASSIVE', 7: 'REJECTED', 9: 'PAS REJ',
                       11: 'OBS ONLY', 13: 'DISMISS'},
          'r_flags': {2: 'SUSP LOCT', 3: 'TIME', 4: 'AREA', 8: 'PRACTICE', 9: 'DATASET', 1: 'BLACKLIST', 5: 'HEIGHT',
                      6: 'SURF', 7: 'CLOUD', 16: 'GROSS', 0: 'OBSTYPE', 10: 'REDUNDANT', 11: 'FLIGHTTRACK', 12: 'MERGE',
                      13: 'THIN', 14: 'RULE', 17: 'NO BIASCOR', 15: 'OBS ERR', 19: 'NO OBS', 18: 'FG', 21: 'FG LB',
                      20: 'OPERATOR', 32: 'NONE'},
          'varnames': {0: 'NUM', 3: 'U', 4: 'V', 8: 'W', 1: 'Z', 57: 'DZ', 9: 'PWC', 28: 'TRH', 29: 'RH', 58: 'RH2M',
                       2: 'T', 59: 'TD', 39: 'T2M', 40: 'TD2M', 11: 'TS', 30: 'PTEND', 60: 'W1', 61: 'WW', 62: 'VV',
                       63: 'CH', 64: 'CM', 65: 'CL', 66: 'NHcbh', 67: 'NL', 93: 'NM', 94: 'NH', 69: 'C', 70: 'NS',
                       71: 'SDEPTH', 72: 'E', 79: 'TRTR', 80: 'RR', 81: 'JJ', 87: 'GCLG', 91: 'N', 92: 'SFALL',
                       110: 'PS', 111: 'DD', 112: 'FF', 118: 'REFL', 119: 'RAWBT', 120: 'RADIANCE', 41: 'U10M',
                       42: 'V10M', 7: 'Q', 56: 'VT', 155: 'VN', 156: 'HEIGHT', 157: 'FLEV', 192: 'RREFL', 193: 'RADVEL',
                       128: 'PDELAY', 162: 'BENDANG', 252: 'IMPPAR', 248: 'REFR', 245: 'ZPD', 246: 'ZWD', 247: 'SPD',
                       242: 'GUST', 251: 'P', 243: 'TMIN', 237: 'UNKNOWN237', 238: 'UNKNOWN238', 239: 'UNKNOWN239'},
          'fullvarnames': {0: 'NUM ordinal (channel) number', 3: 'U m/s u-component of wind',
                           4: 'V m/s v-component of wind', 8: 'W m/s vertical velocity', 1: 'Z (m/s)**2 geopotential',
                           57: 'DZ (m/s)**2 thickness', 9: 'PWC kg/m**2 precipitable water content',
                           28: 'TRH 0..1 transformed relative humidity', 29: 'RH 0..1 relative humidity',
                           58: 'RH2M 0..1 2 metre relative humidity', 2: 'T K upper air temperature',
                           59: 'TD K upper air dew point', 39: 'T2M K 2 metre temperature',
                           40: 'TD2M K 2 metre dew point', 11: 'TS K surface temperature',
                           30: 'PTEND Pa/3h pressure tendency', 60: 'W1 WMO 020004 past weather',
                           61: 'WW WMO 020003 present weather', 62: 'VV m visibility',
                           63: 'CH WMO 020012 type of high clouds', 64: 'CM WMO 020012 type of middle clouds',
                           65: 'CL WMO 020012 type of low clouds', 66: 'NH m cloud base height',
                           67: 'NL WMO 020011 low cloud amount', 93: 'NM WMO 020011 medium cloud amount',
                           94: 'NH WMO 020011 high cloud amount', 69: 'C WMO 500 additional cloud group type',
                           70: 'NS WMO 2700 additional cloud group amount', 71: 'SDEPTH m snow depth',
                           72: 'E WMO 020062 state of ground', 79: 'TRTR h time period of information',
                           80: 'RR kg/m**2 precipitation amount', 81: 'JJ K maximum temperature',
                           87: 'GCLG Table 6 general cloud group', 91: 'N WMO 020011 total cloud amount',
                           92: 'SFALL m 6h snow fall', 110: 'PS Pa surface (station) pressure',
                           111: 'DD degree wind direction', 112: 'FF m/s wind force', 118: 'REFL 0..1 reflectivity',
                           119: 'RAWBT K brightness temperature', 120: 'RADIANCE W/sr/m**3 radiance',
                           41: 'U10M m/s 10m u-component of wind', 42: 'V10M m/s 10m v-component of wind',
                           7: 'Q kg/kg specific humidity', 56: 'VT K virtual temperature',
                           155: 'VN CTH m cloud top height', 156: 'HEIGHT m height', 157: 'FLEV m nominal flight level',
                           192: 'RREFL Db radar reflectivity', 193: 'RADVEL m/s radial velocity',
                           128: 'PDELAY m atmospheric path delay', 162: 'BENDANG rad bending angle',
                           252: 'IMPPAR m impact parameter', 248: 'REFR refractivity', 245: 'ZPD zenith path delay',
                           246: 'ZWD zenith wet delay', 247: 'SPD slant path delay', 242: 'GUST m/s wind gust',
                           251: 'P Pa pressure', 243: 'TMIN K minimum temperature'},
          'veri_run_types': {0: 'FORECAST', 1: 'FIRSTGUESS', 2: 'PREL ANA', 3: 'ANALYSIS', 4: 'INIT ANA', 5: 'LIN ANA'},
          'veri_run_classes': {0: 'HAUPT', 1: 'VOR', 2: 'ASS', 3: 'TEST'},
          'veri_ens_member_names': {0: 'ENS MEAN', -1: 'DETERM', -2: 'ENS SPREAD', -3: 'BG ERROR', -4: 'TALAGRAND',
                                    -5: 'VQC WEIGHT', -6: 'MEMBER', -7: 'ENS MEAN OBS'},
          # add aliases for names used in the model
          'varname_aliases': {"QV": "Q", "qv": "Q","v": "V","u": "U","pres": "P","temp": "T"},
          # reverse mapping between names and variable numbers
          'name2varno': {}
          }

# create a reverse mapping between variable names and variable type numbers
for one_number, one_name in tables["varnames"].items():
    tables["name2varno"][one_name] = one_number
for alias, original in tables["varname_aliases"].items():
    tables["name2varno"][alias] = tables["name2varno"][original]

# types of levels we support
class LevelType(Enum):
    PRESSURE = tables['name2varno']['P']
    MODEL_LEVEL = tables['name2varno']['NUM']

########################################################################################################################


class FeedbackFile:
    """
    A minimal implementation of a feedback file. Only fields used by NDA are created!
    """

    def __init__(self, filename: str = None, gridfile: Union[str, xr.Dataset] = None):
        """
        Read content of an existing file.

        Parameters
        ----------
        filename:
                existing feedback file.

        gridfile:
                reference grid used in DA. indices stored in the file will be taken from this grid. Some operations,
                like adding new observations, are only available with a grid file.
        """
        # load an existing file
        if filename is not None:
            # read the file completely into the memory and close it afterwards.
            ds = read(filename, in_memory=True)
            self.data: xr.Dataset = ds.copy(deep=True)
            ds.close()
        else:
            self.data: xr.Dataset = xr.Dataset()
            self.data.attrs["n_hdr"] = 0
            self.data.attrs["n_body"] = 0

        # load reference grid
        if gridfile is not None:
            if isinstance(gridfile, xr.Dataset):
                self.grid: xr.Dataset = gridfile
            else:
                self.grid: xr.Dataset = read(gridfile)
        else:
            self.grid: xr.Dataset = None

        # create a kdtree to identify nearest grid points
        self.coords: np.ndarray = spherical2cartesian(self.grid["clon"], self.grid["clat"])
        self.kdtree = scipy.spatial.cKDTree(self.coords)

    def add_observation_from_model_output(self, model_file: Union[str, List[str]],
                                          variables: List[str], error: Dict[str, float],
                                          lon: np.ndarray, lat: np.ndarray, levels: np.ndarray = None,
                                          level_type: LevelType = LevelType.PRESSURE, model_grid: str = None,
                                          perfect: bool = False):
        """

        Parameters
        ----------
        model_file:
                file name ot list of file names to extract data from. These names are forwarded to `enstools.io.read`.

        variables:
                list of variable names to extract. All variables will be placed in one report.

        error:
                dictionary with error per variable, e.g.: {"T": 1.0}.

        lon:
                1-d array with longitude coordinates of observations to take (radians).

        lat:
                1-d array with latitude coordinates of observations to take (radians).

        levels:
                levels at which data should be extracted. The units depends on the value of level_type.

        level_type: {"pressure", "model"}
                meaning of the levels argument and also of the level variable in the created file.

        model_grid: optional
                when the model files are not on the same grid as the reference grid for this feedback file (e.g., a grid
                of a higher resolution nature run), then the grid file of this grid is required.

        perfect:
                if set to true, observations are created without adding a random error. Default: False.
        """
        # read the model files and check the content.
        model = read(model_file)

        # load the grid if required and create a kdtree
        if model_grid is not None:
            m_grid = read(model_grid)
            m_coords = spherical2cartesian(m_grid["clon"], m_grid["clat"])
            m_kdtree = scipy.spatial.cKDTree(m_coords)
        else:
            m_grid = self.grid
            m_coords = self.coords
            m_kdtree = self.kdtree
        m_ncells = m_grid["clon"].shape[0]
        m_clon_deg = m_grid["clon"].values * 180.0 / np.pi
        m_clat_deg = m_grid["clat"].values * 180.0 / np.pi
        # make sure that the model data is on the correct grid
        if not m_grid["clon"].shape[0] == model[variables[0]].shape[-1]:
            raise ValueError(f"grid file and model files do not belong together! Cells in grid: {m_grid['clon'].shape[0]}, cells in model file: {model[variables[0]].shape[-1]}!")

        # find the closest indices in the source model grid for the observation locations
        # at first get the resolution. We use the distance between the first grid point and its closest neighbour.
        m_res = m_kdtree.query(m_coords[0, :], k=2)[0][1]

        # now find the closest indices to all grid points of observations
        o_coords = spherical2cartesian(lon, lat)
        m_dist, m_indices = m_kdtree.query(o_coords, distance_upper_bound=m_res*2)
        # remove indices with no match
        o_valid_indices = np.where(m_indices < m_ncells)
        m_valid_indices = np.unique(m_indices[o_valid_indices])

        # create a vertical interpolator for the selected indices
        if levels is not None:
            if "P" in model:
                pressure_variable = "P"
            elif "pres" in model:
                pressure_variable = "pres"
            else:
                raise ValueError("we need a pressure variable in the model file to extract observations on pressure levels. Supported are: pres, P")
            if level_type == LevelType.PRESSURE:
                p = model[pressure_variable][..., m_valid_indices]
                vert_intpol = model2pressure(p, levels)
            elif level_type == LevelType.MODEL_LEVEL:
                # here we assume that there is a first dimension time.
                if not isinstance(levels, np.ndarray):
                    levels = np.asarray(levels)
                if levels.shape == ():
                    levels = levels.reshape(1)
                vert_intpol = lambda x: np.take(x, levels, axis=1)


        # select the requested points from all model variables and interpolate them to requested levels
        variables_per_gridcell = {}
        o_total_number = 0
        for one_var in variables:
            data = model[one_var][..., m_valid_indices]
            if levels is not None:
                data = vert_intpol(data)
            o_total_number += data.size - np.count_nonzero(np.isnan(data.values))
            variables_per_gridcell[one_var] = data
        logging.info(f"total number of observations: {o_total_number}")

        # empty ds for the new reports and observations
        ds = xr.Dataset()

        # create arrays for observations
        # float32 variables
        for obs_array in ["obs", "e_o", "level", "plevel"]:
            ds[obs_array] = xr.DataArray(data=np.zeros(o_total_number, dtype=np.float32), name=obs_array, dims={"d_body": o_total_number})
        body_obs = ds["obs"].values
        body_e_o = ds["e_o"].values
        body_level = ds["level"].values
        body_plevel = ds["plevel"].values
        # int32 variables
        for obs_array in ["varno", "level_typ"]:
            ds[obs_array] = xr.DataArray(data=np.zeros(o_total_number, dtype=np.int32), name=obs_array, dims={"d_body": o_total_number})
        body_varno = ds["varno"].values
        body_level_typ = ds["level_typ"].values

        # create arrays for reports
        # float32 variables
        for hdr_array in ["lon", "lat"]:
            ds[hdr_array] = xr.DataArray(data=np.zeros(m_valid_indices.size, dtype=np.float32), name=hdr_array, dims={"d_hdr": m_valid_indices.size})
        hdr_lon = ds["lon"].values
        hdr_lat = ds["lat"].values
        # int32 variables
        for hdr_array in ["i_body", "l_body", "n_level", "index_x"]:
            ds[hdr_array] = xr.DataArray(data=np.zeros(m_valid_indices.size, dtype=np.int32), name=hdr_array, dims={"d_hdr": m_valid_indices.size})
        hdr_i_body = ds["i_body"].values
        hdr_l_body = ds["l_body"].values
        hdr_n_level = ds["n_level"].values
        hdr_index_x = ds["index_x"].values

        # check if all variables have errors. If not, assign zeros
        for var in variables:
            if not var in error:
                error[var] = 0.0

        # create reports from all observations at one gridpoint
        current_obs = 0
        offset = self.data.attrs["n_body"]
        name2varno = tables["name2varno"]
        for cell in range(m_valid_indices.size):
            hdr_i_body[cell] = offset + current_obs
            n_level = 0
            for level in range(len(levels)):
                n_obs_in_level = 0
                for var in variables:
                    # use only values that are not nan
                    value = variables_per_gridcell[var].values[0, level, cell]
                    pvalue = variables_per_gridcell[pressure_variable].values[0,level,cell]
                    if np.isnan(value):
                        continue
                    n_obs_in_level += 1
                    # collect all information about this observation. If the observation is not
                    # a perfect observation, add a random error.
                    if perfect:
                        body_obs[current_obs] = value
                    else:
                        body_obs[current_obs] = value + np.random.normal(0, error[var])
                    body_e_o[current_obs] = error[var]
                    body_varno[current_obs] = name2varno[var]
                    body_level[current_obs] = levels[level]
                    body_plevel[current_obs] = pvalue
                    body_level_typ[current_obs] = level_type.value
                    current_obs += 1
                if n_obs_in_level > 0:
                    n_level += 1
            hdr_n_level[cell] = n_level
            hdr_l_body[cell] = offset + current_obs - hdr_i_body[cell]
            hdr_lon[cell] = m_clon_deg[m_valid_indices[cell]]
            hdr_lat[cell] = m_clat_deg[m_valid_indices[cell]]
            hdr_index_x[cell] = m_valid_indices[cell]
        logging.info(f"created {m_valid_indices.size} reports.")

        # put variables into the Dataset. if these values already exist, append new data.
        merged = xr.Dataset()
        merged.attrs = self.data.attrs
        for var in ds.variables:
            if self.data.attrs["n_body"] > 0:
                merged[var] = xr.concat([self.data[var], ds[var]], dim=ds[var].dims[0])
            else:
                merged[var] = ds[var]
        self.data = merged
        self.data.attrs["n_hdr"] += m_valid_indices.size
        self.data.attrs["n_body"] += current_obs
        logging.info(f"new total number of reports in file: {self.data.attrs['n_hdr']}")
        logging.info(f"new total number of observations in file: {self.data.attrs['n_body']}")

    def write_to_file(self, filename: str):
        """
        write content added before into a netcdf file

        Parameters
        ----------
        filename:
                name of the new file
        """
    
        self.data.to_netcdf(filename)
        logging.info(f"write all observations to {filename}")
