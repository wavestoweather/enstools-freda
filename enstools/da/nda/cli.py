"""
This is the Command Line Interface for NDA, the Nameless Data-Assimilation Tool.
"""
from enstools.da.support import FeedbackFile, LevelType
from enstools.mpi import init_petsc
from enstools.mpi.logging import log_on_rank, log_and_time
from enstools.mpi.grids import UnstructuredGrid
from enstools.da.nda import DataAssimilation, Algorithm
from enstools.io import read
import numpy as np
import argparse
import runpy
import logging
import os
import pdb


def da(args):
    """
    run the data assimilation.
    """
    # init petsc library
    comm = init_petsc()

    # start timing of the whole process
    log_and_time("Data Assimilation (da) sub-command", logging.INFO, True, comm, 0)

    # get the algorithm to run.
    # loop over all implementations of the Algorithm class
    algorithm = None
    for one_class in Algorithm.__subclasses__():
        if one_class.__name__ == args.algorithm:
            algorithm = one_class
    if algorithm is None:
        log_on_rank(f"unknown algorithm: {args.algorithm}", logging.INFO, comm, 0)
        exit(-1)
    log_on_rank(f"using algorithm {algorithm.__module__}.{algorithm.__name__}", logging.INFO, comm)

    # create the distributed grid structure
    grid_ds = read(args.grid)
    # TODO: estimate required overlap
    grid = UnstructuredGrid(grid_ds, overlap=25, comm=comm)
    # create the DA object. It makes use of the grid object for communication
    da = DataAssimilation(grid, localization_radius=args.loc_radius * 1000, rho=args.rho, det=int(args.include_det))

    # load the ensemble state into memory
    da.load_state(args.first_guess)

    # load observations
    da.load_observations(args.observations)

    # run the actual data assimilation
    da.run(algorithm)

    # store the updated state back into files
    da.save_state(args.output_folder, args.member_folder)

    # show final timing
    log_and_time("Data Assimilation (da) sub-command", logging.INFO, False, comm, 0, True)


def ff_coords_from_arg(arg: str, valid_min: float = None, valid_max: float = None) -> np.ndarray:
    """
    create an numpy array from coordinates given in the command line.

    Parameters
    ----------
    arg:
            value of the obs-lon or obs-lat or levels argument

    Returns
    -------
    1-d numpy array
    """
    # a range argument
    if ":" in arg:
        parts = arg.split(":")
        if len(parts) != 3:
            logging.error("ranges must be specified as start:stop:step!")
            exit(-1)
        values = np.arange(float(parts[0]), float(parts[1]), float(parts[2]), dtype=np.float32)
    elif "," in arg:
        values = np.asarray(list(map(lambda x: float(x), arg.split(","))), dtype=np.float32)
    else:
        values = np.asarray(float(arg), dtype=np.float32)
    if valid_min is not None and np.any(values < valid_min):
        logging.error(f"invalid value for coordinate: {values.min()}, valid range: {valid_min} to {valid_max}")
        exit(-1)
    if valid_max is not None and np.any(values > valid_max):
        logging.error(f"invalid value for coordinate: {values.max()}, valid range: {valid_min} to {valid_max}")
        exit(-1)
    return values


def ff(args):
    """
    create or manipulate feedback files with observations.
    """
    # init petsc library
    comm = init_petsc()

    # creating feedback files works not in parallel
    if comm.Get_size() > 1:
        logging.error("the ff sub-command is not parallelized! Do not run it with mpirun!")
        exit(-1)

    # start timing of the whole process
    log_and_time("Feedback File (ff) sub-command", logging.INFO, True, comm, 0)

    # check arguments, construct the coordinate arrays first
    if args.obs_loc_type == "1d":
        # create observations for given lon/lat pairs
        if args.obs_lon is None or args.obs_lat is None:
            logging.error("the location type '1d' requires --obs-lon and --obs-lat arguments!")
            exit(-1)
        lons = ff_coords_from_arg(args.obs_lon, valid_min=-180, valid_max=180) / 180.0 * np.pi
        lats = ff_coords_from_arg(args.obs_lat, valid_min=-90, valid_max=90) / 180.0 * np.pi
        if lons.size != lats.size:
            logging.error(f"mismatch in size between longitudes({lons.size}) and latitudes({lats.size})!")
            exit(-1)
    elif args.obs_loc_type == 'mesh':
        # create a regular mesh of observations
        if args.obs_lon is None or args.obs_lat is None:
            logging.error("the location type 'mesh' requires --obs-lon and --obs-lat arguments!")
            exit(-1)
        lons_1d = ff_coords_from_arg(args.obs_lon, valid_min=-180, valid_max=180) / 180.0 * np.pi
        lats_1d = ff_coords_from_arg(args.obs_lat, valid_min=-90, valid_max=90) / 180.0 * np.pi
        lons_2d, lats_2d = np.meshgrid(lons_1d, lats_1d)
        lons = lons_2d.ravel()
        lats = lats_2d.ravel()
    elif args.obs_loc_type == 'reduced':
        # create a reduced gaussian grid with a given number of points between the pole and the equator
        logging.error("the --obs-loc-type reduced is not yet implemented!")
        exit(-1)
    else:
        logging.error(f"unsupported type of locations: {args.obs_loc_type}")
        exit(-1)

    # create a dictionary of errors for each variable
    error_dict = {}
    for one_error in args.errors:
        if ":" not in one_error:
            logging.error("--errors must have the format name:error. E.g., T:1.0")
            exit(-1)
        name, value = one_error.split(":")
        error_dict[name] = float(value)
    for one_variable in args.variables:
        if one_variable not in error_dict:
            logging.error(f"no error for variable {one_variable} given in --errors!")
            exit(-1)

    # get the level type
    if args.level_type == "model":
        level_type = LevelType.MODEL_LEVEL
        levels = np.asarray(ff_coords_from_arg(args.levels), dtype=np.int32)
    elif args.level_type == "pressure":
        level_type = LevelType.PRESSURE
        levels = np.asarray(ff_coords_from_arg(args.levels), dtype=np.float32)
    else:
        logging.error(f"unsupported level type: {args.level_type}")
        exit(-1)

    # create the feedback file object. If the destination file is already there, read it!
    if os.path.exists(args.dest):
        logging.info(f"reading existing destination file {args.dest} ...")
        result = FeedbackFile(filename=args.dest, gridfile=args.grid)
    else:
        logging.info("creating a new feedback file...")
        result = FeedbackFile(filename=None, gridfile=args.grid)

    # adding the observations to the result file
    result.add_observation_from_model_output(model_file=args.source,
                                             variables=args.variables,
                                             error=error_dict,
                                             lon=lons,
                                             lat=lats,
                                             levels=np.asarray(levels),
                                             level_type=level_type,
                                             perfect=args.perfect)

    # write the observation to the output file
    result.write_to_file(args.dest)

    # show final timing
    log_and_time("Feedback File (ff) sub-command", logging.INFO, False, comm, 0)


def main():
    """
    Run the actual commandline interface.
    """
    # equivalent to -m mpi4py. See https://mpi4py.readthedocs.io/en/stable/mpi4py.run.html
    runpy.run_module("mpi4py")

    # parse command line options
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(help="Functionalities of different areas are available as sub-commands.")

    # arguments for the actual data assimilation
    parser_da = subparsers.add_parser("da", help="run the data assimilation.")
    parser_da.add_argument("--first-guess", required=True, nargs="+", help="first guess files to be read as background.")
    parser_da.add_argument("--include-det", type=bool, default=False, help="True if the first member is a deterministic run. Default is False.")
    parser_da.add_argument("--output-folder", required=True, help="folder into which output files are written after the data assimilation is done.")
    parser_da.add_argument("--member-folder", help="for member specific destination folders.")
    parser_da.add_argument("--grid", required=True, help="grid definition file which matches the first-guess files.")
    parser_da.add_argument("--observations", required=True, help="A feedback file created with the 'ff' sub-command containing the observations to assimilate.")
    parser_da.add_argument("--loc-radius", type=int, default=500, help="localization radius in km. Default is 500.")
    parser_da.add_argument("--rho", type=float, default=1.0, help="multiplicative inflation factor. Default is 1.0.")
    parser_da.add_argument("--algorithm", default="Default", help="name of the algorithm to run or name of a python file containing the algorithm to run. Default is 'Default'.")
    parser_da.set_defaults(func=da)

    # arguments for the preparation of feedback files
    parser_ff = subparsers.add_parser("ff", help="create Feedback Files with observations.")
    parser_ff.add_argument("--source", required=True, help="model output file from which the observations should be extracted.")
    parser_ff.add_argument("--grid", required=True, help="grid definition file which matches the source file.")
    parser_ff.add_argument("--dest", required=True, help="destination file in which the observations should be stored. If this file already exists, new oberservations are appended.")
    parser_ff.add_argument("--obs-loc-type", default="1d", choices={"1d", "mesh", "reduced"}, help="Type of description of locations of observations. 1d: --obs-lon and --obs-lat contain sequences of coordinates. Observations are extracted for each lon/lat pair. mesh: a regular mesh spun up by the coordinates given on --obs-lon and --obs-lat. reduced: gaussian grid like distribution of locations.")
    parser_ff.add_argument("--obs-lon", help="longitudinal coordinates of the observations in degrees east. Supported are comma-separated values as well as ranges in the format start:stop:step. As usual for ranges in python, the stop value is not included.")
    parser_ff.add_argument("--obs-lat", help="latitudinal coordinates of the observations in degrees north, formated as --obs-lon.")
    parser_ff.add_argument("--obs-lat-lines", type=int, help="number of latidute lines between pole and equator for obs-loc-type reduced.")
    parser_ff.add_argument("--variables", required=True, nargs="+", help="names of variables. The names must match names from the source file.")
    parser_ff.add_argument("--errors", required=True, nargs="+", help="observation error for each variable. Format: name:error.")
    parser_ff.add_argument("--perfect", required=False, action='store_true', help="if given, no random error is added to the observations.")
    parser_ff.add_argument("--levels", required=True, help="vertical levels to extract. The same levels are extracted for all variables. Comma-separated values or a range as in --obs-lon is expected.")
    parser_ff.add_argument("--level-type", default="model", choices={"model", "pressure"}, help="unit of the levels given in --levels.")
    parser_ff.add_argument("--member-folder", nargs="+", help="for member specific destination folders.")
    parser_ff.set_defaults(func=ff)

    # parse the arguments and run the selected function
    args = parser.parse_args()
    args.func(args)
