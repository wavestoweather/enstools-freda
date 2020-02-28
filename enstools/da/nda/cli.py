"""
This is the Command Line Interface for NDA, the Nameless Data-Assimilation Tool.
"""
from enstools.mpi import init_petsc
from enstools.mpi.logging import log_on_rank, log_and_time
from enstools.mpi.grids import UnstructuredGrid
from enstools.da.nda import DataAssimilation, Algorithm
from enstools.io import read
import argparse
import runpy
import logging


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
    log_on_rank(f"using algorithm {algorithm}", logging.INFO, comm)

    # create the distributed grid structure
    grid_ds = read(args.grid)
    # TODO: estimate required overlap
    grid = UnstructuredGrid(grid_ds, overlap=25, comm=comm)

    # create the DA object. It makes use of the grid object for communication
    da = DataAssimilation(grid, localization_radius=args.loc_radius * 1000)

    # load the ensemble state into memory
    da.load_state(args.first_guess)

    # load observations
    da.load_observations(args.observations)

    # run the actual data assimilation
    da.run(algorithm)

    # store the updated state back into files
    da.save_state(args.output_folder)

    # show final timing
    log_and_time("Data Assimilation (da) sub-command", logging.INFO, False, comm, 0, True)


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
    parser_da.add_argument("--output-folder", required=True, help="folder into which output files are written after the data assimilation is done.")
    parser_da.add_argument("--grid", required=True, help="grid definition file which matches the first-guess files.")
    parser_da.add_argument("--observations", required=True, help="A feedback file created with the 'ff' sub-command containing the observations to assimilate.")
    parser_da.add_argument("--loc-radius", type=int, default=1000, help="localization radius in km. Default is 1000.")
    parser_da.add_argument("--algorithm", default="Default", help="name of the algorithm to run or name of a python file containing the algorithm to run. Default is 'Default'.")

    parser_da.set_defaults(func=da)

    # arguments for the preparation of feedback files
    parser_ff = subparsers.add_parser("ff", help="create Feedback Files with observations.")

    # parse the arguments and run the selected function
    args = parser.parse_args()
    args.func(args)
