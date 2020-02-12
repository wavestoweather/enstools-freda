"""
Implementation for the NDA
"""
from enstools.mpi import onRank0
from enstools.mpi.logging import log_and_time, log_on_rank
from enstools.mpi.grids import UnstructuredGrid
from enstools.io.reader import expand_file_pattern, read
from typing import Union, List
from petsc4py import PETSc
import numpy as np
import os
import logging


class DataAssimilation:
    """
    Data Assimilation Tool
    """
    def __init__(self, grid: UnstructuredGrid, comm: PETSc.Comm):
        """
        Create a new data assimilation context for the given grid.

        Parameters
        ----------
        grid: UnstructuredGrid
                Grid and Data management structure for the data assimilation
        """
        # store the comm object and an mpi4py version of it for internal usage
        self.comm = comm
        self.mpi_rank: int = comm.Get_rank()
        self.mpi_size: int = comm.Get_size()
        self.comm_mpi4py = comm.tompi4py()

        # other internal structures.
        self.grid = grid

    def load_state(self, filenames: Union[str, List[str]], member_re: str = None):
        """
        load the state vector from first guess files. One file per member is expended.

        Parameters
        ----------
        filenames:
                list of file names or pattern for names. When no pattern is given, files are ordered alphabetically
                and are expected to represent one member each.

        member_re:
                regular expression that is used to read the member number for the filename or path.
                Example: r'm(\\d\\d\\d)'.
        """
        log_and_time(f"DataAssimilation.load_state", logging.INFO, True, self.comm)
        # make sure that everyone works on the same list of files
        if onRank0(self.comm):
            # open one file, or multiple files?
            if not isinstance(filenames, (list, tuple)):
                if isinstance(filenames, str):
                    filenames = [filenames]
                else:
                    raise NotImplementedError("unsupported type of argument: %s" % type(filenames))

            # construct a list of all filenames
            expanded_filenames = []
            for filename in filenames:
                # is the filename a pattern?
                files = expand_file_pattern(filename)
                for one_file in files:
                    one_file = os.path.abspath(one_file)
                    expanded_filenames.append(one_file)
            expanded_filenames.sort()
        else:
            expanded_filenames = None
        expanded_filenames = self.comm_mpi4py.bcast(expanded_filenames, root=0)

        # open the first file to get the dimensions for the state
        variables = ["pres", "q", "t", "u", "v"]
        log_and_time(f"reading file {expanded_filenames[0]} to get information about the state dimension.", logging.INFO, True, self.comm, 0)
        if onRank0(self.comm):
            ds = read(expanded_filenames[0])
            vertical_layers = ds["t"].shape[1]
            state_shape = (self.grid.ncells, vertical_layers, len(variables), len(expanded_filenames))
        else:
            state_shape = None
        state_shape = self.comm_mpi4py.bcast(state_shape, root=0)
        log_on_rank(f"creating the state with {len(variables)} variables, {len(expanded_filenames)} members and a shape of {state_shape}.", logging.INFO, self.comm, 0)

        # create the state variable
        self.grid.addVariable("state", shape=state_shape)
        log_and_time(f"reading file {expanded_filenames[0]} to get information about the state dimension.", logging.INFO, False, self.comm, 0)

        # loop over all files. Every rank reads one files
        for ifile in range(0, len(expanded_filenames), self.mpi_size):
            if ifile + self.mpi_rank < len(expanded_filenames):
                one_filename = expanded_filenames[ifile + self.mpi_rank]
                log_and_time(f"reading file {one_filename}", logging.INFO, True, self.comm, 0)
                ds = read(one_filename)
                rank_has_data = True
            else:
                one_filename = None
                rank_has_data = False
            rank_has_data = self.comm_mpi4py.allgather(rank_has_data)

            # upload variables in a loop over all variables
            for ivar, varname in enumerate(variables):
                if ds is not None:
                    log_and_time(f"reading variable {varname}", logging.INFO, True, self.comm, -1)
                    values = ds[varname].values[0, ...].transpose()
                    log_and_time(f"reading variable {varname}", logging.INFO, False, self.comm, -1)
                else:
                    values = np.empty(0, PETSc.RealType)

                # for now, upload only one variable at a time.
                for rank in range(self.mpi_size):
                    # skip ranks that have no data anymore
                    if not rank_has_data[rank]:
                        continue
                    # the source uploads data, all other receive only.
                    if rank == self.mpi_rank:
                        self.grid.scatterData("state", values=values, source=rank, part=(slice(None), ivar, ifile + rank))
                    else:
                        self.grid.scatterData("state", values=np.empty(0, PETSc.RealType), source=rank, part=(slice(None), ivar, ifile + rank))

            if one_filename is not None:
                log_and_time(f"reading file {one_filename}", logging.INFO, False, self.comm, 0)

        # update ghost values of the complete state
        self.grid.updateGhost("state")
        log_and_time(f"DataAssimilation.load_state", logging.INFO, False, self.comm)
