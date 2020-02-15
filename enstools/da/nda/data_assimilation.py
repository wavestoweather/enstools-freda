"""
Implementation for the NDA
"""
from enstools.misc import spherical2cartesian
from enstools.mpi import onRank0
from enstools.mpi.logging import log_and_time, log_on_rank
from enstools.mpi.grids import UnstructuredGrid
from enstools.io.reader import expand_file_pattern, read
from typing import Union, List, Tuple
from petsc4py import PETSc
from numba import jit, objmode
import numba.typed
import numba
import numpy as np
import os
import logging
import scipy.spatial


class DataAssimilation:
    """
    Data Assimilation Tool
    """
    def __init__(self, grid: UnstructuredGrid, localization_radius: float = 1000000.0, comm: PETSc.Comm = None):
        """
        Create a new data assimilation context for the given grid.

        Parameters
        ----------
        grid: UnstructuredGrid
                Grid and Data management structure for the data assimilation

        localization_radius:
                radius of localization to be used in the assimilation. Default = 1000000.0 m

        comm:
                MPI communicator. If not given, then the communicator of the grid is used.
        """
        # store the comm object and an mpi4py version of it for internal usage
        if comm is None:
            self.comm = grid.comm
        else:
            self.comm = comm
        self.mpi_rank: int = self.comm.Get_rank()
        self.mpi_size: int = self.comm.Get_size()
        self.comm_mpi4py = self.comm.tompi4py()

        # other internal structures.
        self.grid = grid

        # dataset for observations
        self.observations = {}
        self.obs_kdtree: scipy.spatial.cKDTree = None
        self.obs_coords: np.ndarray = None
        self.localization_radius = localization_radius

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
        variables = ["P", "QV", "T", "U", "V"]
        log_and_time(f"reading file {expanded_filenames[0]} to get information about the state dimension.", logging.INFO, True, self.comm, 0)
        if onRank0(self.comm):
            ds = read(expanded_filenames[0])
            vertical_layers = ds["T"].shape[1]
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

    def load_observations(self, filename: str):
        """
        load observations from a feedback file.

        Parameters
        ----------
        filename:
                name of a netcdf feedback file created by enstools.da.support.FeedbackFile or any compatible tool.
        """
        # read the input file on the first rank and distribute numpy arrays to all ranks
        log_and_time(f"DataAssimilation.load_observations({filename})", logging.INFO, True, self.comm, 0, False)
        log_and_time(f"loading and distributing data", logging.INFO, True, self.comm, 0, False)
        if onRank0(self.comm):
            ff = read(filename)
            variables = {}
            for var in ff.variables:
                variables[var] = (ff[var].shape, ff[var].dtype)
        else:
            ff = None
            variables = None
        variables = self.comm_mpi4py.bcast(variables)
        for var in variables:
            if onRank0(self.comm):
                self.observations[var] = ff[var].values
            else:
                self.observations[var] = np.empty(variables[var][0], dtype=variables[var][1])
            self.comm_mpi4py.Bcast(self.observations[var], root=0)
        log_and_time(f"loading and distributing data", logging.INFO, False, self.comm, 0, False)

        # create a kd-tree from all reports on the first rank. The first rank will schedule the
        log_and_time(f"splitting observation in non-overlapping subsets", logging.INFO, True, self.comm, 0, False)
        if onRank0(self.comm):
            # calculate cartesian coordinates for all observation reports
            self.obs_coords = spherical2cartesian(self.observations["lon"] / 180.0 * np.pi,
                                                  self.observations["lat"] / 180.0 * np.pi)
            self.obs_kdtree = scipy.spatial.cKDTree(self.obs_coords)

            # calculate sets of observation reports that are not overlapping
            self._calculate_non_overlapping_reports()

        # distribute the non-overlapping sets of reports
        if onRank0(self.comm):
            report_set_sizes = [self.observations["report_sets"].shape, self.observations["report_set_indices"].shape]
        else:
            report_set_sizes = None
        report_set_sizes = self.comm_mpi4py.bcast(report_set_sizes, root=0)
        if not onRank0(self.comm):
            self.observations["report_sets"] = np.empty(report_set_sizes[0], dtype=np.int32)
            self.observations["report_set_indices"] = np.empty(report_set_sizes[1], dtype=np.int32)
        self.comm_mpi4py.Bcast(self.observations["report_sets"], root=0)
        self.comm_mpi4py.Bcast(self.observations["report_set_indices"], root=0)

        log_and_time(f"splitting observation in non-overlapping subsets", logging.INFO, False, self.comm, 0, True)
        log_and_time(f"DataAssimilation.load_observations({filename})", logging.INFO, False, self.comm, 0, False)

    def _calculate_non_overlapping_reports(self):
        """
        Create sets of reports that are not overlapping. The result will be stored in self.observations["report_sets"]
        and self.observations["report_set_indices"].
        """
        # define helper functions
        def __get_obs_in_radius(coords: np.ndarray, radius: float, neighbours: np.ndarray) -> int:
            """
            wrapper for self.obs_kdtree that can be called from numba nopython functions.

            Parameters
            ----------
            coords:
                    array the shape (1, 3). The coordinates of one observation.

            radius:
                    search radius for observations. Should be >= self.localization_radius

            neighbours:
                    return values: indices of neighbours within the radius around coods.

            Returns
            -------
            number of values stored in neighbours
            """
            indices = self.obs_kdtree.query_ball_point(coords, r=radius)
            neighbours[:len(indices)] = indices
            return len(indices)

        @jit(nopython=True)
        def __get_first_unused_not_in_blacklist(unused_reports: set, blacklist: set) -> int:
            """
            Find the first unused report that is not yet blacklisted.

            Parameters
            ----------
            unused_reports:
                    set of all not yet used reports.

            blacklist:
                    set of un-usable reports (due to overlap)

            Returns
            -------
            index of the first unused report.
            """
            for i in unused_reports:
                if not i in blacklist:
                    return i
            # nothing found?
            return -1

        @jit(nopython=True)
        def __get_report_sets(coords: np.ndarray, grid_indices: np.ndarray, radius: float):
            """
            Given the coordinates of all report and the localization radius, non-overlapping subsets of
            reports are generated.

            Idea of the algorithm:
            - start with the first report.
            - find all reports within 2x radius, put them into a blacklist
            - iterate over reports until we find the first one not in the blacklist
            - add this report to the non-overlapping set and to the used reports list
            - find all neighbours and blacklist them
            - continue until the end of the reports-array is reached. One set is now ready.
            - clear the blacklist
            - start over again with the first observation not in the used array.
            - continue until the used array contains all reports.

            Parameters
            ----------
            coords:
                    cartesian coordinates of the observations with shape (nobs, 3)

            grid_indices:
                    indices within the global grid to which a report belongs.

            Returns
            -------
            report_sets, report_set_indices:
                    first element: np.ndarray with (nsets, 2). First value: first report in set, second value: number of
                                   reports in set.
                    second element: np.ndarray with indices of reports.
            """
            # create a array for all reports. This will be the second element of the returned tuple
            # the size of the report_sets is not known in advance and can not be allocated here
            report_set_indices = np.empty(coords.shape[0], dtype=np.int32)

            # create a temporal array used to store all neighbour indices. Worst case is, that all reports
            # are neighbours
            neighbour_indices = np.empty(coords.shape[0], dtype=np.int32)

            # store report sets at first in a typed list. The first element is added here to define the type only.
            report_set_list = [np.empty(2, dtype=np.int32)]
            report_set_list.clear()

            # we use a set to keep track of all indices that have already been used and of those that can not be used
            # in the current report set because of overlap
            used_reports = set()
            used_reports.add(numba.int32(0))
            used_reports.clear()
            blacklist = set()
            blacklist.add(numba.int32(0))
            blacklist.clear()

            # we start with all reports in an unused state
            unused_reports = set()
            for i in range(coords.shape[0]):
                unused_reports.add(i)

            # loop over all reports until they all have been used.
            current_report = 0
            while len(used_reports) < coords.shape[0]:
                report_set_info = np.zeros(2, dtype=np.int32)
                first_report_in_set = current_report

                # loop over all reports
                while True:
                    next_report = __get_first_unused_not_in_blacklist(unused_reports, blacklist)
                    if next_report == -1:
                        break
                    used_reports.add(next_report)
                    unused_reports.remove(next_report)
                    report_set_indices[current_report] = next_report
                    current_report += 1

                    # find all neighbours
                    with objmode(n_neighbours="int64"):
                        n_neighbours = __get_obs_in_radius(coords[next_report, :], 2.1 * radius, neighbour_indices)

                    # blacklist all observations in the circle, but use 100% overlaps
                    for j in range(n_neighbours):
                        # the current report will be one of the first neighbours of the coordinates
                        if neighbour_indices[j] == next_report:
                            continue
                        # reports are assigned to the some gird cell in the global grid. They can be processed together.
                        if grid_indices[neighbour_indices[j]] == grid_indices[next_report]:
                            used_reports.add(neighbour_indices[j])
                            unused_reports.remove(neighbour_indices[j])
                            report_set_indices[current_report] = neighbour_indices[j]
                            current_report += 1
                        # neighbours are close, but not assigned to the same grid cell.
                        else:
                            blacklist.add(neighbour_indices[j])

                # store information about this reports set
                report_set_info[0] = first_report_in_set
                report_set_info[1] = current_report - first_report_in_set
                report_set_list.append(report_set_info)

                # reset the blacklist as the next iteration will start with a clean domain
                blacklist.clear()

            # here we can create the result array and copy the content of the list created above.
            report_sets = np.empty((len(report_set_list), 2), dtype=np.int32)
            for i in range(len(report_set_list)):
                report_sets[i, 0] = report_set_list[i][0]
                report_sets[i, 1] = report_set_list[i][1]

            return report_sets, report_set_indices

        # run the split up function
        self.observations["report_sets"], self.observations["report_set_indices"] = \
            __get_report_sets(self.obs_coords,
            self.observations["index_x"],
            self.localization_radius)

        # show information about the non-overlapping sets
        report_set_size_hist,  report_set_size_edges = \
            np.histogram(self.observations['report_sets'][:, 1])
        report_set_size_edges = np.floor(report_set_size_edges).astype(np.int32)
        for bin in range(report_set_size_hist.size):
            log_on_rank(f"report sets with {report_set_size_edges[bin]} to {report_set_size_edges[bin+1]} reports: {report_set_size_hist[bin]}",
                    logging.INFO, self.comm)

    def run(self):
        """
        start the actual data assimilation.
        """
        pass
