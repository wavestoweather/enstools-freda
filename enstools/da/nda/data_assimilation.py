"""
Implementation for the NDA
"""
from enstools.misc import spherical2cartesian
from enstools.mpi import onRank0
from enstools.mpi.logging import log_and_time, log_on_rank
from enstools.mpi.grids import UnstructuredGrid
from enstools.io.reader import expand_file_pattern, read
from typing import Union, List, Dict, Any
from petsc4py import PETSc
from numba import jit, objmode
import numpy as np
import os
import logging
import xarray as xr
from sklearn.neighbors import KDTree


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
        self.grid: UnstructuredGrid = grid
        self.state_file_names: List[str] = None
        self.state_variables: Dict[str, Dict[str, Any]] = {}

        # dataset for observations
        self.observations = {}
        self.obs_kdtree: KDTree = None
        self.obs_coords: np.ndarray = None
        self.localization_radius: float = localization_radius

    def load_state(self, filenames: Union[str, List[str]], member_re: str = None, variables=["P", "QV", "T", "U", "V"]):
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

        variables:
                variables to load from the input files.
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

        # keep the original file names. save_state makes use of them to get file names for output files.
        self.state_file_names = expanded_filenames

        # open the first file to get the dimensions for the state
        log_and_time(f"reading file {expanded_filenames[0]} to get information about the state dimension.", logging.INFO, True, self.comm, 0)
        if onRank0(self.comm):
            ds = read(expanded_filenames[0])
            vertical_layers = ds[variables[0]].shape[1]
            state_shape = (self.grid.ncells, vertical_layers, len(variables), len(expanded_filenames))
            # collect information about the state variables. this is later used for writing files
            for ivar, varname in enumerate(variables):
                self.state_variables[varname] = {
                    "dims": ds[varname].dims,
                    "attrs": ds[varname].attrs,
                    "shape": ds[varname].shape,
                    "index": ivar
                }
            self.state_variables["__names"] = variables
            self.state_variables["__coordinates"] = {}
            for coord in ds.coords:
                self.state_variables["__coordinates"][coord] = ds.coords[coord]
        else:
            state_shape = None
            self.state_variables = None
        state_shape = self.comm_mpi4py.bcast(state_shape, root=0)
        self.state_variables = self.comm_mpi4py.bcast(self.state_variables, root=0)
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
                ds = None
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

    def save_state(self, output_folder: str, member_folder: str = None):
        """
        Collect the distributed state back from all processors and store it back to individual files (one per member).
        The original file names are used.

        Parameters
        ----------
        output_folder:
                folder for all output files. Without member_folder, all files are written into the same folder.

        member_folder:
                A format string for member-sub-folders. The format is used for python string formatting and should
                contain one integer place. Example: 'm%03d'.
        """
        log_and_time(f"DataAssimilation.save_state", logging.INFO, True, self.comm)
        # get a list of variables with the original order
        variables = self.state_variables["__names"]

        # check the exisitence of the output folder.
        if onRank0(self.comm) and not os.path.exists(output_folder):
            raise IOError(f"output folder not found: {output_folder}")

        # create names for output file from the input filenames
        output_files = []
        for one_file in self.state_file_names:
            one_output_file = os.path.join(output_folder, os.path.basename(one_file))
            base, ext = os.path.splitext(one_output_file)
            one_output_file = base + ".nc"
            if one_output_file in output_files:
                raise IOError("names of output files are not unique. Try to use the member_folder argument!")
            output_files.append(one_output_file)

        # loop over all files. Every rank writes one file
        for ifile in range(0, len(output_files), self.mpi_size):
            if ifile + self.mpi_rank < len(output_files):
                one_filename = output_files[ifile + self.mpi_rank]
                log_and_time(f"writing file {one_filename}", logging.INFO, True, self.comm, self.mpi_rank)
                ds = xr.Dataset()
                # restore coordinates
                for coord in self.state_variables["__coordinates"]:
                    ds.coords[coord] = self.state_variables["__coordinates"][coord]
                ds.attrs["ensemble_member"] = ifile + self.mpi_rank + 1
                rank_has_data = True
            else:
                one_filename = None
                rank_has_data = False
                ds = None
            rank_has_data = self.comm_mpi4py.allgather(rank_has_data)

            # download variables in a loop over all variables
            for ivar, varname in enumerate(variables):
                # for now, download only one variable at a time.
                for rank in range(self.mpi_size):
                    # skip ranks that have no data anymore
                    if not rank_has_data[rank]:
                        continue
                    # the destination downloads data, all other receive an empty array.
                    values = self.grid.gatherData("state", dest=rank, part=(slice(None), ivar, ifile + rank))

                    if rank == self.mpi_rank:
                        values = values.transpose().reshape(self.state_variables[varname]["shape"])
                        ds[varname] = xr.DataArray(values, dims=self.state_variables[varname]["dims"],
                                                   name=varname, attrs=self.state_variables[varname]["attrs"])

            # every process now writes the content of one member to disk
            if one_filename is not None:
                # actually store the file on disk
                ds.to_netcdf(one_filename, engine="scipy")
                log_and_time(f"writing file {one_filename}", logging.INFO, False, self.comm, self.mpi_rank)

        log_and_time(f"DataAssimilation.save_state", logging.INFO, False, self.comm)

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
            #self.obs_kdtree = scipy.spatial.cKDTree(self.obs_coords)
            self.obs_kdtree = KDTree(self.obs_coords)

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
            indices = self.obs_kdtree.query_radius(coords, r=radius)
            neighbours[:indices[0].size] = indices[0]
            return indices[0].size

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
            - add this report to the non-overlapping set and remove them from the list of all reports
            - find all neighbours and blacklist them
            - continue until the end of the reports-array is reached. One set is now ready.
            - clear the blacklist
            - start over again with the first observation not yet used.
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
            n_reports = coords.shape[0]
            report_set_indices = np.empty(n_reports, dtype=np.int32)

            # create a temporal array used to store all neighbour indices. Worst case is, that all reports
            # are neighbours
            neighbour_indices = np.empty(n_reports, dtype=np.int32)

            # the list of the report sets is initially large enough to contain all reports. Only the used part is
            # later returned.
            report_set_list = np.empty((n_reports, 2), dtype=np.int32)

            # we start with all reports in an unused state
            unused_reports = np.ones(n_reports, dtype=np.int8)
            available_reports = np.empty(n_reports, dtype=np.int8)
            unused_number = n_reports
            unused_first = 0
            unused_last = n_reports
            available_first = 0

            # loop over all reports until they all have been used.
            current_report = 0
            current_report_set = 0
            next_report = 0
            next_report_coords = np.empty((1, 3), dtype=np.float32)
            while unused_number > 0:
                first_report_in_set = current_report

                # put all unused reports in a list of available reports for the next report set
                is_first = True
                for nr in range(unused_first, unused_last):
                    if unused_reports[nr] == 1:
                        if is_first:
                            unused_first = nr
                            available_first = nr
                            is_first = False
                        available_reports[nr] = 1
                        unused_last = nr
                available_number = unused_number
                unused_last += 1

                # loop over all reports that are not yet part of the current report set or blacklisted
                while available_number > 0:
                    # find the first usable report (not overlapping with localization radii of other reports, not used)
                    for nr in range(available_first, unused_last):
                        if available_reports[nr] == 1:
                            next_report = nr
                            available_reports[nr] = 0
                            available_first = nr + 1
                            available_number -= 1
                            break

                    # remove this report from the list of not yet used reports.
                    unused_reports[next_report] = 0
                    unused_number -= 1
                    report_set_indices[current_report] = next_report
                    current_report += 1

                    # find all neighbours
                    next_report_coords[0, :] = coords[next_report, :]
                    with objmode(n_neighbours="int64"):
                        n_neighbours = __get_obs_in_radius(next_report_coords, 2.1 * radius, neighbour_indices)

                    # blacklist all observations in the circle, but use 100% overlaps (reports that are assigned
                    # to the same grid cell)
                    for j in range(n_neighbours):
                        # skip all observations that are already used or blacklisted for this round.
                        if available_reports[neighbour_indices[j]] == 0:
                            continue
                        # reports are assigned to the some gird cell in the global grid. They can be processed together.
                        if grid_indices[neighbour_indices[j]] == grid_indices[next_report]:
                            unused_reports[neighbour_indices[j]] = 0
                            unused_number -= 1
                            report_set_indices[current_report] = neighbour_indices[j]
                            current_report += 1
                        # all others are not available for this report set.
                        available_reports[neighbour_indices[j]] = 0
                        available_number -= 1

                # store information about this reports set
                report_set_list[current_report_set, 0] = first_report_in_set
                report_set_list[current_report_set, 1] = current_report - first_report_in_set
                current_report_set += 1

            # the result array is only a subset of the original report_set_list, which is large enough for all reports.
            report_sets = report_set_list[:current_report_set, :]

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
