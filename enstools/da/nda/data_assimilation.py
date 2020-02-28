"""
Implementation for the NDA
"""
from enstools.misc import spherical2cartesian, distance
from enstools.da.nda.algorithms import Algorithm
from enstools.da.support import feedback_file
from enstools.mpi import onRank0, isGt1
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
import inspect


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

        # create a kd-tree for all grid points on the current processor
        self.local_kdtree: KDTree = KDTree(self.grid.getLocalArray("coordinates_cartesian"))

    def load_state(self, filenames: Union[str, List[str]], member_re: str = None, variables=None):
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
                variables to load from the input files. If None, then all variables are with a time dimension are
                loaded.
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
            # only specific variables or all variables?
            if variables is None:
                variables = []
                for var in ds.variables:
                    if len(ds[var].dims) >= 2 and ds[var].dims[0] == "time":
                        variables.append(var)
            # collect information about the state variables. this is later used for writing files
            layer_start = 0
            for ivar, varname in enumerate(variables):
                n_layers = 1
                if len(ds[varname].shape) == 3:
                    n_layers = ds[varname].shape[1]
                self.state_variables[varname] = {
                    "dims": ds[varname].dims,
                    "attrs": ds[varname].attrs,
                    "shape": ds[varname].shape,
                    "index": ivar,
                    "layer_start": layer_start,
                    "layer_size": n_layers
                }
                layer_start += n_layers
            self.state_variables["__names"] = variables
            self.state_variables["__coordinates"] = {}
            for coord in ds.coords:
                self.state_variables["__coordinates"][coord] = ds.coords[coord]
            state_shape = (self.grid.ncells, layer_start, len(expanded_filenames))
        else:
            state_shape = None
            self.state_variables = None
        state_shape = self.comm_mpi4py.bcast(state_shape, root=0)
        self.state_variables = self.comm_mpi4py.bcast(self.state_variables, root=0)
        log_on_rank(f"creating the state with {len(self.state_variables['__names'])} variables, {len(expanded_filenames)} members and a shape of {state_shape}.", logging.INFO, self.comm, 0)

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
            for ivar, varname in enumerate(self.state_variables['__names']):
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
                    layer_start = self.state_variables[varname]["layer_start"]
                    layer_end = layer_start + self.state_variables[varname]["layer_size"]
                    if layer_end > layer_start + 1:
                        part = (slice(layer_start, layer_end), ifile + rank)
                    else:
                        part = (layer_start, ifile + rank)
                    if rank == self.mpi_rank:
                        self.grid.scatterData("state", values=values, source=rank, part=part)
                    else:
                        self.grid.scatterData("state", values=np.empty(0, PETSc.RealType), source=rank, part=part)

            if one_filename is not None:
                log_and_time(f"reading file {one_filename}", logging.INFO, False, self.comm, 0)

        # update ghost values of the complete state
        self.grid.updateGhost("state")
        log_and_time(f"DataAssimilation.load_state", logging.INFO, False, self.comm)

    def get_state_variable(self, varname: str) -> np.ndarray:
        """
        get a part the the state.

        Parameters
        ----------
        varname:
                name of the variable to get

        Returns
        -------
        view onto a part of the state, not a copy!
        """
        # is it a known variable?
        if not varname in self.state_variables["__names"]:
            raise ValueError(f"the variable {varname} is not part of the state. Known parts are {', '.join(self.state_variables['__names'])}")

        # get the local part of the state array
        layer_start = self.state_variables[varname]["layer_start"]
        layer_end = self.state_variables[varname]["layer_size"] + layer_start
        if layer_end > layer_start + 1:
            part = (slice(None), slice(layer_start, layer_end), slice(None))
        else:
            part = (slice(None), layer_start, slice(None))
        data = self.grid.getLocalArray("state")[part]
        return data

    def backup_state(self) -> np.ndarray:
        """
        This function returns a copy of the local partition of the state. It can be used in interactive environments
        to test the same algorithm multiple times on the same ensemble state.

        Returns
        -------
        np.ndarray:
                copy of the ensemble state
        """
        return self.grid.getLocalArray("state").copy()

    def restore_state(self, state: np.ndarray):
        """
        restore a copy of the state taken before with `backup_state`.

        Parameters
        ----------
        state: np.ndarray
                copy of the state taken with `backup_state`. The dimensionality has to match the current state.
        """
        dest = self.grid.getLocalArray("state")
        if state.shape != dest.shape:
            raise ValueError(f"restore_state: expected shape: {dest.shape}, given shape: {state.shape}")
        dest[:] = state[:]

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
                    layer_start = self.state_variables[varname]["layer_start"]
                    layer_end = self.state_variables[varname]["layer_size"] + layer_start
                    if layer_end > layer_start + 1:
                        part = (slice(layer_start, layer_end), ifile + rank)
                    else:
                        part = (layer_start, ifile + rank)
                    values = self.grid.gatherData("state", dest=rank, part=part)

                    if rank == self.mpi_rank:
                        values = values.transpose().reshape(self.state_variables[varname]["shape"])
                        ds[varname] = xr.DataArray(values, dims=self.state_variables[varname]["dims"],
                                                   name=varname, attrs=self.state_variables[varname]["attrs"])

            # every process now writes the content of one member to disk
            if one_filename is not None:
                # actually store the file on disk
                ds.to_netcdf(one_filename, engine="scipy")
                log_and_time(f"writing file {one_filename}", logging.INFO, False, self.comm, self.mpi_rank)

        log_and_time(f"DataAssimilation.save_state", logging.INFO, False, self.comm, 0, True)

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
            np.histogram(self.observations['report_sets'][:, 1], bins=min(10, self.observations['report_sets'].shape[0]))
        report_set_size_edges = np.floor(report_set_size_edges).astype(np.int32)
        for bin in range(report_set_size_hist.size):
            log_on_rank(f"report sets with {report_set_size_edges[bin]} to {report_set_size_edges[bin+1]} reports: {report_set_size_hist[bin]}",
                    logging.INFO, self.comm)

    def run(self, algorithm: Algorithm):
        """
        Start the actual data assimilation. The assimilate method of the provided algorithm object will be called as
        often as necessary to process all observations that have been added before with load_observations.

        Parameters
        ----------
        algorithm:
                the class or an instance of a class that implement enstools.da.nda.Algorithm.
        """
        # create an instance of the algorithm argument if not done outside
        if inspect.isclass(algorithm):
            algorithm = algorithm()
        log_and_time(f"DataAssimilation.run({algorithm.__class__.__name__})", logging.INFO, True, self.comm, 0, True)

        # create observation array
        n_obs = self.observations["obs"].shape[0]
        observations = np.empty((n_obs, 3), dtype=np.float32)
        observations[:, 0] = self.observations["obs"]
        observations[:, 1] = self.observations["e_o"]
        observations[:, 2] = self.observations["level"]
        observations_type = np.empty((n_obs, 2), dtype=np.int32)
        observations_type[:, 0] = self.observations["varno"]
        observations_type[:, 1] = self.observations["level_typ"]

        # arrays used to describe reports
        report_set_indices = self.observations["report_set_indices"]
        i_body = self.observations["i_body"]
        l_body = self.observations["l_body"]
        index_x = self.observations["index_x"]

        # create the map of the state. This includes indices of variables inside the state variable
        max_var = max(feedback_file.tables["varnames"].keys()) + 1
        state_map = np.empty((max_var, 2), dtype=np.int32)
        state_map[:] = -1
        for one_var in self.state_variables["__names"]:
            if one_var in feedback_file.tables["name2varno"]:
                varno = feedback_file.tables["name2varno"][one_var]
                state_map[varno, 0] = self.state_variables[one_var]["layer_start"]
                state_map[varno, 1] = self.state_variables[one_var]["layer_size"]

        # array for updated indices of the state
        updated = np.empty(self.grid.getLocalArray("state").shape[0], dtype=np.int8)

        # the coordinates of the local part of the grid
        coords = self.grid.getLocalArray("coordinates_cartesian")
        clon = self.grid.getLocalArray("clon")
        clat = self.grid.getLocalArray("clat")

        # here we loop over all non-overlapping sets of reports created before by _calculate_non_overlapping_reports
        for iset in range(self.observations["report_sets"].shape[0]):
            log_and_time(f"working on report set {iset+1} of {self.observations['report_sets'].shape[0]}",
                         logging.INFO, True, self.comm, 0, False)

            # create the arguments for the next call to assimilate
            reports = np.empty((self.observations["report_sets"][iset, 1], 4), dtype=np.int32)
            for ireport in range(self.observations["report_sets"][iset, 1]):
                index_in_report_set_indices = ireport + self.observations["report_sets"][iset, 0]
                reports[ireport, 0] = i_body[report_set_indices[index_in_report_set_indices]]
                reports[ireport, 1] = l_body[report_set_indices[index_in_report_set_indices]]
                # the index_x variable contains global indices and needs to be translated to local indices owned
                # the individual processors.
                if isGt1(self.comm):
                    reports[ireport, 2] = \
                        self.grid._global2local_permutation_indices[index_x[
                            report_set_indices[index_in_report_set_indices]]]
                else:
                    reports[ireport, 2] = index_x[report_set_indices[index_in_report_set_indices]]

            # filter for reports that are processed on other processors.
            if isGt1(self.comm):
                local_reports = np.where(reports[:, 2] > -1)[0]
                reports = reports[local_reports, :]

            # find grid points within the localization radius of each report.
            # at first, get unique indices of reports on the grid
            unique_indices = np.empty(reports.shape[0], dtype=np.int32)
            unique_indices_dict = {}
            for ireport in range(reports.shape[0]):
                if not reports[ireport, 2] in unique_indices_dict:
                    reports[ireport, 3] = len(unique_indices_dict)
                    unique_indices_dict[reports[ireport, 2]] = reports[ireport, 3]
                    unique_indices[reports[ireport, 3]] = reports[ireport, 2]
                else:
                    reports[ireport, 3] = unique_indices_dict[reports[ireport, 2]]
            unique_indices = unique_indices[:len(unique_indices_dict)]

            # only continue if we have local reports
            if unique_indices.shape[0] > 0:
                # this returns an array of array objects, convert to one array
                _affected_points = self.local_kdtree.query_radius(coords[unique_indices, :], r=self.localization_radius)
                _affected_points_max_length = max(list(map(lambda x: x.shape[0], _affected_points)))
                affected_points = np.empty((len(_affected_points), _affected_points_max_length), dtype=np.int32)
                for one_radius in range(len(_affected_points)):
                    affected_points[one_radius, :_affected_points[one_radius].shape[0]] = _affected_points[one_radius]
                    affected_points[one_radius, _affected_points[one_radius].shape[0]:] = -1

                # calculate weights of each affected point
                weigths = np.zeros(affected_points.shape, dtype=np.float32)
                for one_radius in range(len(_affected_points)):
                    lon_of_points = clon[_affected_points[one_radius]]
                    lat_of_points = clat[_affected_points[one_radius]]
                    dist = distance(clat[unique_indices[one_radius]],
                                    lat_of_points,
                                    clon[unique_indices[one_radius]],
                                    lon_of_points)
                    weigths[one_radius, :_affected_points[one_radius].shape[0]] = \
                        algorithm.weights_for_gridpoint(self.localization_radius, dist)
            else:
                affected_points = np.empty((0, 0), dtype=np.int32)
                weigths = np.empty((0, 0), dtype=np.float32)

            # assimilate the observations of the current report set
            log_and_time(f"{algorithm.__class__.__name__}.assimilate()", logging.INFO, True, self.comm, 0, False)
            # only call the assimilate function if we have anything to do. It is possible the one rank is already ready
            # while another rank is still processing.
            updated[:] = 0
            if unique_indices.shape[0] > 0:
                algorithm.assimilate(self.grid.getLocalArray("state"), state_map,
                                     observations, observations_type, reports, affected_points, weigths, updated)
            self.comm.barrier()
            log_and_time(f"{algorithm.__class__.__name__}.assimilate()", logging.INFO, False, self.comm, 0, False)

            # update overlapping regions in both directions
            local_updated = updated.nonzero()[0]
            self.grid.updateGhost("state", local_indices=local_updated, direction="O2G")
            self.grid.updateGhost("state", local_indices=local_updated, direction="G2O")

            log_and_time(f"working on report set {iset + 1} of {self.observations['report_sets'].shape[0]}",
                         logging.INFO, False, self.comm, 0, False)

        log_and_time(f"DataAssimilation.run({algorithm.__class__.__name__})", logging.INFO, False, self.comm, 0, True)
