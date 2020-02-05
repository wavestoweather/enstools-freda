import logging
from .logging import log_on_rank, log_and_time
from ..mpi import onRank0, isGt1
from enstools.misc import spherical2cartesien
from petsc4py import PETSc
import numpy as np


class UnstructuredGrid:
    def __init__(self, ds, overlap=0, comm=None):
        """
        create a PETSc DMPlex from an ICON grid definition file. When executed in an MPI
        environement, the Grid is distributed over all processes.

        Parameters
        ----------
        ds: xarray.Dataset
                grid definition dataset.

        overlap: int
                number of overlapping grid points on each processor

        comm: PETSc.Comm
                MPI communicator

        """
        # store the comm object and an mpi4py version of it for internal usage
        self.comm = comm
        self.mpi_rank = comm.Get_rank()
        self.mpi_size = comm.Get_size()
        self.comm_mpi4py = comm.tompi4py()

        # start the grid construction...
        log_and_time("UnstructuredGrid.__init__()", logging.INFO, True, self.comm)
        log_and_time("creating and distributing the PETSc grid", logging.INFO, True, self.comm)
        # the grid definition is only read on the first MPI processor. All other start with an empty grid
        log_and_time("reading coordinates of verticies", logging.INFO, True, self.comm)
        if onRank0(comm):
            # create list of vertices for each cell. The ICON grid definition already contains this list,
            # but with the wrong order of dimensions.
            cells = np.asarray(ds["vertex_of_cell"].transpose(), dtype=PETSc.IntType) -1

            # coordinates of the verticies
            vlon = np.asarray(ds["vlon"], dtype=PETSc.RealType)
            vlat = np.asarray(ds["vlat"], dtype=PETSc.RealType)
            coords = np.empty((vlon.size, 2), dtype=PETSc.RealType)
            coords[:, 0] = vlon
            coords[:, 1] = vlat

            # total number of cells
            self.ncells = cells.shape[0]
        else:
            # empty grid for other processors
            cells = np.zeros((0, 3), dtype=PETSc.IntType)
            coords = np.zeros((0, 2), dtype=PETSc.RealType)
            self.ncells = None

        # communicate the total number of cells
        if isGt1(self.comm):
            self.ncells = self.comm_mpi4py.bcast(self.ncells, root=0)
        log_and_time("reading coordinates of verticies", logging.INFO, False, self.comm)

        # create the grid object
        log_and_time("constructing the global DMPLex structure", logging.INFO, True, self.comm)
        self.plex = PETSc.DMPlex().createFromCellList(2, cells, coords, comm=self.comm)
        # create a copy of the plex that is not distributed. We use that later to construct sections for distributions.
        if isGt1(comm):
            self.plex_non_distributed = self.plex.clone()
            self.plex_non_distributed.setNumFields(1)
        else:
            self.plex_non_distributed = self.plex
        self.plex.setNumFields(1)
        log_and_time("constructing the global DMPLex structure", logging.INFO, False, self.comm)

        # create a section with all grid points on the first processor
        log_and_time("distributing the DMPlex on all processors", logging.INFO, True, self.comm)
        self.sections_on_zero = {}
        self.sections_distributed = {}
        self.scatter_to_zero = {}
        self.scatter_to_zero_is = {}
        self.temporal_vectors_on_zero = {}
        self.temporal_vectors_local = {}
        self.temporal_vectors_global = {}
        self.variables_info = {}
        self.variables = {}
        # create default section with dof=1 on rank=0
        self.__createNonDistributedSection(dof=1)
        self.plex.setSection(self.sections_on_zero[1])

        # distribute over all processes
        if isGt1(comm):
            log_and_time("running partitioner", logging.INFO, True, self.comm)
            part = self.plex.getPartitioner()
            part.setType(part.Type.PARMETIS)
            part.setUp()
            self.sf = self.plex.distribute(overlap=overlap)
            self.__createDistributedSection(dof=1)
            log_and_time("running partitioner", logging.INFO, False, self.comm)

            # create scatter context for all ranks to rank zero
            # distribute the grid indices to get the permutation
            if self.mpi_rank == 0:
                indices = np.arange(0, self.ncells, dtype=PETSc.RealType)
                indices_test = np.zeros((self.ncells, 3), dtype=PETSc.RealType)
                indices_test[:, 0] = indices
                indices_test[:, 1] = indices
                indices_test[:, 2] = indices
            else:
                indices = np.zeros(0, dtype=PETSc.RealType)
                indices_test = indices
            # add global indices as new variable to the grid
            self.addVariable("global_indices", values=indices, comm=self.comm)
            self.addVariable("global_indices_test", values=indices_test, comm=self.comm)
            # get the global form (no ghost points)  of the indices
            self.plex.localToGlobal(self.variables["global_indices"], self.temporal_vectors_global[1])
            # scatter this indices to process zero and and filter out ghost points
            self.scatter_to_zero[1].scatter(self.temporal_vectors_global[1], self.temporal_vectors_on_zero[1])
            if onRank0(comm):
                self.permutation_indices = np.asarray(self.temporal_vectors_on_zero[1].getArray(), dtype=PETSc.IntType)

        log_and_time("distributing the DMPlex on all processors", logging.INFO, False, self.comm)
        log_and_time("creating and distributing the PETSc grid", logging.INFO, False, self.comm)

        # setup a dictionary for variables
        log_and_time("distributing support information", logging.INFO, True, self.comm)

        # store cell center coordinates on the grid as sperical coordinates as well as as cartesian coordinates
        log_and_time("calculating cartesian coordinates", logging.INFO, True, self.comm)
        if onRank0(comm):
            clon = ds["clon"].values
            clat = ds["clat"].values
            coords = spherical2cartesien(lon=clon, lat=clat)
        else:
            coords = np.empty((0, 3))
            clon = np.empty(0)
            clat = np.empty(0)
        log_and_time("calculating cartesian coordinates", logging.INFO, False, self.comm)
        log_and_time("distributing coordinate fields", logging.INFO, True, self.comm)
        self.addVariable("clon", values=clon, comm=self.comm)
        self.addVariable("clat", values=clat, comm=self.comm)
        self.addVariable("coordinates_cartesian", values=coords, comm=self.comm)
        log_and_time("distributing coordinate fields", logging.INFO, False, self.comm)
        log_and_time("distributing support information", logging.INFO, False, self.comm)
        log_and_time("UnstructuredGrid.__init__()", logging.INFO, False, self.comm)

    def __createNonDistributedSection(self, dof):
        """
        create a section on the non-distributed mesh for usage in field distribution.

        Parameters
        ----------
        dof: int
                number of values per grid-point

        Returns
        -------
        new section object, The result is also stored in self.sections_on_zero.
        """
        # create the section object
        new_sec = self.plex_non_distributed.createSection(numComp=1, numDof=[0, 0, dof])
        new_sec.setFieldName(0, "cells")
        new_sec.setUp()
        self.sections_on_zero[dof] = new_sec

        # create a temporal vector using this section
        self.plex_non_distributed.setSection(new_sec)
        self.temporal_vectors_on_zero[dof] = self.plex_non_distributed.createGlobalVector()
        self.temporal_vectors_on_zero[dof].zeroEntries()
        self.temporal_vectors_on_zero[dof].assemble()

        # restore default section
        self.plex_non_distributed.setSection(self.sections_on_zero[1])
        return new_sec

    def __createDistributedSection(self, dof):
        """
        create a distributed section for the given number of DoF.

        Parameters
        ----------
        dof: int

        Returns
        -------
        new section object. The result is also added to self.sections_distributed
        """
        # create a new distributed section based on the non-distributed section on rank zero.
        if dof in self.sections_on_zero:
            # create the distributed section by distributing an empty array
            self.sections_distributed[dof], self.temporal_vectors_local[dof] = \
                self.plex.distributeField(self.sf, self.sections_on_zero[dof],
                                          self.temporal_vectors_on_zero[dof])
            # set the new section as default and create a global vector from it.
            self.plex.setSection(self.sections_distributed[dof])
            self.temporal_vectors_global[dof] = self.plex.createGlobalVec()
            # restore default section
            self.plex.setSection(self.sections_distributed[1])
        else:
            raise ValueError("__createDistributedSection: not yet supported if no non-distributed section has been created before.")

        # create a scatter context for this vector size
        if not dof in self.scatter_to_zero:
            # create the scatter context
            self.scatter_to_zero[dof], temp = PETSc.Scatter().toZero(self.temporal_vectors_global[dof])
            temp.destroy()
        return self.sections_distributed[dof]

    def __getPermutationIS(self, dof):
        """
        create an index-set for permutation after gathering data on rank 0.

        Parameters
        ----------
        dof: int
                degrees of freedom

        Returns
        -------
        IS object
        """
        # on rank 0 create a permutation index
        if onRank0(self.comm):
            # only create the index once.
            if dof in self.scatter_to_zero_is:
                return self.scatter_to_zero_is[dof]
            # for one DoF, the index is created from the 1d-permutation_index
            if dof == 1:
                indices = self.permutation_indices
            else:
                indices = np.empty(self.ncells * dof, dtype=PETSc.IntType)
                for index in range(self.permutation_indices.size):
                    for d in range(dof):
                        indices[index*dof+d] = self.permutation_indices[index] * dof + d
            self.scatter_to_zero_is[dof] = PETSc.IS().createGeneral(indices, comm=PETSc.COMM_SELF)
            return self.scatter_to_zero_is[dof]
        else:
            return None

    def __getDoFforArray(self, array=None, dof=None):
        """
        use the shape of an array to the the degrees of freedom. The shape of the array of only checked on rank zero
        and then communicated to all other ranks

        Parameters
        ----------
        array: np.array
                array with the shape to check. If None, the dof argument is used

        dof: int
                a manually set dof.

        Returns
        -------
        int:
                dof required to store this array in a PETSc vector.
        """
        # get the dof from the values argument
        if array is not None:
            if onRank0(self.comm):
                dof = np.prod(array.shape) // self.ncells
            else:
                dof = -1
            dof = self.comm_mpi4py.bcast(dof, root=0)

        # default dof is one
        if dof is None:
            dof = 1
        return dof

    def addVariable(self, name, values=None, dof=None, comm=None):
        """
        create a new vector on the grid.

        Parameters
        ----------
        name: str
                name of the variable

        values: np.array
                if given, values are directly scattered on the grid. The shape of the variable determines the degree
                of freedom (Dof) of this variable. Alternatively, it is possible to specify the dof argument.

        dof: int
                number of values stored in one point.
        """
        # get the dof from the values argument
        dof = self.__getDoFforArray(values, dof)

        # the first dimension must be the number of cells
        if onRank0(comm) and values is not None and not values.shape[0] == self.ncells:
            raise ValueError(
                f"Variable {name}: shape has not the number of grid cells in the first dimension: {values.shape}")

        # when values are provided, a corresponding section if created
        if values is not None and not dof in self.sections_on_zero:
            self.__createNonDistributedSection(dof)
            if isGt1(comm):
                self.__createDistributedSection(dof)

        # create a new variable with the appropriate section
        if isGt1(comm):
            if dof in self.sections_distributed:
                self.plex.setSection(self.sections_distributed[dof])
                self.variables[name] = self.plex.createLocalVec()
                # restore default section
                if dof != 1:
                    self.plex.setSection(self.sections_distributed[1])
            else:
                raise ValueError("variable without distributed section added!")
        else:
            if dof in self.sections_on_zero:
                self.plex.setSection(self.sections_on_zero[dof])
                self.variables[name] = self.plex.createLocalVec()
                # restore default section
                if dof != 1:
                    self.plex.setSection(self.sections_on_zero[1])
            else:
                raise ValueError("variable without non-distributed section added!")


        # store information about this variable
        if values is not None:
            if isGt1(self.comm):
                partition_sizes = self.temporal_vectors_local[1].getSizes()[0], self.temporal_vectors_global[1].getSizes()[0]
                shape_on_zero = self.comm_mpi4py.bcast(values.shape)
            else:
                partition_sizes = self.ncells, self.ncells
                shape_on_zero = values.shape
            if self.mpi_rank > 0:
                shape_on_zero = list(shape_on_zero)
                shape_on_zero[0] = 0
                shape_on_zero = tuple(shape_on_zero)
            self.variables_info[name] = VariableInfo(dof=dof, shape_on_zero=shape_on_zero, partition_sizes=partition_sizes)
        else:
            self.variables_info[name] = VariableInfo(dof=dof, shape_on_zero=None)

        if values is not None:
            self.scatterData(name, values=values, dof=dof, comm=comm)

    def scatterData(self, name, values=None, dof=None, comm=None):
        """

        Parameters
        ----------
        name
        values

        Returns
        -------

        """
        log_and_time(f"UnstructuredGrid.scatterData({name})", logging.INFO, True, comm)
        # make sure we have data continuous in memory
        if values is not None:
            values = np.require(values, requirements="C")
        # with more than one processor, we need to distribute the data. Otherwise, we just store it locally
        if isGt1(self.comm):
            # on rank 0 write the data into the total grid vector
            dof = self.__getDoFforArray(values, dof)
            if comm is not None and comm.Get_rank() == 0:
                self.temporal_vectors_on_zero[dof].getArray()[:] = values.ravel()
            self.temporal_vectors_on_zero[dof].assemble()

            # distribute the values. Create a new local vector including ghost values for this purpose
            _, self.variables[name] = self.plex.distributeField(sf=self.sf,
                                                                sec=self.sections_on_zero[dof],
                                                                vec=self.temporal_vectors_on_zero[dof],
                                                                newsec=self.sections_distributed[dof])

            #self.plex.localToLocal(self.variables[name], self.variables[name])
            #self.plex.localToGlobal(newvec, self.variables[name])
        else:
            self.variables[name].getArray()[:] = values.ravel()
        log_and_time(f"UnstructuredGrid.scatterData({name})", logging.INFO, False, comm)

    def gatherData(self, name, insert_mode=PETSc.InsertMode.INSERT):
        """

        Parameters
        ----------
        name
        comm

        Returns
        -------

        """
        # with more than one processor, we need to collect the data from all the processors. Otherwise, we just read it from
        # the local copy of the vector
        if isGt1(self.comm):
            # switch the plex section to the correct dof
            dof = self.variables_info[name].dof
            self.plex.setSection(self.sections_distributed[dof])

            # is the insert modes is anything other then INSERT, set the temporal vector to zero before the transfer
            # The mode INSERT will overwrite the values anyway.
            if insert_mode != PETSc.InsertMode.INSERT:
                self.temporal_vectors_global[dof].zeroEntries()
                self.temporal_vectors_global[dof].assemble()

            # create a copy without ghost values and send all partitions to rank 0
            self.plex.localToGlobal(self.variables[name], self.temporal_vectors_global[dof], addv=insert_mode)
            self.scatter_to_zero[dof].scatter(self.temporal_vectors_global[dof], self.temporal_vectors_on_zero[dof])

            # on rank zero correct the permutation of the vector
            if onRank0(self.comm):
                self.temporal_vectors_on_zero[dof].permute(self.__getPermutationIS(dof), True)

            # we need to create a copy of the result as we are always using the same temporal vector for the transfer
            result_array = self.temporal_vectors_on_zero[dof].getArray().copy()
            if self.variables_info[name].shape_on_zero is not None and self.variables_info[name].shape_on_zero != result_array.shape:
                result_array = np.require(result_array.reshape(self.variables_info[name].shape_on_zero), requirements="C")

            # switch the plex section back to the default 1
            self.plex.setSection(self.sections_distributed[1])
            return np.copy(result_array)
        else:
            result_array = self.variables[name].getArray()
            if self.variables_info[name].shape_on_zero is not None and self.variables_info[name].shape_on_zero != result_array.shape:
                result_array = np.require(result_array.reshape(self.variables_info[name].shape_on_zero), requirements="C")
            return result_array

    def getGlobalArray(self, name):
        """
        to be called on a single processor to get a local partition of a global array that includes no overlapping
        ghost region.

        Parameters
        ----------
        name: str
                name of the Variable. It must have been added with addVariable before.

        Returns
        -------
        np.array
        """
        dof = self.variables_info[name].dof
        self.plex.setSection(self.sections_distributed[dof])
        self.plex.localToGlobal(self.variables[name], self.temporal_vectors_global[dof])
        self.plex.setSection(self.sections_distributed[1])
        result_array = np.copy(self.temporal_vectors_global[dof].getArray())
        # reshape the result is required
        if self.variables_info[name].shape_global is not None and self.variables_info[name].shape_global != result_array.shape:
            result_array = np.require(result_array.reshape(self.variables_info[name].shape_global), requirements="C")
        return result_array

    def getLocalArray(self, name):
        """
        to be called on a single processor to get a local partition of a global array that includes the overlapping
        ghost region.

        Parameters
        ----------
        name: str
                name of the Variable. It must have been added with addVariable before.

        Returns
        -------
        np.array
        """
        result_array = np.copy(self.variables[name].getArray())
        # reshape the result is required
        if self.variables_info[name].shape_local is not None and self.variables_info[name].shape_local != result_array.shape:
            result_array = np.require(result_array.reshape(self.variables_info[name].shape_local), requirements="C")
        return result_array


class VariableInfo():
    """
    store information about variables added to the DMPlex object
    """
    def __init__(self, dof, shape_on_zero=None, partition_sizes=None):
        self.dof = dof
        # the global shape contains all points on rank 0 and zero points elsewhere
        self.shape_on_zero = shape_on_zero
        # local shapes depend on local partition sizes
        if partition_sizes is not None:
            self.shape_local = list(self.shape_on_zero)
            self.shape_local[0] = partition_sizes[0]
            self.shape_local = tuple(self.shape_local)
            self.shape_global = list(self.shape_on_zero)
            self.shape_global[0] = partition_sizes[1]
            self.shape_global = tuple(self.shape_global)
        else:
            self.shape_local = None
            self.shape_global = None

    def __str__(self):
        return f"VariableInfo: shape_on_zero={self.shape_on_zero}, shape_local={self.shape_local}, shape_global={self.shape_global}"


