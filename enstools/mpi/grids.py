import logging
from .logging import log_on_rank, log_and_time
from ..mpi import onRank0, isGt1
from enstools.misc import spherical2cartesien
from petsc4py import PETSc
from typing import Dict
import numpy as np
import zlib


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
        self.variables_info: Dict[str, VariableInfo] = {}
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
            else:
                indices = np.zeros(0, dtype=PETSc.RealType)
            # add global indices as new variable to the grid
            self.addVariable("global_indices", values=indices)
            # get the global form (no ghost points)  of the indices
            self.plex.localToGlobal(self.variables["global_indices"], self.temporal_vectors_global[1])
            # scatter this indices to process zero and and filter out ghost points
            self.scatter_to_zero[1].scatter(self.temporal_vectors_global[1], self.temporal_vectors_on_zero[1])
            if onRank0(self.comm):
                self.permutation_indices = np.asarray(self.temporal_vectors_on_zero[1].getArray(), dtype=PETSc.IntType)

            # store owner and owner indices of ghost regions on every processor and include an invers mapping.
            # -----------------------------------------------------------------------------------------------
            # at first, get the local owned sizes everywhere
            self.owned_sizes = np.zeros(self.mpi_size, dtype=PETSc.IntType)
            local_owned_size = np.asarray(self.temporal_vectors_global[1].getSizes()[0],dtype=PETSc.IntType)
            self.comm_mpi4py.Allgather(local_owned_size, self.owned_sizes)
            
            # create a new vector containing the owner
            self.addVariable("owner")
            self.getLocalArray("owner")[:] = 0
            self.getLocalArray("owner")[:self.owned_sizes[self.mpi_rank]] = self.mpi_rank
            self.variables["owner"].assemble()
            owner = self.gatherData("owner", insert_mode=PETSc.InsertMode.ADD)
            self.scatterData("owner", owner)

            # create a new vector containing the local indices in a local array
            self.addVariable("owner_indices")
            self.getLocalArray("owner_indices")[:] = 0
            self.getLocalArray("owner_indices")[:self.owned_sizes[self.mpi_rank]] = \
                np.arange(self.owned_sizes[self.mpi_rank], dtype=PETSc.RealType)
            self.variables["owner_indices"].assemble()
            owner_indices = self.gatherData("owner_indices", insert_mode=PETSc.InsertMode.ADD)
            self.scatterData("owner_indices", owner_indices)

            # a forward and backward mapping of all ghost points
            self.addVariable("ghost_indices")
            self.getLocalArray("ghost_indices")[:] = 0
            self.ghost_mapping: Dict[int, GhostMapping] = {}
            # array with all indices that have some ghost a ghost assigned to it.
            self.ghost_mapping_all_owned_with_remote_ghost = np.empty(0, dtype=PETSc.IntType)
            for one_rank in range(self.mpi_size):
                ghost_indices_local = self.getLocalArray("ghost_indices")
                if one_rank == self.mpi_rank:
                    ghost_indices_local[self.owned_sizes[self.mpi_rank]:] = \
                        np.arange(self.owned_sizes[self.mpi_rank], self.temporal_vectors_local[1].getSizes()[0], dtype=PETSc.RealType)
                else:
                    ghost_indices_local[:] = 0
                self.variables["ghost_indices"].assemble()
                ghost_indices = self.gatherData("ghost_indices", insert_mode=PETSc.InsertMode.ADD)
                self.scatterData("ghost_indices", ghost_indices)
                ghost_indices_local = self.getLocalArray("ghost_indices")
                ghost_indices_local[self.owned_sizes[self.mpi_rank]:] = 0
                indices_with_ghosts = np.nonzero(self.getLocalArray("ghost_indices"))[0]
                owned_by_remote_rank = np.where(self.getLocalArray("owner") == one_rank)[0]
                owned_by_remote_indices = self.getLocalArray("owner_indices")[owned_by_remote_rank]
                if one_rank != self.mpi_rank and (owned_by_remote_indices.size > 0 or indices_with_ghosts.size > 0):
                    # create mapping for this rank
                    self.ghost_mapping[one_rank] = GhostMapping(
                        rank=one_rank,
                        local_indices_that_are_remote_ghost=indices_with_ghosts,
                        remote_indices_of_ghosts=ghost_indices_local[indices_with_ghosts],
                        remote_indices_that_are_local_ghost=owned_by_remote_indices,
                        local_indices_of_ghosts=owned_by_remote_rank
                    )
                    # add points to mapping for all ranks
                    self.ghost_mapping_all_owned_with_remote_ghost = np.append(self.ghost_mapping_all_owned_with_remote_ghost, indices_with_ghosts)
            self.ghost_mapping_all_owned_with_remote_ghost.sort()
            self.ghost_mapping_all_owned_with_remote_ghost = np.unique(self.ghost_mapping_all_owned_with_remote_ghost)
            self.removeVariable("ghost_indices")

            # set limit for transfer buffers. 128MB
            self.buffer_size_limit = 1048576 * 128

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
        self.addVariable("clon", values=clon)
        self.addVariable("clat", values=clat)
        self.addVariable("coordinates_cartesian", values=coords)
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

    def addVariable(self, name, values=None, dof=None):
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
        if onRank0(self.comm) and values is not None and not values.shape[0] == self.ncells:
            raise ValueError(f"Variable {name}: shape has not the number of grid cells in the first dimension: {values.shape}")

        # when values are provided, a corresponding section if created
        if values is not None and not dof in self.sections_on_zero:
            self.__createNonDistributedSection(dof)
            if isGt1(self.comm):
                self.__createDistributedSection(dof)

        # create a new variable with the appropriate section
        if isGt1(self.comm):
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
            # use the real shape of the given array
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
        else:
            # use the default shape given by the DoF
            if dof > 1:
                shape_on_zero = (self.ncells, dof)
            else:
                shape_on_zero = (self.ncells,)
            if isGt1(self.comm):
                partition_sizes = self.temporal_vectors_local[1].getSizes()[0], self.temporal_vectors_global[1].getSizes()[0]
            else:
                partition_sizes = self.ncells, self.ncells
            if self.mpi_rank > 0:
                shape_on_zero = list(shape_on_zero)
                shape_on_zero[0] = 0
                shape_on_zero = tuple(shape_on_zero)
        self.variables_info[name] = VariableInfo(dof=dof, shape_on_zero=shape_on_zero, partition_sizes=partition_sizes)

        # if we have already values, scatter them to all processes
        if values is not None:
            self.scatterData(name, values=values, dof=dof)

    def removeVariable(self, name):
        """
        remove a variable and related support structures from the grid.
        """
        # delete the actual vector object.
        self.variables[name].destroy()
        del self.variables[name]

        # check if the DoF related structures are still required.
        info = self.variables_info[name]
        del self.variables_info[name]
        dof = info.dof
        if dof != 1:
            # find out if other variables with the same DoF exist.
            others_exist = False
            for other in self.variables_info:
                if self.variables_info[other].dof == dof:
                    others_exist = True
                    break

            # without others, remove support vectors, etc
            if not others_exist:
                if dof in self.temporal_vectors_on_zero:
                    self.temporal_vectors_on_zero[dof].destroy()
                    del self.temporal_vectors_on_zero[dof]
                if dof in self.temporal_vectors_global:
                    self.temporal_vectors_global[dof].destroy()
                    del self.temporal_vectors_global[dof]
                if dof in self.temporal_vectors_local:
                    self.temporal_vectors_local[dof].destroy()
                    del self.temporal_vectors_local[dof]
                if dof in self.sections_on_zero:
                    self.sections_on_zero[dof].destroy()
                    del self.sections_on_zero[dof]
                if dof in self.sections_distributed:
                    self.sections_distributed[dof].destroy()
                    del self.sections_distributed[dof]
                if dof in self.scatter_to_zero:
                    self.scatter_to_zero[dof].destroy()
                    del self.scatter_to_zero[dof]
                if dof in self.scatter_to_zero_is:
                    self.scatter_to_zero_is[dof].destroy()
                    del self.scatter_to_zero_is[dof]
        
    def scatterData(self, name, values=None, dof=None):
        """

        Parameters
        ----------
        name
        values

        Returns
        -------

        """
        log_and_time(f"UnstructuredGrid.scatterData({name})", logging.INFO, True, self.comm)
        # make sure we have data continuous in memory
        if values is not None:
            values = np.require(values, requirements="C")
        # with more than one processor, we need to distribute the data. Otherwise, we just store it locally
        if isGt1(self.comm):
            # on rank 0 write the data into the total grid vector
            dof = self.__getDoFforArray(values, dof)
            if onRank0(self.comm):
                # make sure, the shape of the given array matches the expected shape stored as variable information
                if not values.shape == self.variables_info[name].shape_on_zero:
                    log_on_rank(f"scatterData: array with shape {self.variables_info[name].shape_on_zero} expected, but {values.shape} given.", logging.ERROR, self.comm, self.mpi_rank)
                    self.comm_mpi4py.Abort()
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
        log_and_time(f"UnstructuredGrid.scatterData({name})", logging.INFO, False, self.comm)

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
                # FIXME
                assert dof == 1

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
            return result_array
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
        # without multiple processes, global and local arrays are identical
        if not isGt1(self.comm):
            return self.getLocalArray(name)
        # for multiple processes we have to use localToGlobal function
        dof = self.variables_info[name].dof
        self.plex.setSection(self.sections_distributed[dof])
        self.plex.localToGlobal(self.variables[name], self.temporal_vectors_global[dof])
        if dof != 1:
            self.plex.setSection(self.sections_distributed[1])
        result_array = np.copy(self.temporal_vectors_global[dof].getArray())
        # reshape the result is required
        if self.variables_info[name].shape_global is not None and self.variables_info[name].shape_global != result_array.shape:
            result_array = np.require(result_array.reshape(self.variables_info[name].shape_global), requirements="C")
        return result_array

    def getLocalArray(self, name: str):
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
        result_array = self.variables[name].getArray()
        # reshape the result is required
        if self.variables_info[name].shape_local is not None and self.variables_info[name].shape_local != result_array.shape:
            result_array = np.require(result_array.reshape(self.variables_info[name].shape_local), requirements="C")
        return result_array

    def updateGhost(self, name: str, local_indices: np.array = None, direction: str = "O2G"):
        """
        Copy changes made on an array by the owner to users of the Ghost points on other processors.
        This function makes use of arrays obtained by calls to getLocalArray.

        Parameters
        ----------
        name: str
                name of the variable to update
        
        local_indices: np.array
                local indices on the owner process that should be copied to neighbours. If None, the complete 
                overlapping region is updated.

        direction: {'O2G', 'G2O'}
                communication direction: owner to ghost (O2G) or ghost to owner (G2O). The default is to copy locally
                updated owned grid points to remote ghosts.
        """
        # do nothing if we are running with one processor only
        if not isGt1(self.comm):
            return
        log_and_time(f"UnstructuredGrid.updateGhost({name})", logging.INFO, True, self.comm)

        # unless a full update is performed, the receiver needs to know which indices are on the way. Here everyone
        # tells everyone what indices are intended for transmission
        if local_indices is not None:
            remote_indices = self.comm_mpi4py.allgather(local_indices)

        # construct source and destination indices for each remote rank and find the maximal number of indices
        # to transfer
        max_indices_to_transfer = 0
        indices_send = {}
        indices_recv = {}
        for rank in self.ghost_mapping:
            # send our data to external users
            if direction == "O2G":
                # select the indices that the current rank should send to the rank "rank".
                if local_indices is not None:
                    _indices_send = np.intersect1d(local_indices,
                                                   self.ghost_mapping[rank].local_indices_that_are_remote_ghost,
                                                   assume_unique=True)
                else:
                    _indices_send = self.ghost_mapping[rank].local_indices_that_are_remote_ghost
                if _indices_send.size > 0:
                    indices_send[rank] = _indices_send
                    max_indices_to_transfer = max(max_indices_to_transfer, _indices_send.size)
                # select the indices that the remote rank should use to write the received values to
                if local_indices is not None:
                    _, _, _indices_of_indices = np.intersect1d(remote_indices[rank],
                                                               self.ghost_mapping[rank].remote_indices_that_are_local_ghost,
                                                               assume_unique=True,
                                                               return_indices=True)
                    _indices_recv = self.ghost_mapping[rank].local_indices_of_ghosts[_indices_of_indices]
                else:
                    _indices_recv = self.ghost_mapping[rank].local_indices_of_ghosts
                if _indices_recv.size > 0:
                    indices_recv[rank] = _indices_recv
                    max_indices_to_transfer = max(max_indices_to_transfer, _indices_recv.size)
            # get data back form external users and take over their changes
            elif direction == "G2O":
                # select the indices that the current rank should send to the rank "rank".
                if local_indices is not None:
                    _indices_send = np.intersect1d(local_indices,
                                                   self.ghost_mapping[rank].local_indices_of_ghosts,
                                                   assume_unique=True)
                else:
                    _indices_send = self.ghost_mapping[rank].local_indices_of_ghosts
                if _indices_send.size > 0:
                    indices_send[rank] = _indices_send
                    max_indices_to_transfer = max(max_indices_to_transfer, _indices_send.size)
                # select the indices that the remote rank should use to write the received values to
                if local_indices is not None:
                    _, _, _indices_of_indices = np.intersect1d(remote_indices[rank],
                                                               self.ghost_mapping[rank].remote_indices_of_ghosts,
                                                               assume_unique=True,
                                                               return_indices=True)
                    _indices_recv = self.ghost_mapping[rank].local_indices_that_are_remote_ghost[_indices_of_indices]
                else:
                    _indices_recv = self.ghost_mapping[rank].local_indices_that_are_remote_ghost
                if _indices_recv.size > 0:
                    indices_recv[rank] = _indices_recv
                    max_indices_to_transfer = max(max_indices_to_transfer, _indices_recv.size)
            else:
                raise NotImplementedError("only update direction O2G and G2O are implemented!")

        # get information about thw variable. we need to know the number of dimensions
        buffer_shape = list(self.variables_info[name].shape_on_zero)

        # calculate the maximal number of indices to transfer at once taking the maximal buffer size into account.
        if onRank0(self.comm):
            max_indices_in_buffer = int(max(1, np.rint(self.buffer_size_limit / 4 / (self.mpi_size - 1) / (np.prod(buffer_shape) / self.ncells))))
        else:
            max_indices_in_buffer = None
        max_indices_in_buffer = self.comm_mpi4py.bcast(max_indices_in_buffer)

        # split up the transfer into smaller chunks to make sure, the the totally used buffer size remains below the
        # total buffer limit.
        for start_index in range(0, max_indices_to_transfer, max_indices_in_buffer):
            # use a checksum of the name as tag in MPI messages
            name_tag = zlib.crc32(f"{name}{start_index}".encode()) // 2

            # create a buffer for sending and receiving
            buffers_send = {}
            for rank in indices_send:
                buffer_shape[0] = max(0, min(indices_send[rank].size - start_index, max_indices_in_buffer))
                if buffer_shape[0] == 0 and rank in buffers_send:
                    del buffers_send[rank]
                elif buffer_shape[0] > 0:
                    if (rank in buffers_send and buffers_send[rank].shape != tuple(buffer_shape)) or rank not in buffers_send:
                        buffers_send[rank] = np.empty(tuple(buffer_shape), dtype=PETSc.RealType)
            buffers_recv = {}
            for rank in indices_recv:
                buffer_shape[0] = max(0, min(indices_recv[rank].size - start_index, max_indices_in_buffer))
                if buffer_shape[0] == 0 and rank in buffers_recv:
                    del buffers_recv[rank]
                elif buffer_shape[0] > 0:
                    if (rank in buffers_recv and buffers_recv[rank].shape != tuple(buffer_shape)) or rank not in buffers_recv:
                        buffers_recv[rank] = np.empty(tuple(buffer_shape), dtype=PETSc.RealType)

            # start asynchronous receivers
            requests_resv = {}
            for rank in buffers_recv:
                requests_resv[rank] = self.comm_mpi4py.Irecv(buffers_recv[rank], source=rank, tag=name_tag)

            # start sending the data
            requests_send = {}
            for rank in buffers_send:
                # copy the data from the source array
                buffers_send[rank][:, ...] = self.getLocalArray(name)[indices_send[rank][start_index:start_index + max_indices_in_buffer], ...]
                # start the sender
                requests_send[rank] = self.comm_mpi4py.Isend(buffers_send[rank], dest=rank, tag=name_tag)

            # wait for all communication to complete
            for req in requests_send:
                requests_send[req].wait()
            for req in requests_resv:
                requests_resv[req].wait()

            # copy results into the destination position
            for rank in buffers_recv:
                # copy the data
                self.getLocalArray(name)[indices_recv[rank][start_index:start_index + max_indices_in_buffer], ...] = buffers_recv[rank]
        
        # wait for all to have a consistent state
        self.comm.barrier()
        self.variables[name].assemble()
        log_and_time(f"UnstructuredGrid.updateGhost({name})", logging.INFO, False, self.comm)


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


class GhostMapping():
    """
    Bidirectional mapping between Ghost points and their actual owner. 
    """
    def __init__(self, rank: int, 
                local_indices_that_are_remote_ghost: np.array, 
                remote_indices_of_ghosts: np.array, 
                remote_indices_that_are_local_ghost: np.array, 
                local_indices_of_ghosts: np.array):

        # mapping for direction local to remote
        self.local_indices_that_are_remote_ghost = local_indices_that_are_remote_ghost
        self.remote_indices_of_ghosts = remote_indices_of_ghosts
        
        # create a mapping to the local-to-remote direction
        self.local_to_remote_mapping = {}
        for i in range(self.local_indices_that_are_remote_ghost.size):
            self.local_to_remote_mapping[self.local_indices_that_are_remote_ghost[i]] = self.remote_indices_of_ghosts[i]

        # mapping for direction remote to local
        self.remote_indices_that_are_local_ghost = remote_indices_that_are_local_ghost
        self.local_indices_of_ghosts = local_indices_of_ghosts
        
        # create a mapping to the remote-to-local direction
        self.remote_to_local_mapping = {}
        for i in range(self.remote_indices_that_are_local_ghost.size):
            self.remote_to_local_mapping[self.remote_indices_that_are_local_ghost[i]] = self.local_indices_of_ghosts[i]
