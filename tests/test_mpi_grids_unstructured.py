# tests for the mpi-distributed grid
import os
import pytest
import numpy as np
from petsc4py import PETSc
from enstools.mpi.grids import UnstructuredGrid
from enstools.mpi import init_petsc, onRank0, isGt1
from enstools.misc import download
from enstools.core.tempdir import TempDir
from enstools.io import read


@pytest.fixture
def get_tmpdir(comm):
    """
    create a temporal directory which will be removed on exit
    """
    if onRank0(comm):
        tmpdir = TempDir(check_free_space=False)
    else:
        tmpdir = None
    comm.barrier()
    return tmpdir


@pytest.fixture
def gridfile(get_tmpdir, comm):
    """
    download a small grid definition file to a temporal folder and read its content
    """
    # download a 320km grid
    if onRank0(comm):
        gridfile = download(url="http://icon-downloads.mpimet.mpg.de/grids/public/edzw/icon_grid_0009_R02B03_R.nc", 
                            destination=os.path.join(get_tmpdir.getpath(), "R02B03.nc"))
        ds = read(gridfile)
    else: 
        ds = None
    comm.barrier()
    return ds

@pytest.fixture
def comm():
    """
    initialize the PETSc library and create a COMM_WORLD object
    """
    result = init_petsc()
    return result


@pytest.fixture
def grid_without_overlap(gridfile, comm):
    """
    try to create a new Unstructured Grid instance
    """
    # create a grid definition for this grid
    grid = UnstructuredGrid(gridfile, comm=comm)

    # check the number of grid points
    assert grid.ncells == 5120

    # wait for all ranks to finish the test
    comm.barrier()
    return grid


@pytest.fixture
def grid_with_overlap(gridfile, comm):
    """
    create a grid with overlapping region
    """
    # create a grid definition for this grid
    grid = UnstructuredGrid(gridfile, overlap=1, comm=comm)

    # check the number of grid points
    assert grid.ncells == 5120

    # wait for all ranks to finish the test
    comm.barrier()
    return grid


def test_grid_without_overlap(grid_without_overlap, gridfile, comm):
    """
    test some properties of a non-overlapping grid
    """
    # check coordinates
    clon = grid_without_overlap.gatherData("clon")
    clat = grid_without_overlap.gatherData("clat")
    coords = grid_without_overlap.gatherData("coordinates_cartesian")
    if onRank0(comm):
        np.testing.assert_array_equal(gridfile["clon"].values.astype(PETSc.RealType), clon)
        np.testing.assert_array_equal(gridfile["clat"].values.astype(PETSc.RealType), clat)
        assert coords.shape == (clon.shape[0], 3)


def test_grid_with_overlap(grid_with_overlap, gridfile, comm):
    """
    compare two grids with and without overlap
    """
    # compare arrays with and without overlap
    clon_with = grid_with_overlap.getLocalArray("clon")
    clon_without = grid_with_overlap.getGlobalArray("clon")
    if isGt1(comm):
        # with overlap, we have more points then without
        assert clon_with.size > clon_without.size
        # the beginning of both arrays should be identical
        np.testing.assert_array_equal(clon_with[:clon_without.size], clon_without)
    else:
        np.testing.assert_array_equal(clon_with, clon_without)

    # compare data gathered from both processes to data read from the gridfile
    clon_gathered = grid_with_overlap.gatherData("clon")
    if onRank0(comm):
        clon_original = gridfile["clon"].values.astype(PETSc.RealType)
        np.testing.assert_array_equal(clon_gathered, clon_original)
    
    # create a new array, write the rank in this array
    grid_with_overlap.addVariable("rank")
    if isGt1(comm):
        assert grid_with_overlap.variables["rank"].getSize() < grid_with_overlap.ncells
    grid_with_overlap.getLocalArray("rank")[:] = comm.Get_rank() + 1
    grid_with_overlap.variables["rank"].assemble()
    
    # check that all local values have the value of the rank
    rank_on_processor_without_overlap = grid_with_overlap.getLocalArray("rank")
    rank_on_processor_with_overlap = grid_with_overlap.getGlobalArray("rank")
    np.testing.assert_array_equal(rank_on_processor_without_overlap, comm.Get_rank() + 1)
    np.testing.assert_array_equal(rank_on_processor_with_overlap, comm.Get_rank() + 1)
    
    # gather all data with adding of values
    ranks = grid_with_overlap.gatherData("rank", insert_mode=PETSc.InsertMode.ADD)
    # the maximum is reached in overlapping areas.
    if onRank0(comm) and isGt1(comm):
        assert ranks.max() > comm.Get_size()
        assert ranks.max() <= np.sum(np.arange(comm.Get_size()) + 1)
 
    # check the owned sizes on each rank
    if isGt1(comm):
        # everyone should now the sizes owned be every other process
        assert grid_with_overlap.owned_sizes[comm.Get_rank()] == grid_with_overlap.temporal_vectors_global[1].getSizes()[0]
 
        # check if owned indices are correctly labeled as owned.
        owner = grid_with_overlap.getGlobalArray("owner")
        np.testing.assert_array_equal(owner, comm.Get_rank())

        # make sure the ghost region is not owned by the rank itself
        owner = grid_with_overlap.getLocalArray("owner")
        np.testing.assert_array_equal(owner[:grid_with_overlap.owned_sizes[comm.Get_rank()]], comm.Get_rank())
        ghost = owner[grid_with_overlap.owned_sizes[comm.Get_rank()]:]
        np.testing.assert_equal(ghost != comm.Get_rank(), True)
        np.testing.assert_equal(ghost < comm.Get_size(), True)
        
        # make sure, that only owned points have indices for remote ghost points
        for rank in range(comm.Get_size()):
            if rank != comm.Get_rank() and rank in grid_with_overlap.ghost_mapping:
                assert grid_with_overlap.ghost_mapping[rank].local_indices_that_are_remote_ghost.max() < grid_with_overlap.owned_sizes[comm.Get_rank()]
                assert grid_with_overlap.ghost_mapping[rank].remote_indices_of_ghosts.size > 0

        # make sure, that only ghost points have indices for remotely owned points
        for rank in range(comm.Get_size()):
            if rank != comm.Get_rank() and rank in grid_with_overlap.ghost_mapping:
                # remote indices are remotely owned
                assert grid_with_overlap.ghost_mapping[rank].remote_indices_that_are_local_ghost.max() < grid_with_overlap.owned_sizes[rank]
                # we must have local indices of ghosts
                assert grid_with_overlap.ghost_mapping[rank].local_indices_of_ghosts.size > 0
                # local indices of ghosts must be in the local ghost range
                assert grid_with_overlap.ghost_mapping[rank].local_indices_of_ghosts.min() >= grid_with_overlap.owned_sizes[comm.Get_rank()]

    comm.barrier()


def test_grid_remove_variable(grid_with_overlap, comm):
    """
    add and remove a variable from the grid
    """

    # add a variable with DoF=1
    grid_with_overlap.addVariable("test", values=np.arange(grid_with_overlap.ncells, dtype=PETSc.RealType))

    # remove variable again. Support structured for dof=1 should never be removed
    grid_with_overlap.removeVariable("test")
    assert "test" not in grid_with_overlap.variables
    assert "test" not in grid_with_overlap.variables_info
    assert 1 in grid_with_overlap.sections_on_zero
    if isGt1(comm) and onRank0(comm):
        assert 1 in grid_with_overlap.scatter_to_zero_is
    
    # add a larger variable
    grid_with_overlap.addVariable("test", values=np.zeros((grid_with_overlap.ncells, 90), dtype=PETSc.RealType))
    # retrieve it once, that will create the scatter context
    test = grid_with_overlap.gatherData("test")
    if onRank0(comm):
        assert test.shape == (grid_with_overlap.ncells, 90)
    assert 90 in grid_with_overlap.sections_on_zero
    if isGt1(comm) and onRank0(comm):
        assert 90 in grid_with_overlap.scatter_to_zero_is

    # remove the variable again
    grid_with_overlap.removeVariable("test")
    assert 90 not in grid_with_overlap.sections_on_zero
    if isGt1(comm) and onRank0(comm):
        assert 90 not in grid_with_overlap.scatter_to_zero_is


def test_ghost_update(grid_with_overlap, comm):
    """
    create a variable with random noise and transfer ghost regions 
    """
    noise = np.random.rand(grid_with_overlap.ncells)
    grid_with_overlap.addVariable("noise", values=noise)

    # with more than one task, we can check the update. Modify the array including
    # Ghost region on task 1, update, check ghost region on task 0
    if isGt1(comm):
        # modify content incl. ghost on one processor
        local_noise = grid_with_overlap.getLocalArray("noise")
        clon = grid_with_overlap.getLocalArray("clon")
        if comm.Get_rank() == 1:
            local_noise[:] = clon[:]
        grid_with_overlap.variables["noise"].assemble()

        # perform the update
        grid_with_overlap.updateOwnerToGhost("noise")

        # check the update
        updated_noise = grid_with_overlap.getLocalArray("noise")
        if comm.Get_rank() == 0:
            np.testing.assert_array_equal(updated_noise[grid_with_overlap.ghost_mapping[1].local_indices_of_ghosts], clon[grid_with_overlap.ghost_mapping[1].local_indices_of_ghosts])
        