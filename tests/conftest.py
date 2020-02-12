import os
import pytest
from enstools.mpi.grids import UnstructuredGrid
from enstools.mpi import init_petsc, onRank0, isGt1
from enstools.misc import download
from enstools.core.tempdir import TempDir
from enstools.io import read


@pytest.fixture(scope="session")
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


@pytest.fixture(scope="session")
def gridfile(get_tmpdir, comm):
    """
    download a small grid definition file to a temporal folder and read its content
    """
    # download a 320km grid
    if onRank0(comm):
        gridfile = download(url="http://icon-downloads.mpimet.mpg.de/grids/public/edzw/icon_grid_0016_R02B06_G.nc",
                            destination=os.path.join(get_tmpdir.getpath(), "R02B06.nc"))
        ds = read(gridfile)
    else:
        ds = None
    comm.barrier()
    return ds


@pytest.fixture(scope="session")
def comm():
    """
    initialize the PETSc library and create a COMM_WORLD object
    """
    result = init_petsc()
    return result


@pytest.fixture(scope="session")
def grid_without_overlap(gridfile, comm):
    """
    try to create a new Unstructured Grid instance
    """
    # create a grid definition for this grid
    grid = UnstructuredGrid(gridfile, comm=comm)

    # check the number of grid points
    assert grid.ncells == 327680

    # wait for all ranks to finish the test
    comm.barrier()
    return grid


@pytest.fixture(scope="session")
def grid_with_overlap(gridfile, comm):
    """
    create a grid with overlapping region
    """
    # create a grid definition for this grid
    grid = UnstructuredGrid(gridfile, overlap=1, comm=comm)

    # check the number of grid points
    assert grid.ncells == 327680

    # wait for all ranks to finish the test
    comm.barrier()
    return grid
