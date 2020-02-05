# tests for the mpi-distributed grid
import os
import pytest
import numpy as np
from petsc4py import PETSc
from enstools.mpi.grids import UnstructuredGrid
from enstools.mpi import init_petsc, onRank0
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
    download a small grid definition file to a temporal folder
    """
    # download a 320km grid
    if onRank0(comm):
        gridfile = download(url="http://icon-downloads.mpimet.mpg.de/grids/public/edzw/icon_grid_0009_R02B03_R.nc", 
                            destination=os.path.join(get_tmpdir.getpath(), "R02B03.nc"))
    else: 
        gridfile = None
    comm.barrier()
    return gridfile

@pytest.fixture
def comm():
    """
    initialize the PETSc library and create a COMM_WORLD object
    """
    result = init_petsc()
    return result


def test_init_grid(gridfile, comm):
    """
    try to create a new Unstructured Grid instance
    """
    # read a griddefinition file using the read function
    if onRank0(comm):
        ds = read(gridfile)
    else:
        ds = None

    # create a grid definition for this grid
    grid = UnstructuredGrid(ds, comm=comm)

    # check the number of grid points
    assert grid.ncells == 5120

    # check coordinates
    clon = grid.gatherData("clon")
    clat = grid.gatherData("clat")
    coords = grid.gatherData("coordinates_cartesian")
    if onRank0(comm):
        np.testing.assert_array_equal(ds["clon"].values.astype(PETSc.RealType), clon)
        np.testing.assert_array_equal(ds["clat"].values.astype(PETSc.RealType), clat)
        assert coords.shape == (clon.shape[0], 3)
    
    comm.barrier()