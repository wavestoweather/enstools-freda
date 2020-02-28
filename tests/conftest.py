import os
import pytest
from enstools.da.support import FeedbackFile
from enstools.da.support.feedback_file import LevelType
from enstools.mpi.grids import UnstructuredGrid
from enstools.mpi import init_petsc, onRank0, isGt1
from enstools.misc import download, generate_coordinates
from enstools.core.tempdir import TempDir
from enstools.io import read
import numpy as np


@pytest.fixture(scope="session")
def get_tmpdir(comm):
    """
    create a temporal directory which will be removed on exit
    """
    if onRank0(comm):
        # if the environment variable SCRATCH is set, it is used a a prefix for temporal directories.
        scratch = os.getenv("SCRATCH")
        tmpdir = TempDir(parentdir=scratch, check_free_space=False)
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
        gridfile = "/archive/meteo/external-models/dwd/grids/icon_grid_0016_R02B06_G.nc"
        if not os.path.exists(gridfile):
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
    grid = UnstructuredGrid(gridfile, overlap=20, comm=comm)

    # check the number of grid points
    assert grid.ncells == 327680

    # wait for all ranks to finish the test
    comm.barrier()
    return grid


@pytest.fixture(scope="session")
def ff(gridfile, comm):
    """
    create a feedback file for a given grid
    """
    if onRank0(comm):
        assert gridfile is not None
        _ff = FeedbackFile(gridfile=gridfile)
        assert _ff.grid is not None
        return _ff
    else:
        return None


@pytest.fixture(scope="session")
def ff_with_obs(ff: FeedbackFile, comm):
    """
    use model output file to create observations within the feedback file
    """
    if not onRank0(comm):
        comm.barrier()
        return None

    # create one report every one degree
    lon, lat = generate_coordinates(2.0, lat_range=[-30, 30], unit="radians")
    lon, lat = np.meshgrid(lon, lat)
    lon = lon.ravel()
    lat = lat.ravel()
    assert lon.shape == (5400,)
    assert lat.shape == (5400,)

    # use data of the last ensemble member to generate test observations
    # add observations into an empty file
    ff.add_observation_from_model_output(
        "/archive/meteo/external-models/dwd/icon/oper/icon_oper_eps_gridded-global_rolling/202002/20200201T00/igaf2020020100.m040.grb",
        variables=["T", "U", "V"],
        error={"T": 1.0, "U": 1.0, "V": 1.0},
        lon=lon,
        lat=lat,
        levels=[100000, 50000]
    )

    # here we should have for each coordinate one report
    assert ff.data["i_body"].shape[0] == lon.shape[0]
    # we do not have six observations because some grid points are above 1000 hPa at the surface
    assert ff.data["obs"].shape[0] > lon.shape[0] * 3
    assert ff.data["obs"].shape[0] < lon.shape[0] * 6

    # add observations into an file with existing content
    ff.add_observation_from_model_output(
        "/archive/meteo/external-models/dwd/icon/oper/icon_oper_eps_gridded-global_rolling/202002/20200201T00/igaf2020020100.m040.grb",
        variables=["QV", "P"],
        error={"QV": 0.001, "P": 100.0},
        lon=lon,
        lat=lat,
        levels=[85, 60],
        level_type=LevelType.MODEL_LEVEL
    )

    # here we should have additional observations for each grid point
    assert ff.data["i_body"].shape[0] == lon.shape[0] * 2
    assert ff.data["obs"].shape[0] > lon.shape[0] * 5
    assert ff.data["obs"].shape[0] < lon.shape[0] * 10

    # entries in i_body and l_body have to match
    i_body = ff.data["i_body"].values
    l_body = ff.data["l_body"].values
    for index in range(i_body.shape[0] - 1):
        assert i_body[index + 1] == i_body[index] + l_body[index]

    # wait for all to finish.
    comm.barrier()
    return ff


@pytest.fixture(scope="session")
def ff_file(ff_with_obs: FeedbackFile, get_tmpdir: TempDir, comm):
    """
    write the content of a feedback file to a file
    """
    if onRank0(comm):
        filename = os.path.join(get_tmpdir.getpath(), "observations.nc")
        ff_with_obs.write_to_file(filename)
    else:
        filename = None
    comm.barrier()
    return filename