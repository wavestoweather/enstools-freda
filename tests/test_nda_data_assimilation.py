import pytest
import numpy as np
from enstools.da.nda import DataAssimilation
from enstools.io import read
from enstools.mpi import onRank0
from enstools.mpi.grids import UnstructuredGrid


@pytest.fixture
def da(grid_with_overlap: UnstructuredGrid, comm):
    """
    Initialize the DataAssimilation object
    """
    da = DataAssimilation(grid_with_overlap, comm)
    da.load_state("/archive/meteo/external-models/dwd/icon/oper/icon_oper_eps_gridded-global_rolling/202002/20200201T00/igaf2020020100.m00[1-5].grb")

    # compare uploaded data with input files
    if onRank0(comm):
        ds = read("/archive/meteo/external-models/dwd/icon/oper/icon_oper_eps_gridded-global_rolling/202002/20200201T00/igaf2020020100.m00[1-5].grb")
        assert ds["t"].dims == ("time", "ens", "generalVerticalLayer", "cell")

    # only check variable v=4 on first five ensemble members
    for ens in range(5):
        state_v = da.grid.gatherData("state", part=(slice(None), 4, ens))
        if onRank0(comm):
            np.testing.assert_array_equal(state_v, ds["v"].values[0, ens, ...].transpose())
    return da


def test_load_state(da: DataAssimilation):
    """
    load first guess files into the state
    """
    # check that the data from file one ended up in the correct spot
    pass