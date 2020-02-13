import pytest
from enstools.da.support import FeedbackFile
from enstools.misc import generate_coordinates
from enstools.mpi import onRank0
import numpy as np


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


def test_add_observation_from_model_output(ff: FeedbackFile, comm):
    """
    use model output file to create observations within the feedback file
    """
    if not onRank0(comm):
        return

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
        variables=["QV"],
        lon=lon,
        lat=lat,
        levels=[100000, 50000]
    )

    # here we should have additional observations for each grid point
    assert ff.data["i_body"].shape[0] == lon.shape[0] * 2
    assert ff.data["obs"].shape[0] > lon.shape[0] * 4
    assert ff.data["obs"].shape[0] < lon.shape[0] * 8

    # entries in i_body and l_body have to match
    i_body = ff.data["i_body"].values
    l_body = ff.data["l_body"].values
    for index in range(i_body.shape[0] - 1):
        assert i_body[index + 1] == i_body[index] + l_body[index]
