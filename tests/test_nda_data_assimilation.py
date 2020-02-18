import pytest
import numpy as np
import scipy.spatial
import os
from enstools.io.reader import expand_file_pattern
from enstools.da.nda import DataAssimilation
from enstools.io import read
from enstools.mpi import onRank0
from enstools.mpi.grids import UnstructuredGrid
from enstools.core.tempdir import TempDir


@pytest.fixture
def da(grid_with_overlap: UnstructuredGrid, ff_file: str, comm):
    """
    Initialize the DataAssimilation object
    """
    da = DataAssimilation(grid_with_overlap)
    da.load_state("/archive/meteo/external-models/dwd/icon/oper/icon_oper_eps_gridded-global_rolling/202002/20200201T00/igaf2020020100.m00[1-5].grb")

    # compare uploaded data with input files
    if onRank0(comm):
        ds = read("/archive/meteo/external-models/dwd/icon/oper/icon_oper_eps_gridded-global_rolling/202002/20200201T00/igaf2020020100.m00[1-5].grb")
        assert ds["T"].dims == ("time", "ens", "generalVerticalLayer", "cell")

    # only check variable v=4 on first five ensemble members
    for ens in range(5):
        state_v_start = da.state_variables["V"]["layer_start"]
        state_v_end = da.state_variables["V"]["layer_size"] + state_v_start
        state_v = da.grid.gatherData("state", part=(slice(state_v_start, state_v_end), ens))
        if onRank0(comm):
            np.testing.assert_array_equal(state_v, ds["V"].values[0, ens, ...].transpose())

    # load observations
    da.load_observations(ff_file)
    for var in ["obs", "i_body"]:
        assert var in da.observations

    # check that all reports are contained in non-overlapping subsets. The corresponding information is
    # available everywhere and can also be tested everywhere
    assert da.observations["report_set_indices"].size == da.observations["i_body"].size
    all_reports = set(np.arange(da.observations["i_body"].size))
    all_non_overlapping = set(da.observations["report_set_indices"])
    assert all_reports == all_non_overlapping

    # the number of report sets should be smalle than the number of unique grid indices
    unique_indices = np.unique(da.observations["index_x"])
    assert da.observations["report_sets"].shape[0] <= unique_indices.size

    # check that all reports in report_set_indices are refered to in report_sets
    refered_indices = set()
    for ireport_set in range(da.observations["report_sets"].shape[0]):
        for iindex in range(da.observations["report_sets"][ireport_set, 0], da.observations["report_sets"][ireport_set, 0] + da.observations["report_sets"][ireport_set, 1]):
            refered_indices.add(iindex)
    diff = all_reports - refered_indices
    assert len(diff) == 0
    diff2 = refered_indices - all_reports
    assert len(diff2) == 0

    # check that the observations in each report_set really have no overlapping points
    global_coords = da.grid.gatherData("coordinates_cartesian")
    if onRank0(comm):
        kdtree = scipy.spatial.cKDTree(global_coords)
        for iset in range(da.observations["report_sets"].shape[0]):
            one_set = da.observations["report_sets"][iset, :]
            cells_in_set = set()
            obs_cells_in_set = set()
            for ireport_in_set in range(one_set[0], one_set[0] + one_set[1]):
                # find the actual index of this report
                ireport = da.observations["report_set_indices"][ireport_in_set]
                # do not check reports that are located at the same grid cell, then an other report within the
                # same set of reports
                if da.observations["index_x"][ireport] in obs_cells_in_set:
                    continue
                obs_cells_in_set.add(da.observations["index_x"][ireport])
                # find all neighbours for this report
                neighbours = kdtree.query_ball_point(global_coords[da.observations["index_x"][ireport], :],
                                                     r=da.localization_radius)
                for neighbour in neighbours:
                    assert neighbour not in cells_in_set
                    cells_in_set.add(neighbour)

    comm.barrier()
    return da


def test_save_state(da: DataAssimilation, get_tmpdir: TempDir, comm):
    """
    the da object includes a complete state. Store it into files and read back the content.
    """
    if onRank0(comm):
        output_path = get_tmpdir.getpath()
    else:
        output_path = None
    output_path = comm.tompi4py().bcast(output_path, root=0)
    da.save_state(output_folder=output_path)

    # read the original data and compare with the newly written files
    if onRank0(comm):
        # original data
        ds_orig = read("/archive/meteo/external-models/dwd/icon/oper/icon_oper_eps_gridded-global_rolling/202002/20200201T00/igaf2020020100.m00[1-5].grb")
        # newly written data
        new_files = expand_file_pattern(f"{get_tmpdir.getpath()}/igaf2020020100.m00[1-5].nc")
        for one_file in new_files:
            assert os.path.exists(one_file)
        ds_new = read(new_files)

        # loop over all variables in the new files
        for var in ["P", "QV", "T", "U", "V", "FR_ICE"]:
            orig = np.asarray(ds_orig[var])
            new = np.asarray(ds_new[var])
            assert orig.shape == new.shape
            np.testing.assert_array_equal(orig, new)
    comm.barrier()

