import skyfield.sgp4lib as sgp4
from skyfield.constants import RAD2DEG, pi
from .sat_track import sat_geoc, sat_xyz
from datetime import datetime, timezone
import numpy as np


def aeolus_like_coordinates(start_time: datetime, end_time: datetime, start_lon: float = 0, anomaly: float = 0,
                            include_metric: bool = False):
    """
    Calculate coordinates for a satellite that has a track like the real AEOLUS, but not starting from the real track.

    The calculation is based on this real position and orbital elements:
    1 43600U 18066A   20328.59243159  .00047717  00000-0  18606-3 0  9997
    2 43600  96.7201 332.5980 0007354  96.3422 263.8689 15.86918300130608
    (http://celestrak.com/NORAD/elements/active.txt)

    Parameters
    ----------
    start_time: datetime
                Start time to the calculated track.

    end_time:   datetime
                End time for the calculated track.

    start_lon:  float
                longitude of the first overpass over the equator.

    anomaly:    float
                distance from the equator along the satellite track in degrees

    include_metric: bool
                include metric coordinates of the satellite for the purpose of plotting

    Returns
    -------
    tuple
                (lon, lat, alt) Position of the satellite in 3km steps.
    """

    # make sure, that the start time is in UTC
    start_time = start_time.astimezone(timezone.utc)
    end_time = end_time.astimezone(timezone.utc)

    # orbital elements of the real satellite
    # retrieved on 2020-11-24
    l1 = "1 43600U 18066A   20328.59243159  .00047717  00000-0  18606-3 0  9997"
    l2 = "2 43600  96.7201 332.5980 0007354  96.3422 263.8689 15.86918300130608"
    sat = sgp4.EarthSatellite(l1, l2, name="AEOLUS")
    xpdotp = 1440.0 / (2.0 * pi)
    b_star = sat.model.bstar
    b_coef = sat.model.ndot * (xpdotp * 1440.0)
    incl = sat.model.inclo * RAD2DEG
    # r_asc = sat.model.nodeo * RAD2DEG
    ecc = sat.model.ecco
    arg_per = sat.model.argpo * RAD2DEG
    # m_anom = sat.model.mo * RAD2DEG
    m_motion = sat.model.no_kozai * xpdotp

    # calculate parameters, that would move the satellite to the coordinates 0°E, 0°N for the given start_time
    r_asc = 0
    m_anom = -arg_per
    t_diff = start_time - sat.epoch.utc_datetime()
    lon, lat = sat_geoc(sat.epoch.utc_datetime(), t_diff.total_seconds(), b_coef, b_star, incl, r_asc, ecc, arg_per, m_anom, m_motion)
    while abs(lon) > 0.001 or abs(lat) > 0.001:
        r_asc -= lon
        if r_asc < 0.0:
            r_asc = 360.0 + r_asc
        m_anom -= lat
        lon, lat = sat_geoc(sat.epoch.utc_datetime(), t_diff.total_seconds(), b_coef, b_star, incl, r_asc, ecc, arg_per, m_anom, m_motion)

    # add start offset
    r_asc += start_lon
    if r_asc > 360.0:
        r_asc -= 360.0
    elif r_asc < 0.0:
        r_asc += 360
    m_anom += anomaly

    # measurement cycles take 7s
    time_to_measure = np.arange(0, (end_time - start_time).total_seconds() + 1, 7)
    lon, lat = sat_geoc(sat.epoch.utc_datetime(), t_diff.total_seconds() + time_to_measure, b_coef, b_star, incl, r_asc, ecc, arg_per, m_anom, m_motion)

    # calculate metric coordinates of the satellite as well.
    if include_metric:
        x, y, z = sat_xyz(sat.epoch.utc_datetime(), t_diff.total_seconds() + time_to_measure, b_coef, b_star, incl, r_asc, ecc, arg_per, m_anom, m_motion)
        return lon, lat, x, y, z
    else:
        return lon, lat