from skyfield.api import load
import datetime as dt
import skyfield.sgp4lib as sgp4
import numpy as np

ts = load.timescale()


def comp_sat(epoch, time_range, b_coef, incl, r_asc, ecc, arg_per, m_anom, m_motion):
    """
    Calculates the position of a satellite for given orbital elements and points in time.

    Parameters
    ----------
    epoch: datetime.datetime
        The point in time of the given orbital elements (UTC).
    time_range: np.array
        Contains the points of time in seconds since the epoch where the position has to be calculated.
    b_coef: float
        First Derivative of Mean Motion aka the Ballistic Coefficient in 1/day^2.
    incl: float
        Orbital inclination in degrees.
    r_asc: float
        Right Ascension of the Ascending Node in degrees.
    ecc: float
        Orbital eccentricity.
    arg_per: float
        Argument of Perigee in degrees.
    m_anom: float
        Mean anomaly in degrees.
    m_motion: float
        Mean motion in revolutions/day.

    Returns
    -------
    longitude: np.ndarray
        Longitude of the satellite for the given point in time in degrees.

    latitude: np.ndarray
        Latitude of the satellite for the given point in time in degrees.

    altitude: np.ndarray
        Altitude of the satellite for the given point in time in kilometers from earths geocenter.

    """
    epoch_dp = epoch.hour/24. + epoch.minute/24./60. + epoch.second/24./60./60. + epoch.microsecond/24./60./60.*1e-6
    t_range = ts.utc(epoch.year, month=epoch.month, day=epoch.day, hour=epoch.hour,
                     minute=epoch.minute, second=time_range)
    epoch_str = "{:s}.{:s}".format(epoch.strftime("%y%j"), str(epoch_dp)[2:10])
    b_coef = "{:8.8f}".format(b_coef)[1:]
    ecc = str(ecc)[2:]

    l1 = "1 00000U 00000AAA {:s}  {:>8}  00000-0  12236-3 0  0000".format(epoch_str, b_coef)
    l2 = "2 00000 {:08.4f} {:08.4f} {:>7} {:8.4f} {:8.4f} {:11.8f}0000000".format(incl, r_asc, ecc, arg_per, m_anom, m_motion)

    sat = sgp4.EarthSatellite(l1, l2)
    ra, dec, alt = sat.at(t_range).radec()
    return ra._degrees, dec.degrees, alt.km


