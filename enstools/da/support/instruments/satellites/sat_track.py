from skyfield.api import load
import skyfield.sgp4lib as sgp4
import numpy as np


def comp_sat(epoch, time_range, b_coef, b_star, incl, r_asc, ecc, arg_per, m_anom, m_motion):
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
    b_star: float
       Drag Term aka Radiation Pressure Coefficient in units of inverse Earth radii.
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
    ts = load.timescale()

    # hours, minutes, etc. are converted to a fraction of the reference day
    epoch_dp = epoch.hour/24. + epoch.minute/24./60. + epoch.second/24./60./60. + epoch.microsecond/24./60./60.*1e-6
    epoch_str = "{:s}.{:s}".format(epoch.strftime("%y%j"), str("%.8f" % epoch_dp)[2:10])
    t_range = ts.utc(epoch.year, month=epoch.month, day=epoch.day, hour=epoch.hour,
                     minute=epoch.minute, second=time_range)
    b_coef = "{:8.8f}".format(b_coef)[1:]

    # Check plausibility of input:
    exponent = 1 + int("{:e}".format(b_star)[-3:])
    if exponent > 9:
        raise ValueError("b_star is too big for the SGP4 model")
    elif exponent < -9:
        str_b_star = " 00000-0"
    else:
        if b_star < 0:
            mantissa = "{:e}".format(b_star).replace(".","")[:6]
            str_b_star = "{:s}{:+d}".format(mantissa, exponent)
        elif b_star > 0:
            mantissa = " " + "{:e}".format(b_star).replace(".","")[:5]
            str_b_star = "{:s}{:+d}".format(mantissa, exponent)
        else:
            str_b_star = " 00000-0"

    if not (0 <= incl <= 180):
        raise ValueError("Inclination has to be between 0 and 180 degrees")
    if not (0 <= r_asc <= 360):
        raise ValueError("Right ascension has to be between 0 and 360 degrees")
    if not (0 <= ecc < 1):
        raise ValueError("Eccentricity has to be between 0 and 1 in the SPG4 model")
    ecc = "{:f}".format(ecc)[2:]
    if not (0 <= arg_per <= 360):
        raise ValueError("Argument of perigree has to be between 0 and 360 degrees")

    # Create two line element set:
    l1 = "1 00000U 00000AAA {:s}  {:>8}  00000-0 {:s} 0  0000".format(epoch_str, b_coef, str_b_star)
    l2 = "2 00000 {:08.4f} {:08.4f} {:>7} {:8.4f} {:8.4f} {:11.8f}0000000".format(incl, r_asc, ecc, arg_per, m_anom, m_motion)

    sat = sgp4.EarthSatellite(l1, l2)
    # Compute satellites position:
    return sat.at(t_range)
    

def sat_geoc(epoch, time_range, b_coef, b_star, incl, r_asc, ecc, arg_per, m_anom, m_motion):
    sat_pos = comp_sat(epoch, time_range, b_coef, b_star, incl, r_asc, ecc, arg_per, m_anom, m_motion)
    topos = sat_pos.subpoint()
    ra = topos.longitude
    dec = topos.latitude
    return ra._degrees, dec.degrees 


def sat_xyz(epoch, time_range, b_coef, b_star, incl, r_asc, ecc, arg_per, m_anom, m_motion):
    sat_pos = comp_sat(epoch, time_range, b_coef, b_star, incl, r_asc, ecc, arg_per, m_anom, m_motion)
    x, y, z = sat_pos.position.km
    return x, y, z

    
