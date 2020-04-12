from skyfield.api import load
import datetime as dt
import skyfield.sgp4lib as sgp4
import numpy as np

ts = load.timescale()

def comp_sat(epoch, time_range, b_coef, incl, r_asc, ecc, arg_per, m_anom, m_motion):
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


b_coef = .00030597

incl = 96.7217
r_asc = 108.6657
ecc = 0.0006209
arg_per = 115.0686
m_anom = 245.1232
m_motion = 15.86481001

timerange = np.arange(0, 864000, 60) # calculate position every 60 seconds for 10 days
ra1, dec1, alt1 = comp_sat(dt.datetime.utcnow(), timerange, b_coef,  
                        incl, r_asc, ecc, arg_per, m_anom, m_motion)
print(ra1)
print(dec1)
print(alt1)
print("\n")
ra2, dec2, alt2 = comp_sat(dt.datetime.utcnow(), timerange, 0,  
                      incl, r_asc, ecc, arg_per, m_anom, m_motion)

print(ra2)
print(dec2)
print(alt2)



