import numpy as np
import datetime as dt

from sat_track import comp_sat



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



