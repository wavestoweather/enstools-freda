import numpy as np
import datetime as dt

from sat_track import comp_sat
from sat_plot import sat_plot
import matplotlib.pyplot as plt
b_star = 0.12236e-3
b_coef = .00030597
incl = 96.7217
r_asc = 108.6657
ecc = 0.0006209
arg_per = 115.0686
m_anom = 245.1232
m_motion = 15.86481001

timerange = np.arange(0, 864000/10, 60) # calculate position every 60 seconds for 10 days
ra, dec = comp_sat(dt.datetime.utcnow(), timerange, b_coef, b_star,
                           incl, r_asc, ecc, arg_per, m_anom, m_motion)

sat_plot(ra,dec)
plt.show()
