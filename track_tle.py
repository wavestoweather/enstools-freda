from skyfield.api import load
import skyfield.sgp4lib as sgp4




ts = load.timescale()

# 1 43600U 18066A   20101.24998104  .00030597  00000-0  12236-3 0  9996
# 2 43600  96.7217 108.6657 0006209 115.0686 245.1232 15.86481001 94550

epoch_y = 20
epoch_d = 101.24998104
b_coef = .00030597
bstar = "12236-3"

incl = 96.7217
r_asc = 108.6657
ecc = 0.0006209
arg_per = 115.0686
m_anom = 245.1232
m_motion = 15.86481001

b_coef = str(b_coef)[1:]
ecc = str(ecc)[2:]

l1 = "1 00000U 00000AAA {:2d}{:3.8f}  {:>8}  00000-0  {} 0  0000".format(epoch_y, epoch_d, b_coef, bstar)
l2 = "2 00000 {:08.4f} {:08.4f} {:>7} {:8.4f} {:8.4f} {:11.8f}0000000".format(incl, r_asc, ecc, arg_per, m_anom, m_motion)
print(l1)
print(l2)

sat = sgp4.EarthSatellite(l1, l2, name="AEOLUS")

print(sat.at(ts.now()).radec())
