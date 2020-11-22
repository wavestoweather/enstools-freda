from skyfield.api import load
import skyfield.sgp4lib as sgp4


ts = load.timescale()

l1 = "1 43600U 18066A   20101.24998104  .00030597  00000-0  12236-3 0  9996"
l2 = "2 43600  96.7217 108.6657 0006209 115.0686 245.1232 15.86481001 94550"
sat = sgp4.EarthSatellite(l1, l2, name="AEOLUS")

print(sat.at(ts.now()).radec())
