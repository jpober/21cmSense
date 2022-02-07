"""Some global configuration options for 21cmSense."""
from astropy.cosmology import Planck15 as _Planck15

PROGRESS = True  # whether to display progress bars for some calculations.
COSMO = _Planck15
