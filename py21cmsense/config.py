"""
Some global configuration options for 21cmSense. These are per-session only.
"""
from astropy.cosmology import Planck15 as _p15

ALLOW_DEFAULT_UNITS = True  # whether to allow default units to be applied, otherwise
# exception will be raised if a float is passed where a
# quantity is expected.

PROGRESS = True  # whether to display progress bars for some calculations.
COSMO = _p15  # Cosmology to use throughout (by default)
