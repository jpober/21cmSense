"""A module defining functions to produce antenna positions algorithmically.

Each function here defined may take arbitrary parameters, but must return
a single array of shape (Nant, 3) with units of meters, corresponding to (x,y,z) positions
of antennae centred at zero.
"""
import numpy as np
from astropy import constants as cnst
from astropy import units as un

from . import _utils as ut


def hera(hex_num, separation, dl, units="m") -> np.ndarray:
    """
    Produce a simple regular hexagonal array.

    .. note:: This has no offset for the three parallelograms.

    Parameters
    ----------
    hex_num : int
        Number of antennas per side of the hexagon
    separation : float or Quantity
        The distance between antennas along a side. If float, assumed to be in `units`.
        May have units of distance or time, the latter interpreted as a distance travelled
        by light.
    dl : float or Quantity
        The distance between rows of antennas. If float, assumed to be in `units`.
        May have units of distance or time, the latter interpreted as a distance travelled
        by light.
    units : str or astropy.units.Unit
        The units of `l` and `dl`. If `l` and `dl` are astropy Quantities, this is ignored.
        Must be in a format recognized by astropy.

    Returns
    -------
    antpos
        A 2D array of antenna positions, shape ``(Nants, 3)``.
    """
    separation = ut.apply_or_convert_unit(units)(separation)
    dl = ut.apply_or_convert_unit(units)(dl)

    try:
        separation = separation.to("m")
        dl = dl.to("m")
    except un.UnitConversionError:
        separation = (separation * cnst.c).to("m")
        dl = (dl * cnst.c).to("m")

    antpos = []
    cen_z = 0
    for row in np.arange(hex_num):
        for cen_x in np.arange((2 * hex_num - 1) - row):
            dx = row / 2.0
            antpos.append(((cen_x + dx) * separation.value, row * dl.value, cen_z))
            if row != 0:
                antpos.append(((cen_x + dx) * separation.value, -row * dl.value, cen_z))

    return np.array(antpos) * un.m
