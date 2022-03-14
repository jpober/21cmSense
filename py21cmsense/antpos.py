"""A module defining functions to produce antenna positions algorithmically.

Each function here defined may take arbitrary parameters, but must return
a single array of shape (Nant, 3) with units of meters, corresponding to (x,y,z) positions
of antennae centred at zero.
"""
import numpy as np
from astropy import units as un
from typing import Optional

from . import types as tp
from . import yaml


@yaml.yaml_func()
@un.quantity_input(equivalencies=tp.time_as_distance)
def hera(
    hex_num, separation: tp.Length = 14 * un.m, dl: Optional[tp.Length] = None
) -> tp.Meters:
    """
    Produce a simple regular hexagonal array.

    .. note:: This has no offset for the three parallelograms.

    Parameters
    ----------
    hex_num
        Number of antennas per side of the hexagon
    separation
        The distance between antennas along a side.
        May have units of distance or time, the latter interpreted as a distance travelled
        by light.
    dl
        The distance between rows of antennas.
        May have units of distance or time, the latter interpreted as a distance travelled
        by light. If not provided, assume sin(60) * separation (i.e. equilateral triangles).

    Returns
    -------
    antpos
        A 2D array of antenna positions, shape ``(Nants, 3)``.
    """
    if dl is None:
        dl = np.sin(60) * separation

    separation = separation.to_value("m")
    dl = dl.to_value("m")

    antpos = []
    cen_z = 0
    for row in np.arange(hex_num):
        for cen_x in np.arange((2 * hex_num - 1) - row):
            dx = row / 2.0
            antpos.append(((cen_x + dx) * separation, row * dl, cen_z))
            if row != 0:
                antpos.append(((cen_x + dx) * separation, -row * dl, cen_z))

    return np.array(antpos) * un.m
