"""
A module defining functions to produce antenna positions algorithmically.
Each function here defined may take arbitrary parameters, but must return
a single list or array of 3-tuples, corresponding to (x,y,z) positions
of antennae centred at zero.
"""
import aipy
import numpy as np


def hera(nside, L, dL, l_in_ns=True):
    """

    Parameters
    ----------
    nside
    L
    dL
    l_in_ns

    Returns
    -------

    """
    if l_in_ns:
        L /= aipy.const.len_ns
        dL /= aipy.const.len_ns

    antpos = []
    cen_y, cen_z = 0, 0
    for row in np.arange(nside):
        for cen_x in np.arange((2 * nside - 1) - row):
            dx = row / 2.
            antpos.append(((cen_x + dx) * L, row * dL, cen_z))
            if row != 0:
                antpos.append(((cen_x + dx) * L, -row * dL, cen_z))

    return antpos
