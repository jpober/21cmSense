"""A module defining functions to produce antenna positions algorithmically.

Each function here defined may take arbitrary parameters, but must return
a single array of shape (Nant, 3) with units of meters, corresponding to (x,y,z) positions
of antennae centred at zero.
"""
from __future__ import annotations

import numpy as np
from astropy import units as un
from typing import Optional

from . import types as tp
from . import yaml


@yaml.yaml_func()
def hera(
    hex_num,
    separation: tp.Length = 14 * un.m,
    split_core: bool = False,
    outriggers: bool = False,
    row_separation: tp.Length | None = None,
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
    split_core
        Whether to split the hex into three offset segments.
    outriggers
        Whether to add outrigger antennas.

    Returns
    -------
    antpos
        A 2D array of antenna positions, shape ``(Nants, 3)``.
    """
    sep = separation.to_value("m")

    if row_separation is None:
        row_sep = sep * np.sqrt(3) / 2
    else:
        row_sep = row_separation.to_value("m")

    # construct the main hexagon
    positions = []
    for row in range(hex_num - 1, -hex_num + split_core, -1):
        # adding split_core deletes a row if it's true
        for col in range(2 * hex_num - abs(row) - 1):
            x_pos = sep * ((2 - (2 * hex_num - abs(row))) / 2 + col)
            y_pos = row * row_sep
            positions.append([x_pos, y_pos, 0])

    # basis vectors (normalized to sep)
    up_right = sep * np.asarray([0.5, row_sep / sep, 0])
    up_left = sep * np.asarray([-0.5, row_sep / sep, 0])

    # split the core if desired
    if split_core:
        new_pos = []
        for pos in positions:
            # find out which sector the antenna is in
            theta = np.arctan2(pos[1], pos[0])
            if pos[0] == 0 and pos[1] == 0:
                new_pos.append(pos)
            elif -np.pi / 3 < theta < np.pi / 3:
                new_pos.append(np.asarray(pos) + (up_right + up_left) / 3)
            elif np.pi / 3 <= theta < np.pi:
                new_pos.append(np.asarray(pos) + up_left - (up_right + up_left) / 3)
            else:
                new_pos.append(pos)

        # update the positions
        positions = new_pos

    # add outriggers if desired
    if outriggers:
        # The specific displacements of the outrigger sectors are
        # designed specifically for redundant calibratability and
        # "complete" uv-coverage, but also to avoid specific
        # obstacles on the HERA site (e.g. a road to a MeerKAT antenna)
        exterior_hex_num = outriggers + 2
        for row in range(exterior_hex_num - 1, -exterior_hex_num, -1):
            for col in range(2 * exterior_hex_num - abs(row) - 1):
                x_pos = (
                    ((2 - (2 * exterior_hex_num - abs(row))) / 2 + col)
                    * sep
                    * (hex_num - 1)
                )
                y_pos = row * (hex_num - 1) * row_sep
                theta = np.arctan2(y_pos, x_pos)
                if np.sqrt(x_pos**2 + y_pos**2) > sep * (hex_num + 1):
                    if 0 < theta <= 2 * np.pi / 3 + 0.01:
                        positions.append(
                            np.asarray([x_pos, y_pos, 0]) - 4 * (up_right + up_left) / 3
                        )
                    elif 0 >= theta > -2 * np.pi / 3:
                        positions.append(
                            np.asarray([x_pos, y_pos, 0]) - 2 * (up_right + up_left) / 3
                        )
                    else:
                        positions.append(
                            np.asarray([x_pos, y_pos, 0]) - 3 * (up_right + up_left) / 3
                        )

    return np.array(positions) * un.m
