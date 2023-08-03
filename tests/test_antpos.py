import pytest

import numpy as np
from astropy import units as un

from py21cmsense.antpos import hera


@pytest.mark.parametrize("n", [3, 5, 8, 1])
def test_hera_split_core(n):
    # https://en.wikipedia.org/wiki/Centered_hexagonal_number
    # 3*n^2 - 3n + 1
    antpos1 = hera(hex_num=n)
    assert len(antpos1) == 3 * n**2 - 3 * n + 1
    antpos2 = hera(hex_num=n, split_core=True)
    assert len(antpos2) == len(antpos1) - n


@pytest.mark.parametrize("n", [3, 5, 8, 1])
def test_hera_outriggers(n):
    # https://en.wikipedia.org/wiki/Centered_hexagonal_number
    # 3*n^2 - 3n + 1
    antpos1 = hera(hex_num=n)
    assert len(antpos1) == 3 * n**2 - 3 * n + 1
    antpos2 = hera(hex_num=n, outriggers=1)
    assert len(antpos2) >= len(antpos1)


def test_hera_set_row_sep():
    antpos1 = hera(4)
    antpos2 = hera(4, row_separation=14 * np.sin(np.pi / 3) * un.m)

    assert np.allclose(antpos1, antpos2)

    antpos3 = hera(4, row_separation=12.12 * un.m)
    assert not np.allclose(antpos1, antpos3)
