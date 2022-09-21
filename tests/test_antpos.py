import pytest

import numpy as np

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
