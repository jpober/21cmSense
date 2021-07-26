import pytest

import numpy as np
from astropy import units

from py21cmsense import GaussianBeam, beam


def test_abc():
    with pytest.raises(TypeError):
        beam.PrimaryBeam(150.0)


def test_gaussian_beam():
    bm = GaussianBeam(150.0, dish_size=14)

    assert bm.frequency == 150.0 * units.MHz
    assert bm.dish_size == 14 * units.m

    bm = GaussianBeam(0.15 * units.GHz, dish_size=1400 * units.cm)

    assert bm.frequency == 150.0 * units.MHz
    assert bm.dish_size == 14 * units.m

    assert not hasattr(bm.dish_size_in_lambda(), "units")
    assert bm.area() == bm.area(150.0)

    with pytest.raises(NotImplementedError):
        GaussianBeam.from_uvbeam()

    assert bm.uv_resolution == bm.dish_size_in_lambda()
    assert bm.sq_area() < bm.area()
    assert bm.fwhm() > bm.width()
    assert bm.first_null() < np.pi * units.rad / 2
