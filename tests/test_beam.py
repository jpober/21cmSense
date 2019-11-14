import pytest

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
