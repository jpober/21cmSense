import pytest

import numpy as np
from astropy import units

from py21cmsense import GaussianBeam, Observation, Observatory, PowerSpectrum


@pytest.fixture(scope="module")
def bm():
    return GaussianBeam(150.0, dish_size=14)


@pytest.fixture(scope="module")
def observatory(bm):
    return Observatory(
        antpos=np.array([[0, 0, 0], [14, 0, 0], [28, 0, 0], [70, 0, 0]]), beam=bm
    )


@pytest.fixture(scope="module")
def observation(observatory):
    return Observation(observatory=observatory)


def test_units(observation):
    ps = PowerSpectrum(observation=observation)

    assert ps.horizon_buffer.to("littleh/Mpc").unit == units.littleh / units.Mpc
    assert ps.k_21.to("littleh/Mpc").unit == units.littleh / units.Mpc
    assert ps.delta_21.to("mK^2").unit == units.mK ** 2
    assert callable(ps.p21)
    assert ps.k_min.to("littleh/Mpc").unit == units.littleh / units.Mpc
    assert ps.k_max.to("littleh/Mpc").unit == units.littleh / units.Mpc
    assert ps.k1d.to("littleh/Mpc").unit == units.littleh / units.Mpc
    assert ps.power_normalisation(0.1).unit == units.dimensionless_unscaled
    assert ps.horizon_limit(10).to("littleh/Mpc").unit == units.littleh / units.Mpc

    ps = PowerSpectrum(
        observation=observation,
        k_21=np.array([1, 2, 3]) * units.littleh / units.Mpc,
        delta_21=np.array([1, 2, 3]) * units.mK ** 2,
    )
    ps2 = PowerSpectrum(
        observation=observation,
        k_21=np.array([1, 2, 3]) / units.Mpc,
        delta_21=np.array([1, 2, 3]) * units.mK ** 2,
    )

    assert np.all(ps.k_21 < ps2.k_21)


def test_sensitivity_2d(observation):
    ps = PowerSpectrum(observation=observation)
    sense_thermal = ps.calculate_sensitivity_2d(thermal=True, sample=False)
    sense_full = ps.calculate_sensitivity_2d()
    assert all(np.all(sense_thermal[key] <= sense_full[key]) for key in sense_thermal)

    with pytest.raises(ValueError):
        ps.calculate_sensitivity_2d(thermal=False, sample=False)


def test_plots(observation):
    # this is a dumb test, just checking that it doesn't error.
    ps = PowerSpectrum(observation=observation)
    sense2d = ps.calculate_sensitivity_2d()
    ps.plot_sense_2d(sense2d)
