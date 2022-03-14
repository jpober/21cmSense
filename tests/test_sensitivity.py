import pytest

import numpy as np
from astropy import units

from py21cmsense import GaussianBeam, Observation, Observatory, PowerSpectrum
from py21cmsense.sensitivity import Sensitivity, _kconverter


@pytest.fixture(scope="module")
def bm():
    return GaussianBeam(150.0 * units.MHz, dish_size=14 * units.m)


@pytest.fixture(scope="module")
def observatory(bm):
    return Observatory(
        antpos=np.array([[0, 0, 0], [14, 0, 0], [28, 0, 0], [70, 0, 0]]) * units.m,
        beam=bm,
    )


@pytest.fixture(scope="module")
def observation(observatory):
    return Observation(observatory=observatory)


def test_units(observation):
    ps = PowerSpectrum(observation=observation)

    assert ps.horizon_buffer.to("littleh/Mpc").unit == units.littleh / units.Mpc
    assert ps.k_21.to("littleh/Mpc").unit == units.littleh / units.Mpc
    assert ps.delta_21.to("mK^2").unit == units.mK**2
    assert callable(ps.p21)
    assert ps.k_min.to("littleh/Mpc").unit == units.littleh / units.Mpc
    assert ps.k_max.to("littleh/Mpc").unit == units.littleh / units.Mpc
    assert ps.k1d.to("littleh/Mpc").unit == units.littleh / units.Mpc
    assert isinstance(ps.power_normalisation(0.1 * units.littleh / units.Mpc), float)
    assert ps.horizon_limit(10).to("littleh/Mpc").unit == units.littleh / units.Mpc

    ps = PowerSpectrum(
        observation=observation,
        k_21=np.array([1, 2, 3]) * units.littleh / units.Mpc,
        delta_21=np.array([1, 2, 3]) * units.mK**2,
    )
    ps2 = PowerSpectrum(
        observation=observation,
        k_21=np.array([1, 2, 3]) / units.Mpc,
        delta_21=np.array([1, 2, 3]) * units.mK**2,
    )

    assert np.all(ps.k_21 < ps2.k_21)


def test_sensitivity_2d(observation):
    ps = PowerSpectrum(observation=observation)
    sense_thermal = ps.calculate_sensitivity_2d(thermal=True, sample=False)
    sense_full = ps.calculate_sensitivity_2d()
    assert all(np.all(sense_thermal[key] <= sense_full[key]) for key in sense_thermal)

    with pytest.raises(ValueError):
        ps.calculate_sensitivity_2d(thermal=False, sample=False)


def test_sensitivity_2d_grid(observation, caplog):
    ps = PowerSpectrum(observation=observation)
    sense_ungridded = ps.calculate_sensitivity_2d(thermal=True, sample=True)
    kperp = (
        np.array([x.value for x in sense_ungridded.keys()])
        * list(sense_ungridded.keys())[0].unit
    )
    sense = ps.calculate_sensitivity_2d_grid(
        kperp_edges=np.linspace(kperp.min().value, kperp.max().value, 10) * kperp.unit,
        kpar_edges=ps.k1d,
    )
    assert sense.shape == (9, len(ps.k1d) - 1)

    ps.calculate_sensitivity_2d_grid(
        kperp_edges=np.linspace(ps.k_21.min().value / 2, ps.k_21.max().value * 2, 10)
        * ps.k_21.unit,
        kpar_edges=ps.k_21 / 2,
    )
    assert "minimum kbin is being restricted" in caplog.text
    assert "maximum kbin is being restricted" in caplog.text


def test_sensitivity_1d_binned(observation):
    ps = PowerSpectrum(observation=observation)
    assert np.all(
        ps.calculate_sensitivity_1d() == ps.calculate_sensitivity_1d_binned(ps.k1d)
    )


def test_plots(observation):
    # this is a dumb test, just checking that it doesn't error.
    ps = PowerSpectrum(observation=observation)
    sense2d = ps.calculate_sensitivity_2d()
    ps.plot_sense_2d(sense2d)


def test_sensitivity_optimistic(observation):
    ps = PowerSpectrum(observation=observation, foreground_model="optimistic")
    assert ps.horizon_limit(10.0) > ps.horizon_limit(5.0)


def test_limited_k_range(observation, caplog):
    ps = PowerSpectrum(
        observation=observation,
        k_21=np.array([1, 2, 3]) * units.littleh / units.Mpc,
        delta_21=np.array([1, 2, 3]) * units.mK**2,
    )

    ps.k1d

    assert any(
        "The minimum k value is being restricted" in rec.msg for rec in caplog.records
    )


def test_infs_in_trms(observation):
    # default dumb layout should have lots of infs..
    assert np.any(np.isinf(observation.Trms))
    ps = PowerSpectrum(observation=observation)
    ps.calculate_sensitivity_2d()
    # merely get through the calculations...


def test_write_to_custom_filename(observation, tmp_path):
    out = tmp_path / "outfile.h5"
    ps = PowerSpectrum(observation=observation)
    out2 = ps.write(filename=out)
    assert out2 == out


def test_kconverter():
    with pytest.raises(ValueError, match="no units supplied!"):
        _kconverter(1)


def test_load_yaml_bad():
    with pytest.raises(
        ValueError,
        match="yaml_file must be a string filepath or a raw dict from such a file",
    ):
        Sensitivity.from_yaml(1)


def test_track(observatory):
    """Test that setting `track` is the same as setting obs_duration."""
    obs1 = Observation(observatory=observatory, obs_duration=1 * units.hour)
    obs2 = Observation(observatory=observatory, track=1 * units.hour)

    assert np.all(obs1.uv_coverage == obs2.uv_coverage)
