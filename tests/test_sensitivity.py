import pytest

import numpy as np
import warnings
from astropy import units
from astropy.cosmology.units import littleh

from py21cmsense import GaussianBeam, Observation, Observatory, PowerSpectrum
from py21cmsense.sensitivity import Sensitivity


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

    assert ps.horizon_buffer.to("littleh/Mpc").unit == littleh / units.Mpc
    assert ps.k1d.to("littleh/Mpc").unit == littleh / units.Mpc
    assert isinstance(ps.power_normalisation(0.1 * littleh / units.Mpc), float)
    assert ps.horizon_limit(10).to("littleh/Mpc").unit == littleh / units.Mpc


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


def test_infs_in_trms(observation):
    # default dumb layout should have lots of infs..
    assert np.any(np.isinf(observation.Trms))
    ps = PowerSpectrum(observation=observation)
    ps.calculate_sensitivity_2d()
    # merely get through the calculations...


def test_write_to_custom_filename(observation, tmp_path):
    out = tmp_path / "outfile.h5"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ps = PowerSpectrum(observation=observation)
        out2 = ps.write(filename=out)
    assert out2 == out


def test_load_yaml_bad():
    with pytest.raises(
        ValueError,
        match="yaml_file must be a string filepath or a raw dict from such a file",
    ):
        Sensitivity.from_yaml(1)

    with pytest.raises(ImportError, match="Could not import"):
        PowerSpectrum.from_yaml(
            {
                "plugins": ["this.is.not.a.module"],
                "observatory": {
                    "antpos": np.random.random((20, 3)) * units.m,
                    "beam": {
                        "class": "GaussianBeam",
                        "frequency": 150 * units.MHz,
                        "dish_size": 14 * units.m,
                    },
                },
            }
        )


def test_systematics_mask(observation):
    ps = PowerSpectrum(
        observation=observation,
        systematics_mask=lambda kperp, kpar: np.zeros(len(kpar), dtype=bool),
    )
    assert len(ps.calculate_sensitivity_2d()) == 0


def test_track(observatory):
    """Test that setting `track` is the same as setting obs_duration."""
    obs1 = Observation(observatory=observatory, obs_duration=1 * units.hour)
    obs2 = Observation(observatory=observatory, track=1 * units.hour)

    assert np.all(obs1.uv_coverage == obs2.uv_coverage)


def test_clone(observation):
    ps = PowerSpectrum(
        observation=observation,
    )

    ps2 = ps.clone()
    assert ps2 == ps


def test_bad_theory(observation):
    with pytest.raises(
        ValueError, match="The theory_model must be an instance of TheoryModel"
    ):
        PowerSpectrum(observation=observation, theory_model=3)
