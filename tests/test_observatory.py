import pytest

import numpy as np
import pyuvdata
from astropy import units
from astropy.coordinates import EarthLocation
from pathlib import Path

from py21cmsense import Observatory
from py21cmsense.beam import GaussianBeam


@pytest.fixture(scope="module")
def bm():
    return GaussianBeam(frequency=150.0 * units.MHz, dish_size=14 * units.m)


def test_antpos(bm):
    a = Observatory(antpos=np.zeros((10, 3)) * units.m, beam=bm)
    assert a.antpos.unit == units.m

    assert np.all(a.baselines_metres == 0)

    # If bad units given, should raise error.
    with pytest.raises(units.UnitConversionError):
        Observatory(antpos=np.zeros((10, 3)) * units.s, beam=bm)

    # Need more than one antenna
    with pytest.raises(AssertionError):
        Observatory(antpos=np.zeros((1, 3)) * units.m, beam=bm)


def test_observatory_class(bm):
    a = Observatory(antpos=np.zeros((3, 3)) * units.m, beam=bm)
    b = a.clone()
    assert a == b


def test_Trcv(bm):
    a = Observatory(antpos=np.zeros((3, 3)) * units.m, beam=bm, Trcv=10 * units.mK)
    assert a.Trcv.unit == units.mK


def test_observatory(bm):
    a = Observatory(antpos=np.zeros((3, 3)) * units.m, beam=bm)
    assert a.frequency == bm.frequency
    assert a.baselines_metres.shape == (3, 3, 3)
    assert (
        a.baselines_metres * a.metres_to_wavelengths
    ).unit == units.dimensionless_unscaled
    assert a.baseline_lengths.shape == (3, 3)
    assert np.all(a.baseline_lengths == 0)

    b = Observatory(
        antpos=np.array([[0, 0, 0], [1, 0, 0], [3, 0, 0]]) * units.m, beam=bm
    )
    assert units.isclose(
        b.shortest_baseline / b.metres_to_wavelengths, 1 * units.m, rtol=1e-3
    )
    assert units.isclose(
        b.longest_baseline / b.metres_to_wavelengths, 3 * units.m, rtol=1e-3
    )
    assert b.observation_duration < 1 * units.day
    assert len(b.get_redundant_baselines()) == 6  # including swapped ones
    with pytest.raises(AssertionError):
        b.time_offsets_from_obs_int_time(b.observation_duration * 1.1)

    assert len(b.time_offsets_from_obs_int_time(b.observation_duration / 1.05)) == 2
    assert units.isclose(
        b.longest_used_baseline() / b.metres_to_wavelengths, 3 * units.m, rtol=1e-3
    )


def test_grid_baselines(bm):
    a = Observatory(
        antpos=np.random.normal(loc=0, scale=50, size=(20, 3)) * units.m, beam=bm
    )
    bl_groups = a.get_redundant_baselines()
    bl_coords = a.baseline_coords_from_groups(bl_groups)
    bl_counts = a.baseline_weights_from_groups(bl_groups)

    with pytest.raises(ValueError):
        a.grid_baselines(bl_coords)

    grid0 = a.grid_baselines()
    grid1 = a.grid_baselines(bl_coords, bl_counts)
    assert np.allclose(grid0, grid1)


def test_min_max_antpos(bm):
    a = Observatory(
        antpos=np.array([np.linspace(0, 50, 11), np.zeros(11), np.zeros(11)]).T
        * units.m,
        beam=bm,
        min_antpos=7 * units.m,
    )

    assert len(a.antpos) == 9

    a = Observatory(
        antpos=np.array([np.linspace(0, 50, 11), np.zeros(11), np.zeros(11)]).T
        * units.m,
        beam=bm,
        max_antpos=10 * units.m,
    )

    assert len(a.antpos) == 2


def test_from_uvdata(bm):
    uv = pyuvdata.UVData()
    uv.antenna_positions = (
        np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [40, 0, 40]]) * units.m
    )
    uv.telescope_location = [
        x.value for x in EarthLocation.from_geodetic(0, 0).to_geocentric()
    ]

    a = Observatory.from_uvdata(uvdata=uv, beam=bm)
    assert np.all(a.antpos == uv.antenna_positions)


def test_different_antpos_loaders(tmp_path: Path):
    antpos = np.array([[0, 0, 0], [14, 0, 0], [28, 0, 0], [70, 0, 0]])

    np.save(tmp_path / "antpos.npy", antpos)
    np.savetxt(tmp_path / "antpos.txt", antpos)

    beamtxt = """
    beam:
        class: GaussianBeam
        frequency: !astropy.units.Quantity
            unit: !astropy.units.Unit {unit: MHz}
            value: 150
        dish_size: !astropy.units.Quantity
            unit: !astropy.units.Unit {unit: m}
            value: 14.0
    """

    yamlnpy = """
    antpos: !astropy.units.Quantity
        unit: !astropy.units.Unit {{unit: m}}
        value: !npy {}/antpos.npy
    {}
    """.format(
        tmp_path,
        beamtxt,
    )

    with open(tmp_path / "npy.yml", "w") as fl:
        fl.write(yamlnpy)

    obsnpy = Observatory.from_yaml(tmp_path / "npy.yml")

    yamltxt = """
    antpos: !astropy.units.Quantity
        unit: !astropy.units.Unit {{unit: m}}
        value: !txt {}/antpos.txt
    {}
    """.format(
        tmp_path,
        beamtxt,
    )
    with open(tmp_path / "txt.yml", "w") as fl:
        fl.write(yamltxt)

    obstxt = Observatory.from_yaml(tmp_path / "txt.yml")

    assert obsnpy == obstxt


def test_longest_used_baseline(bm):
    a = Observatory(
        antpos=np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]]) * units.m, beam=bm
    )

    assert np.isclose(
        a.longest_used_baseline() / a.metres_to_wavelengths, 2 * units.m, atol=1e-3
    )
    assert np.isclose(
        a.longest_used_baseline(bl_max=1.5 * units.m) / a.metres_to_wavelengths,
        1 * units.m,
        atol=1e-4,
    )


def test_from_yaml(bm):
    obs = Observatory.from_yaml(
        {
            "antpos": np.random.random((20, 3)) * units.m,
            "beam": {
                "class": "GaussianBeam",
                "frequency": 150 * units.MHz,
                "dish_size": 14 * units.m,
            },
        }
    )
    assert obs.beam == bm

    with pytest.raises(ValueError):
        Observatory.from_yaml(3)
