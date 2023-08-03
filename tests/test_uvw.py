"""Tests of the phasing code for calculating UVWs."""
import pytest

import numpy as np
from astropy import units as un
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time
from pyuvdata import utils as uvutils

from py21cmsense._utils import phase_past_zenith


@pytest.mark.parametrize("lat", [-1.0, -0.5, 0, 0.5, 1.0])
@pytest.mark.parametrize("use_apparent", [True, False])
def test_phase_at_zenith(lat, use_apparent):
    bls_enu = np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
        ]
    )

    uvws = phase_past_zenith(
        time_past_zenith=0.0 * un.day,
        bls_enu=bls_enu,
        latitude=lat * un.rad,
        use_apparent=use_apparent,
    )

    assert np.allclose(np.squeeze(uvws), bls_enu, atol=5e-3)


@pytest.mark.parametrize("use_apparent", [True, False])
def test_phase_past_zenith(use_apparent):
    bls_enu = np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
        ]
    )

    # Almost rotated to the horizon.
    uvws = np.squeeze(
        phase_past_zenith(
            time_past_zenith=0.2 * un.day,
            bls_enu=bls_enu,
            latitude=0 * un.rad,
            use_apparent=use_apparent,
        )
    )

    assert uvws[0][0] < 0.35  # Much foreshortened
    assert np.isclose(uvws[1][1], 1)  # N-S direction doesn't get foreshortened.


def test_phase_past_zenith_shape():
    bls_enu = np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
            [1, 0, 0],
            [0, 10, 0],
            [10, 0, 0],
        ]
    )

    times = np.array([0, 0.1, 0, 0.1]) * un.day

    # Almost rotated to the horizon.
    uvws = phase_past_zenith(
        time_past_zenith=times, bls_enu=bls_enu, latitude=0 * un.rad
    )

    assert uvws.shape == (5, 4, 3)
    assert np.allclose(uvws[0], uvws[2])  # Same baselines
    assert np.allclose(uvws[:, 0], uvws[:, 2])  # Same times
    assert np.allclose(uvws[:, 1], uvws[:, 3])  # Same times


@pytest.mark.parametrize("lat", [-1.0, -0.5, 0, 0.5, 1.0])
def test_use_apparent(lat):
    bls_enu = np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
        ]
    )

    times = np.linspace(-1, 1, 3) * un.hour

    # Almost rotated to the horizon.
    uvws = phase_past_zenith(
        time_past_zenith=times, bls_enu=bls_enu, latitude=lat * un.rad
    )
    uvws0 = phase_past_zenith(
        time_past_zenith=times,
        bls_enu=bls_enu,
        latitude=lat * un.rad,
        use_apparent=True,
    )

    np.testing.assert_allclose(uvws, uvws0, atol=1e-2)


@pytest.mark.parametrize("lat", [-1.0, -0.5, 0, 0.5, 1.0])
@pytest.mark.parametrize("time_past_zenith", [-1 * un.hour, 0 * un.hour, 1 * un.hour])
def test_calc_app_coords(lat, time_past_zenith):
    # Generate ra/dec of zenith at time in the phase_frame coordinate system
    # to use for phasing
    telescope_location = EarthLocation.from_geodetic(lon=0, lat=lat * un.rad)

    # JD is arbitrary
    jd = 2454600

    zenith_coord = SkyCoord(
        alt=90 * un.deg,
        az=0 * un.deg,
        obstime=Time(jd, format="jd"),
        frame="altaz",
        location=telescope_location,
    )
    zenith_coord = zenith_coord.transform_to("icrs")

    obstime = zenith_coord.obstime + time_past_zenith

    ra = zenith_coord.ra.to_value("rad")
    dec = zenith_coord.dec.to_value("rad")
    app_ra, app_dec = uvutils.calc_app_coords(
        ra, dec, time_array=obstime.utc.jd, telescope_loc=telescope_location
    )

    assert np.isclose(app_ra, ra, atol=0.02)  # give it 1 degree wiggle room.
    assert np.isclose(app_dec, dec, atol=0.02)
