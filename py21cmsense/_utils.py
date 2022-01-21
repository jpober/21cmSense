"""Utility functions for 21cmSense."""
import numpy as np
import tqdm
from astropy import units as un
from astropy.coordinates import ICRS, EarthLocation, SkyCoord
from astropy.time import Time
from pyuvdata import utils as uvutils

from . import config


def between(xmin, xmax):
    """Return an attrs validation function that checks a number is within bounds."""

    def validator(instance, att, val):
        assert xmin <= val <= xmax

    return validator


def positive(instance, att, x):
    """An attrs validator that checks a value is positive."""
    assert x > 0, "must be positive"


def nonnegative(instance, att, x):
    """An attrs validator that checks a value is non-negative."""
    assert x >= 0, "must be non-negative"


def find_nearest(array, value):
    """Find closest value in `array` to `value`."""
    return np.abs(array.reshape(-1, 1) - value).argmin(0)


def trunc(x, ndecimals=0):
    """Truncate a floating point number to a given number of decimals."""
    decade = 10 ** ndecimals
    return np.trunc(x * decade) / decade


def phase(jd, ra, dec, telescope_location, uvws0):
    """
    Compute UVWs phased to a given RA/DEC at a particular epoch.

    This function was copied from the pyuvdata.UVData.phase method, and modified to be
    simpler.

    Parameters
    ----------
    jd : float or array_like of float
        The Julian date of the observation.
    ra : float
        The ra to phase to in radians.
    dec : float
        The dec to phase to in radians.
    telescope_location : :class:`astropy.coordinates.EarthLocation`
        The location of the reference point of the telescope, in geodetic frame (i.e.
        it has lat, lon, height attributes).
    uvws0 : array
        The UVWs when phased to zenith.

    Returns
    -------
    uvws : array
        Array of the same shape as `uvws0`, with entries modified to the new phase
        center.
    """
    frame_phase_center = SkyCoord(ra=ra, dec=dec, unit="radian", frame="icrs")

    obs_time = Time(np.atleast_1d(jd), format="jd")

    itrs_telescope_location = telescope_location.get_itrs(obstime=obs_time)

    frame_telescope_location = itrs_telescope_location.transform_to(ICRS)
    frame_telescope_location.representation_type = "cartesian"

    uvw_ecef = uvutils.ECEF_from_ENU(
        uvws0,
        telescope_location.lat.rad,
        telescope_location.lon.rad,
        telescope_location.height,
    )
    unique_times, r_inds = np.unique(obs_time, return_inverse=True)
    uvws = np.zeros((uvw_ecef.shape[0], unique_times.size, 3), dtype=np.float64)
    for ind, jd in tqdm.tqdm(
        enumerate(unique_times),
        desc="computing UVWs",
        total=len(unique_times),
        unit="times",
        disable=not config.PROGRESS or unique_times.size == 1,
    ):
        itrs_uvw_coord = SkyCoord(
            x=uvw_ecef[:, 0] * un.m,
            y=uvw_ecef[:, 1] * un.m,
            z=uvw_ecef[:, 2] * un.m,
            frame="itrs",
            obstime=jd,
        )
        frame_uvw_coord = itrs_uvw_coord.transform_to("icrs")

        # this takes out the telescope location in the new frame,
        # so these are vectors again
        frame_rel_uvw = (
            frame_uvw_coord.cartesian.get_xyz().value.T
            - frame_telescope_location[ind].cartesian.get_xyz().value
        )

        uvws[:, ind, :] = uvutils.phase_uvw(
            frame_phase_center.ra.rad, frame_phase_center.dec.rad, frame_rel_uvw
        )
    return uvws[:, r_inds, :]


@un.quantity_input
def phase_past_zenith(time_past_zenith: un.day, uvws0: np.ndarray, latitude: un.rad):
    """Compute UVWs phased to a point rotated from zenith by a certain amount of time.

    This function specifies a longitude and time of observation without loss of
    generality -- all that matters is the time since a hypothetical point was at zenith,
    and the latitude of the array.

    Parameters
    ----------
    time_past_zenith
        The time passed since the point was at zenith. If float, assumed to be in units
        of days.
    uvws0 : array
        The UVWs when phased to zenith.
    latitude
        The latitude of the center of the array, in radians.

    Returns
    -------
    uvws
        The array of UVWs correctly phased.
    """
    # Generate ra/dec of zenith at time in the phase_frame coordinate system
    # to use for phasing
    telescope_location = EarthLocation.from_geodetic(lon=0, lat=latitude)

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

    # Get the RA that was the meridian at jd - time_past_zenith

    return phase(
        jd + time_past_zenith.value,
        zenith_coord.ra.rad,
        zenith_coord.dec.rad,
        telescope_location,
        uvws0,
    )
