"""Utility functions for 21cmSense."""
import numpy as np
from astropy import units as un
from astropy.coordinates import EarthLocation, SkyCoord
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


@un.quantity_input
def phase_past_zenith(
    time_past_zenith: un.day, bls_enu: np.ndarray, latitude, use_apparent: bool = True
):
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

    obstimes = zenith_coord.obstime + time_past_zenith
    lsts = obstimes.sidereal_time("apparent", longitude=0.0).rad

    if not hasattr(lsts, "__len__"):
        lsts = np.array([lsts])

    if use_apparent:
        app_ra, app_dec = uvutils.calc_app_coords(
            zenith_coord.ra.to_value("rad"),
            zenith_coord.dec.to_value("rad"),
            time_array=obstimes.utc.jd,
            telescope_loc=telescope_location,
        )

        app_ra = np.tile(app_ra, len(bls_enu))
        app_dec = np.tile(app_dec, len(bls_enu))

    else:
        app_ra = zenith_coord.ra.to_value("rad") * np.ones(len(bls_enu) * len(lsts))
        app_dec = zenith_coord.dec.to_value("rad") * np.ones(len(bls_enu) * len(lsts))

    # Now make everything nbls * ntimes big.
    _lsts = np.tile(lsts, len(bls_enu))
    uvws = np.repeat(bls_enu, len(lsts), axis=0)

    out = uvutils.calc_uvw(
        app_ra=app_ra,
        app_dec=app_dec,
        lst_array=_lsts,
        uvw_array=uvws,
        telescope_lat=latitude.to_value("rad"),
        telescope_lon=0.0,
        from_enu=True,
        use_ant_pos=False,
    )
    return out.reshape((len(bls_enu), len(lsts), 3))
