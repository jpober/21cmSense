"""
Utility functions for 21cmSense.
"""
import numpy as np
from astropy import units as un
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time
from pyuvdata import utils as uvutils

from . import config


class UnitError(ValueError):
    pass


def apply_or_convert_unit(unit, allow_unitless=False, array=False):
    """
    Return a function that converts numbers to quantities (or converts quantities to
    specified units).

    Parameters
    ----------
    unit : astropy unit or str

    Returns
    -------
    callable: function which converts its argument to a quantity
    """

    def converter(quantity):
        if array:
            quantity = np.array(quantity)

        if hasattr(quantity, "unit"):
            return quantity.to(unit)
        else:
            if allow_unitless:
                return quantity

            if not config.ALLOW_DEFAULT_UNITS:
                raise UnitError(
                    "This value is required to have units convertible to {}".format(
                        unit
                    )
                )

            return un.Quantity(quantity, unit)

    return converter


def between(min, max):
    """Return an attrs validation function that validates that a number is within certain bounds"""

    def validator(instance, att, val):
        assert min <= val <= max

    return validator


def positive(instance, att, x):
    """attrs validator that checks a value is positive"""
    assert x > 0, "must be positive"


def nonnegative(instance, att, x):
    """attrs validator that checks a value is non-negative"""
    assert x >= 0, "must be non-negative"


def find_nearest(array, value):
    """Find closest value in `array` to `value`"""
    idx = np.abs(array.reshape(-1, 1) - value).argmin(0)
    return idx


def trunc(x, ndecimals=0):
    """Truncate a floating point number to a given number of decimals"""
    decade = 10 ** ndecimals
    return np.trunc(x * decade) / decade


def phase(jd, ra, dec, telescope_location, uvws0):
    """
    Compute UVWs phased to a given RA/DEC at a particular epoch.

    This function was copied from the pyuvdata.UVData.phase method, and modified to be
    simpler.

    Parameters
    ----------
    jd : float
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

    obs_time = Time(jd, format="jd")
    telescope_loc_xyz = uvutils.XYZ_from_LatLonAlt(
        telescope_location.lat.rad,
        telescope_location.lon.rad,
        telescope_location.height,
    )

    itrs_telescope_location = SkyCoord(
        x=telescope_loc_xyz[0] * un.m,
        y=telescope_loc_xyz[1] * un.m,
        z=telescope_loc_xyz[2] * un.m,
        frame="itrs",
        obstime=obs_time,
    )

    frame_telescope_location = itrs_telescope_location.transform_to("icrs")
    frame_telescope_location.representation_type = "cartesian"

    uvw_ecef = uvutils.ECEF_from_ENU(
        uvws0,
        telescope_location.lat.rad,
        telescope_location.lon.rad,
        telescope_location.height,
    )

    itrs_uvw_coord = SkyCoord(
        x=uvw_ecef[:, 0] * un.m,
        y=uvw_ecef[:, 1] * un.m,
        z=uvw_ecef[:, 2] * un.m,
        frame="itrs",
        obstime=obs_time,
    )
    frame_uvw_coord = itrs_uvw_coord.transform_to("icrs")

    # this takes out the telescope location in the new frame,
    # so these are vectors again
    frame_rel_uvw = (
        frame_uvw_coord.cartesian.get_xyz().value.T
        - frame_telescope_location.cartesian.get_xyz().value
    )

    uvws = uvutils.phase_uvw(
        frame_phase_center.ra.rad, frame_phase_center.dec.rad, frame_rel_uvw
    )
    return uvws


def phase_past_zenith(time_past_zenith, uvws0, latitude):
    """
    Compute UVWs phased to a point which has rotated from zenith by a certain amount
    of time.

    This function specifies a longitude and time of observation without loss of generality
    -- all that matters is the time since a hypothetical point was at zenith, and the
    latitude of the array.

    Parameters
    ----------
    time_past_zenith : float or Quantity
        The time passed since the point was at zenith. If float, assumed to be in units
        of days.
    uvws0 : array
        The UVWs when phased to zenith.
    latitude : float or Quantity
        The latitude of the center of the array, in radians.

    Returns
    -------

    """
    # Generate ra/dec of zenith at time in the phase_frame coordinate system
    # to use for phasing
    time_past_zenith = apply_or_convert_unit("day")(time_past_zenith)
    latitude = apply_or_convert_unit("rad")(latitude)
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
