"""
Common 21 cm conversions.

Provides conversions between observing co-ordinates and cosmological co-ordinates.
"""
import numpy as np
from astropy import constants as cnst
from astropy import units as u
from astropy.cosmology import Planck15

from . import _utils

# The frequency of the 21cm line emission.
f21 = 1.42040575177 * u.GHz


def f2z(fq):
    """
    Convert frequency to redshift for 21 cm line.

    Parameters
    ----------
    fq : float or astropy.Quantity
        If float, it is interpreted as being in GHz.

    Returns
    -------
    dimensionless astropy.Quantity : The redshift
    """
    fq = _utils.apply_or_convert_unit("GHz")(fq)
    return f21 / fq - 1


def z2f(z):
    """
    Convert redshift to z=0 frequency for 21 cm line.

    Parameters
    ----------
    z : float
        Redshift

    Returns
    -------
    astropy.Quantity : the frequency
    """
    return f21 / (1 + z)


def dL_dth(z, cosmo=Planck15):
    """
    Return the factor to convert radians to transverse distance at redshift z.

    Parameters
    ----------
    z : float
        The redshift

    Returns
    -------
    astropy.Quantity : the factor (in Mpc/h/radian) which converts from an angle
        to a transverse distance.

    Notes
    -----
    From Furlanetto et al. (2006)
    """
    return cosmo.h * cosmo.comoving_transverse_distance(z) / u.rad / u.littleh


def dL_df(z, cosmo=Planck15):
    """
    Get the factor to convert bandwidth to line-of-sight distance in Mpc/h.

    Parameters
    ----------
    z : float
        The redshift
    """
    return (cosmo.h * cnst.c * (1 + z) / (z2f(z) * cosmo.H(z) * u.littleh)).to(
        "Mpc/(MHz*littleh)"
    )


def dk_du(z, cosmo=Planck15):
    """
    Get factor converting bl length in wavelengths to h/Mpc.

    Parameters
    ----------
    z : float
        redshift

    Notes
    -----
    Valid for u >> 1
    """
    # from du = 1/dth, which derives from du = d(sin(th)) using the small-angle approx
    return 2 * np.pi / dL_dth(z, cosmo) / u.rad


def dk_deta(z, cosmo=Planck15):
    """
    Get gactor converting 1/GHz to h/Mpc.

    Parameters
    ----------
    z: float
        Redshift
    """
    return 2 * np.pi / dL_df(z, cosmo)


def X2Y(z, cosmo=Planck15):
    """
    Obtain the conversion factor between observing co-ordinates and cosmological volume.

    Parameters
    ----------
    z: float
        Redshift
    cosmo: astropy.cosmology.FLRW instance
        A cosmology.

    Returns
    -------
    astropy.Quantity: the conversion factor. Units are Mpc^3/h^3 / (sr GHz).
    """
    return dL_dth(z, cosmo) ** 2 * dL_df(z, cosmo)
