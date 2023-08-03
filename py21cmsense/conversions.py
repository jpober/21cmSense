"""
Common 21 cm conversions.

Provides conversions between observing co-ordinates and cosmological co-ordinates.
"""
import numpy as np
from astropy import constants as cnst
from astropy import units as un
from astropy.cosmology import FLRW, Planck15
from astropy.cosmology.units import littleh
from typing import Union

from . import types as tp

# The frequency of the 21cm line emission.
f21 = 1.42040575177 * un.GHz


@un.quantity_input
def f2z(fq: tp.Frequency) -> float:
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
    return float(f21 / fq - 1)


@un.quantity_input
def z2f(z: Union[float, np.array]) -> un.Quantity[un.GHz]:
    """
    Convert redshift to z=0 frequency for 21 cm line.

    Parameters
    ----------
    z
        Redshift

    Returns
    -------
    astropy.Quantity : the frequency
    """
    return f21 / (1 + z)


def dL_dth(
    z: Union[float, np.array],
    cosmo: FLRW = Planck15,
    approximate=False,
) -> un.Quantity[un.Mpc / un.rad / littleh]:
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
    if approximate:
        return (
            (1.9 * (1.0 / un.arcmin) * ((1 + z) / 10.0) ** 0.2).to(1 / un.rad)
            * un.Mpc
            / littleh
        )
    else:
        return cosmo.h * cosmo.comoving_transverse_distance(z) / un.rad / littleh


def dL_df(
    z: Union[float, np.array],
    cosmo: FLRW = Planck15,
    approximate=False,
) -> un.Quantity[un.Mpc / un.MHz / littleh]:
    """
    Get the factor to convert bandwidth to line-of-sight distance in Mpc/h.

    Parameters
    ----------
    z : float
        The redshift
    """
    if approximate:
        return (
            (1.7 / 0.1)
            * ((1 + z) / 10.0) ** 0.5
            * (cosmo.Om0 / 0.15) ** -0.5
            * un.Mpc
            / littleh
            / un.MHz
        )
    else:
        return (cosmo.h * cnst.c * (1 + z) / (z2f(z) * cosmo.H(z) * littleh)).to(
            "Mpc/(MHz*littleh)"
        )


def dk_du(
    z: Union[float, np.array],
    cosmo: FLRW = Planck15,
    approximate=False,
) -> un.Quantity[littleh / un.Mpc]:
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
    return 2 * np.pi / dL_dth(z, cosmo, approximate=approximate) / un.rad


def dk_deta(
    z: Union[float, np.array],
    cosmo: FLRW = Planck15,
    approximate=False,
) -> un.Quantity[un.MHz * littleh / un.Mpc]:
    """
    Get gactor converting inverse frequency to inverse distance.

    Parameters
    ----------
    z: float
        Redshift
    """
    return 2 * np.pi / dL_df(z, cosmo, approximate=approximate)


def X2Y(
    z: Union[float, np.array],
    cosmo: FLRW = Planck15,
    approximate=False,
) -> un.Quantity[un.Mpc**3 / littleh**3 / un.steradian / un.MHz]:
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
    astropy.Quantity: the conversion factor. Units are Mpc^3/h^3 / (sr MHz).
    """
    return dL_dth(z, cosmo, approximate=approximate) ** 2 * dL_df(
        z, cosmo, approximate=approximate
    )
