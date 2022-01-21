"""
Common 21 cm conversions.

Provides conversions between observing co-ordinates and cosmological co-ordinates.
"""
# from __future__ import annotations
import numpy as np
from astropy import constants as cnst
from astropy import units as u
from astropy.cosmology import Planck15, FLRW
from astropy.cosmology.units import littleh
from typing import Union

from . import types as tp

# The frequency of the 21cm line emission.
f21 = 1.42040575177 * u.GHz

@u.quantity_input
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

@u.quantity_input
def z2f(z: Union[float, np.array]) -> u.Quantity[u.GHz]:
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


def dL_dth(z: Union[float, np.array], cosmo: FLRW=Planck15) -> u.Quantity[u.Mpc / u.rad / littleh]:
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
    return cosmo.h * cosmo.comoving_transverse_distance(z) / u.rad / littleh


def dL_df(z: Union[float, np.array], cosmo: FLRW=Planck15) -> u.Quantity[u.Mpc / u.MHz / littleh]:
    """
    Get the factor to convert bandwidth to line-of-sight distance in Mpc/h.

    Parameters
    ----------
    z : float
        The redshift
    """
    return (cosmo.h * cnst.c * (1 + z) ** 2 / (z2f(z) * cosmo.H(z) * littleh)).to(
        "Mpc/(MHz*littleh)"
    )


def dk_du(z: Union[float, np.array], cosmo: FLRW=Planck15) -> u.Quantity[littleh / u.Mpc]:
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


def dk_deta(z: Union[float, np.array], cosmo: FLRW=Planck15) -> u.Quantity[u.MHz * littleh / u.Mpc]:
    """
    Get gactor converting inverse frequency to inverse distance.

    Parameters
    ----------
    z: float
        Redshift
    """
    return 2 * np.pi / dL_df(z, cosmo)


def X2Y(z: Union[float, np.array], cosmo: FLRW=Planck15) -> u.Quantity[u.Mpc**3 / littleh**3 / u.steradian / u.GHz]:
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
