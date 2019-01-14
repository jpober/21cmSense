import aipy
import numpy as np


# Convert frequency (GHz) to redshift for 21cm line.
def f2z(fq):
    F21 = 1.42040575177
    return F21 / fq - 1


# Multiply by this to convert an angle on the sky to a transverse distance in
# Mpc/h at redshift z
def dL_dth(z):
    """[h^-1 Mpc]/radian, from Furlanetto et al. (2006)"""
    return 1.9 * (1.0 / aipy.const.arcmin) * ((1 + z) / 10.0) ** 0.2


# Multiply by this to convert a bandwidth in GHz to a line of sight distance in
# Mpc/h at redshift z
def dL_df(z, omega_m=0.266):
    """[h^-1 Mpc]/GHz, from Furlanetto et al. (2006)"""
    return (1.7 / 0.1) * ((1 + z) / 10.0) ** 0.5 * (omega_m / 0.15) ** -0.5 * 1e3


# Multiply by this to convert a baseline length in wavelengths (at the frequency
# corresponding to redshift z) into a tranverse k mode in h/Mpc at redshift z
def dk_du(z):
    """2pi * [h Mpc^-1] / [wavelengths], valid for u >> 1."""
    return 2 * np.pi / dL_dth(z)  # from du = 1/dth, which derives from du = d(sin(th)) using the small-angle approx


# Multiply by this to convert eta (FT of freq.; in 1/GHz) to line of sight k
# mode in h/Mpc at redshift z
def dk_deta(z):
    """2pi * [h Mpc^-1] / [GHz^-1]"""
    return 2 * np.pi / dL_df(z)


# scalar conversion between observing and cosmological coordinates
def X2Y(z):
    """[h^-3 Mpc^3] / [str * GHz]"""
    return dL_dth(z) ** 2 * dL_df(z)


# A function used for binning
def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx
