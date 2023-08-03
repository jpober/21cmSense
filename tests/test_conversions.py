import pytest

import numpy as np
from astropy import units
from astropy.cosmology import Planck15

from py21cmsense import conversions as cnv


def test_f2z():
    assert np.isclose(cnv.f2z(0.15 * units.GHz), cnv.f2z(150.0 * units.MHz), rtol=1e-6)
    assert np.isclose(cnv.f2z(0.142 * units.GHz), 9, atol=0.1)


def test_z2f():
    assert units.isclose(cnv.z2f(9), 142 * units.MHz, atol=1 * units.MHz)


def test_dL_dth():
    (cnv.dL_dth(10) * units.rad).to("Mpc/littleh")
    cosmo = Planck15.clone(H0=Planck15.H0 / 1.1)
    assert cnv.dL_dth(10, cosmo) < cnv.dL_dth(10)


def test_dl_df():
    (cnv.dL_df(10) * 1 * units.MHz).to("Mpc/littleh")
    cosmo = Planck15.clone(H0=Planck15.H0 / 1.1)
    assert cnv.dL_df(10, cosmo) < cnv.dL_df(10)


def test_dk_du():
    cnv.dk_du(10).to("littleh/Mpc")
    cosmo = Planck15.clone(H0=Planck15.H0 / 1.1)
    assert cnv.dk_du(10, cosmo) > cnv.dk_du(10)


def test_dk_deta():
    (cnv.dk_deta(10) * 1 / units.MHz).to("littleh/Mpc")
    cosmo = Planck15.clone(H0=Planck15.H0 / 1.1)
    assert cnv.dk_deta(10, cosmo) > cnv.dk_deta(10)


def test_X2Y():
    cnv.X2Y(10).to("Mpc^3 / (littleh^3 sr GHz)")
    cosmo = Planck15.clone(H0=Planck15.H0 / 1.1)
    assert cnv.X2Y(10, cosmo) < cnv.X2Y(10)


def test_approx_dL_dth():
    assert np.isclose(cnv.dL_dth(10), cnv.dL_dth(10.0, approximate=True), rtol=0.02)


def test_approx_dL_df():
    assert np.isclose(cnv.dL_df(10), cnv.dL_df(10.0, approximate=True), rtol=0.02)
