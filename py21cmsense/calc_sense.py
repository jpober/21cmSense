#! /usr/bin/env python
"""
Calculates the expected sensitivity of a 21cm experiment to a given 21cm power spectrum.  Requires as input an array .npz file created with mk_array_file.py.
"""
from __future__ import division
from __future__ import print_function

from builtins import zip

import numpy as np
from scipy import interpolate

from . import conversions as conv
import tqdm

# You can change this to have any model you want, as long as mk, mpk and p21 are returned

def get_eor_ps(k, h=0.7, power=None):
    """
    Generate a callable function for the EoR power spectrum.

    Parameters
    ----------
    k : str or array
        Either a filename pointing to a file with columns k, Delta^2(k),
        or an array of k values [units Mpc]
    h : float
        Hubble parameter.
    power : array, optional
        If k is an array, this is the Delta^2(k) corresponding to k.

    Returns
    -------
    callable: function of a single variable, k (in Mpc/h) returning the
        power (Delta^2(k)) at that k.

    """
    if isinstance(k, str):
        model = np.loadtxt(k)
        k, power = model[:, 0] / h, model[:, 1]  # k, Delta^2(k)
        # note that we're converting from Mpc to h/Mpc

    # interpolation function for the EoR model
    return interpolate.interp1d(k, power, kind="linear")


def horizon_limit(umag, z, freq, first_null=None, buffer=0.1, model='moderate'):
    # calculate horizon limit for baseline of length umag
    if model in ["moderate", "pessimistic"]:
        hor = conv.dk_deta(z) * umag / freq + buffer
    elif model in ["optimistic"]:
        if first_null is None:
            raise ValueError("If using optimistic model, require first_null")
        hor = conv.dk_deta(z) * umag / freq * np.sin(first_null / 2)

    else:
        raise ValueError("model must be either pessimistic, moderate, or optimistic")

    return hor


def calculate_sensitivity_2d(uv_coverage, freq, p21, n_per_day,
                             obs_duration, dish_size_in_lambda, t_int, Trx,
                             n_channels, bandwidth,
                             n_days=1, horizon_buffer=0.1, foreground_model='moderate', no_ns_baselines=False,
                             report=False):
    n_lstbins = n_per_day * 60.0/ obs_duration

    mink = p21.x.min()
    maxk = p21.x.max()

    if np.isscalar(dish_size_in_lambda):
        dish_size_in_lambda *= freq / 0.15

    bm = 1.13 * (2.35 * (0.45 / dish_size_in_lambda)) ** 2
    Tsky = 60e3 * (3e8 / (freq * 1e9)) ** 2.55
    Tsys = Tsky + Trx

    # for an airy disk, even though beam model is Gaussian
    first_null = 1.22 / dish_size_in_lambda

    z = conv.f2z(freq)
    kpls = conv.dk_deta(z) * np.fft.fftfreq(n_channels, bandwidth / n_channels)

    # set up blank arrays/dictionaries
    kprs = []
    # sense will include sample variance, Tsense will be Thermal only
    sense, Tsense = {}, {}

    uv_coverage *= t_int
    SIZE = uv_coverage.shape[0]

    # Cut unnecessary data out of uv coverage: auto-correlations & half of uv
    # plane (which is not statistically independent for real sky)
    uv_coverage[SIZE // 2, SIZE // 2] = 0.0
    uv_coverage[:, : SIZE // 2] = 0.0
    uv_coverage[SIZE // 2:, SIZE // 2] = 0.0
    if no_ns_baselines:
        uv_coverage[:, SIZE // 2] = 0.0

    # loop over uv_coverage to calculate k_pr
    nonzero = np.where(uv_coverage > 0)
    for iu, iv in tqdm.tqdm(zip(nonzero[1], nonzero[0]), desc="calculating 2D sensitivity", unit='uv-bins', disable=not report):
        u, v = (
            (iu - SIZE // 2) * dish_size_in_lambda,
            (iv - SIZE // 2) * dish_size_in_lambda,
        )
        umag = np.sqrt(u ** 2 + v ** 2)
        k_perp = umag * conv.dk_du(z)
        kprs.append(k_perp)

        hor = horizon_limit(
            umag=umag,
            z=z,
            freq=freq,
            first_null=first_null,
            buffer=horizon_buffer,
            model=foreground_model
        )

        if k_perp not in sense:
            sense[k_perp] = np.zeros_like(kpls)
            Tsense[k_perp] = np.zeros_like(kpls)

        for i, k_par in enumerate(kpls):
            # exclude k_parallel modes contaminated by foregrounds
            if np.abs(k_par) < hor:
                continue
            k = np.sqrt(k_par ** 2 + k_perp ** 2)
            if k < mink or k > maxk:
                continue

            tot_integration = uv_coverage[iv, iu] * n_days

            delta21 = p21(k)
            bm2 = bm / 2.0  # beam^2 term calculated for Gaussian; see Parsons et al. 2014
            bm_eff = bm ** 2 / bm2  # this can obviously be reduced; it isn't for clarity

            scalar = conv.X2Y(z) * bm_eff * bandwidth * k ** 3 / (2 * np.pi ** 2)
            Trms = Tsys / np.sqrt(2 * (bandwidth * 1e9) * tot_integration)

            # add errors in inverse quadrature
            sense[k_perp][i] += 1.0 / (scalar * Trms ** 2 + delta21) ** 2
            Tsense[k_perp][i] += 1.0 / (scalar * Trms ** 2) ** 2

    # errors were added in inverse quadrature, now need to invert and take
    # square root to have error bars; also divide errors by number of indep. fields
    for k_perp in sense.keys():
        mask = sense[k_perp] > 0
        sense[k_perp][mask] = sense[k_perp][mask] ** -0.5 / np.sqrt(n_lstbins)
        sense[k_perp][~mask] = np.inf
        Tsense[k_perp][mask] = Tsense[k_perp][mask] ** -0.5 / np.sqrt(n_lstbins)
        Tsense[k_perp][~mask] = np.inf

    return kpls, sense, Tsense


def average_sensitivity_to_1d(sense, Tsense, maxk, bandwidth, freq, kpls, report=False):
    z = conv.f2z(freq)

    # bin the result in 1D
    delta = conv.dk_deta(z) * (1.0 / bandwidth)  # default bin size is given by bandwidth
    kmag = np.arange(delta, maxk, delta)
    sense1d = np.zeros_like(kmag)
    Tsense1d = np.zeros_like(kmag)
    for ind, kpr in enumerate(tqdm.tqdm(sense.keys(), desc='averaging to 1D', unit='kpar bins', disable=not report)):
        for i, kpl in enumerate(kpls):
            k = np.sqrt(kpl ** 2 + kpr ** 2)
            if k > maxk:
                continue

            # add errors in inverse quadrature for further binning
            sense1d[conv.find_nearest(kmag, k)] += 1.0 / sense[kpr][i] ** 2
            Tsense1d[conv.find_nearest(kmag, k)] += 1.0 / Tsense[kpr][i] ** 2

    # invert errors and take square root again for final answer
    for ind, kbin in enumerate(sense1d):
        sense1d[ind] = kbin ** -0.5 if kbin else np.inf
        Tsense1d[ind] = Tsense1d[ind] ** -0.5 if Tsense1d[ind] else np.inf

    return kmag, sense1d, Tsense1d


def calculate_significance(sense1d, p21, kmag):
    """
    calculate significance with least-squares fit of amplitude

    Returns
    -------

    """
    A = p21(kmag)
    M = p21(kmag)
    wA, wM = A * (1.0 / sense1d), M * (1.0 / sense1d)
    wA, wM = np.matrix(wA).T, np.matrix(wM).T
    amp = (wA.T * wA).I * (wA.T * wM)

    # errorbars
    X = np.matrix(wA).T * np.matrix(wA)
    err = np.sqrt((1.0 / np.float(X)))
    return amp / err
