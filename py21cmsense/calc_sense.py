#! /usr/bin/env python
"""
Calculates the expected sensitivity of a 21cm experiment to a given 21cm power spectrum.  Requires as input an array .npz file created with mk_array_file.py.
"""
from __future__ import division
from __future__ import print_function

from builtins import zip

import attr
import numpy as np
import tqdm
from astropy import constants as cnst
from astropy import units
from attr import validators as vld, converters as cnv
from cached_property import cached_property
from scipy import interpolate

from . import conversions as conv
from ._utils import apply_or_convert_unit
from . import observatory as obs
from . import _utils as ut

def get_eor_ps(k, h=0.7, power=None):
    """
    Generate a callable function for the EoR power spectrum.

    You can change this to have any model you want, as long as mk, mpk and p21
    are returned


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


@attr.s(kw_only=True)
class Sensitivity:
    observatory = attr.ib(validator=vld.instance_of(obs.Observatory))
    _ubins = attr.ib()
    freq = attr.ib(converter=apply_or_convert_unit("MHz"), validator=ut.positive)
    p21 = attr.ib()
    n_per_day = attr.ib(converter=float, validator=ut.positive)
    obs_duration = attr.ib(default=None, converter=cnv.optional(apply_or_convert_unit("min")),
                           validator=vld.optional(ut.positive))
    integration_time = attr.ib(converter=apply_or_convert_unit('s'), validator=ut.positive)
    Trcv = attr.ib(convert=apply_or_convert_unit("K"), validator=ut.nonnegative)
    n_channels = attr.ib(converter=int, validator=ut.positive)
    bandwidth = attr.ib(converter=apply_or_convert_unit("MHz"), validator=ut.positive)
    n_days = attr.ib(default=1, converter=int, validator=ut.positive)
    horizon_buffer = attr.ib(default=0.1, converter=float, validator=ut.nonnegative)
    redundancy_tol = attr.ib(default=0, converter=int, validator=ut.nonnegative)
    foreground_model = attr.ib(default="moderate",
                               validator=vld.in_(['pessimistic', 'moderate', 'optimistic']))
    bl_min = attr.ib(default=0, converter=ut.apply_or_convert_unit("m"), validator=ut.nonnegative)
    bl_max = attr.ib(default=np.inf, converter=ut.apply_or_convert_unit('m'), validator=ut.nonnegative)
    no_ns_baselines = attr.ib(default=False, converter=bool)

    def get_uvbins(self, report=False):
        return self.observatory.get_redundant_baselines(
            bl_min=self.bl_min, bl_max=self.bl_max,
            ref_fq=self.freq, ndecimals=self.redundancy_tol,
            report=report)[-1]

    def get_blmax_from_uvbins(self, uvbins):
        return np.max([uv[-1] for uv in uvbins.keys()])

    @cached_property
    def uv_coverage(self):
        uvbins = self.get_uvbins()
        bl_max = self.get_blmax_from_uvbins(uvbins)

        quadsum, uvsum = self.observatory.grid_baselines(
            bl_max=bl_max, integration_time=self.integration_time,
            uvbins=uvbins, ref_fq=self.freq,
            observation_duration=self.obs_duration, report=False
        )

        SIZE = uvsum.shape[0]

        # Cut unnecessary data out of uv coverage: auto-correlations & half of uv
        # plane (which is not statistically independent for real sky)
        uvsum[SIZE // 2, SIZE // 2] = 0.0
        uvsum[:, : SIZE // 2] = 0.0
        uvsum[SIZE // 2:, SIZE // 2] = 0.0

        if self.no_ns_baselines:
            uvsum[:, SIZE // 2] = 0.0

        return uvsum * self.integration_time

    @cached_property
    def dish_size_in_lambda(self):
        return self.observatory.beam.dish_size_in_lambda(self.freq)

    @cached_property
    def n_lstbins(self):
        return self.n_per_day * 60.0*units.s / self.obs_duration

    @cached_property
    def mink(self):
        return self.p21.x.min()

    @cached_property
    def maxk(self):
        return self.p21.x.max()

    @cached_property
    def Tsky(self):
        return 60e3 * (cnst.c / self.freq / units.m) ** 2.55

    @cached_property
    def Tsys(self):
        return self.Tsky + self.Trcv

    @cached_property
    def redshift(self):
        return conv.f2z(self.freq)

    @cached_property
    def kparallel(self):
        return conv.dk_deta(self.redshift) * np.fft.fftfreq(self.n_channels, self.bandwidth / self.n_channels)

    @cached_property
    def tot_integration(self):
        return self.uv_coverage * self.n_days

    @cached_property
    def Trms(self):
        return self.Tsys / np.sqrt(2 * (self.bandwidth * 1e9) * self.tot_integration)

    @cached_property
    def X2Y(self):
        return conv.X2Y(self.redshift)

    def power_normalisation(self, k):
        return self.X2Y * self.beam_eff * self.bandwidth * k ** 3 / (2 * np.pi ** 2)

    @cached_property
    def ubins(self):
        size = self.uv_coverage.shape[0]
        return (np.arange(size) - size // 2) * self.dish_size_in_lambda

    @cached_property
    def k1d(self):
        delta = conv.dk_deta(self.redshift) * (1.0 / self.bandwidth)  # default bin size is given by bandwidth
        return np.arange(delta, self.maxk, delta)

    def thermal_noise(self, k_par, k_perp):
        k = np.sqrt(k_par ** 2 + k_perp ** 2)

        scalar = self.power_normalisation(k)

        # add errors in inverse quadrature
        return scalar * self.Trms ** 2

    def sample_noise(self, k_par, k_perp):
        k = np.sqrt(k_par ** 2 + k_perp ** 2)
        if k < self.mink or k > self.maxk:
            return np.inf

        return self.p21(k)

    def calculate_sensitivity_2d(self, report=False, sources=['thermal', 'sample']):

        # set up blank arrays/dictionaries
        kprs = []
        sense = {}

        # loop over uv_coverage to calculate k_pr
        nonzero = np.where(self.uv_coverage > 0)
        for iu, iv in tqdm.tqdm(zip(nonzero[1], nonzero[0]), desc="calculating 2D sensitivity", unit='uv-bins',
                                disable=not report):
            u, v = self.ubins[iu], self.ubins[iv]

            umag = np.sqrt(u ** 2 + v ** 2)
            k_perp = umag * conv.dk_du(self.redshift)
            kprs.append(k_perp)

            hor = self.horizon_limit(umag)

            if k_perp not in sense:
                sense[k_perp] = np.zeros_like(self.kparallel)

            for i, k_par in enumerate(self.kparallel):
                # exclude k_parallel modes contaminated by foregrounds
                if np.abs(k_par) < hor:
                    continue

                val = 0
                for source in sources:
                    if len(source) == 2 and hasattr(source[0], "__len__"):
                        val += source[1](self, k_par, k_perp)
                    else:
                        val += getattr(self, source + "_noise")(k_par, k_perp)

                    sense[k_perp][i] += 1.0 / val ** 2

        # errors were added in inverse quadrature, now need to invert and take
        # square root to have error bars; also divide errors by number of indep. fields
        for k_perp in sense.keys():
            mask = sense[k_perp] > 0
            sense[k_perp][mask] = sense[k_perp][mask] ** -0.5 / np.sqrt(self.n_lstbins)
            sense[k_perp][~mask] = np.inf

        return sense

    def horizon_limit(self, umag):
        # calculate horizon limit for baseline of length umag
        if self.foreground_model in ["moderate", "pessimistic"]:
            return conv.dk_deta(self.redshift) * umag / self.freq + self.horizon_buffer
        elif self.foreground_model in ["optimistic"]:
            return conv.dk_deta(self.redshift) * umag / self.freq * np.sin(self.first_null / 2)

    def _average_sense_to_1d(self, sense, report=False):

        # bin the result in 1D

        sense1d = np.zeros_like(self.k1d)

        for ind, kpr in enumerate(
                tqdm.tqdm(sense.keys(), desc='averaging to 1D', unit='kpar bins', disable=not report)):
            for i, kpl in enumerate(self.kparallel):
                k = np.sqrt(kpl ** 2 + kpr ** 2)
                if k > self.maxk:
                    continue

                # add errors in inverse quadrature for further binning
                sense1d[conv.find_nearest(self.k1d, k)] += 1.0 / sense[kpr][i] ** 2

        # invert errors and take square root again for final answer
        for ind, kbin in enumerate(sense1d):
            sense1d[ind] = kbin ** -0.5 if kbin else np.inf

        return sense1d

    def calculate_sensitivity_1d(self, sense=None, report=None, sources=None):
        if sense is None:
            sense = self.calculate_sensitivity_2d(report=None, sources=sources)

        return self._average_sense_to_1d(sense, report=report)

    def calculate_significance(self, sense1d=None, report=True, sources=['thermal', 'sample']):
        """
        calculate significance with least-squares fit of amplitude

        Returns
        -------

        """
        if self.p21 is None:
            raise NotImplementedError("significance is not possible without an input 21cm power spectrum")

        if sense1d is None:
            sense1d = self.calculate_sensitivity_2d(report=report, sources=sources)

        A = self.p21(self.k1d)

        wA = A * (1.0 / sense1d), M * (1.0 / sense1d)
        wA, wM = np.matrix(wA).T, np.matrix(wM).T
        amp = (wA.T * wA).I * (wA.T * wM)

        # errorbars
        X = np.matrix(wA).T * np.matrix(wA)
        err = np.sqrt((1.0 / np.float(X)))
        return amp / err
