"""
Module providing the definition of an Observatory.

This replaces the original usage of an aipy.AntennaArray with something much more
simple, and suited to the needs of this particular package.
"""

from abc import ABC

import attr
import numpy as np
from astropy import constants as cnst
from astropy import units as u
from attr import validators as vld
from cached_property import cached_property
import tqdm
from collections import defaultdict
from astropy import time

from . import _utils as ut


@attr.s(frozen=True)
class PrimaryBeam(ABC):
    _dish_size = attr.ib(converter=ut.apply_or_convert_unit("m", allow_unitless=True),
                         validator=ut.positive)

    reference_freq = attr.ib(None, converter=ut.apply_or_convert_unit("MHz"),
                             validator=vld.optional(ut.positive))

    @cached_property
    def dish_size(self):
        "The dish size (in m)"
        if hasattr(self._dish_size, "unit"):
            return self._dish_size
        else:
            if self.reference_freq is not None:
                return (self._dish_size * cnst.c / self.reference_freq).to("m")
            else:
                return ut.apply_or_convert_unit("m")(self._dish_size)

    def dish_size_in_lambda(self, freq=None):
        """The dish size in units of wavelengths, for a given frequency.

        If frequency is not given, will attempt to use the Observatory's `reference_freq`
        """
        if freq is None and self.reference_freq is not None:
            freq = self.reference_freq
        elif freq is None and self.reference_freq is None:
            raise ValueError("You must supply a frequency")

        freq = ut.apply_or_convert_unit('MHz')(freq)
        return self.dish_size / (cnst.c / freq)

    @cached_property
    def dish_size_in_lambda_ref(self):
        """
        If it exists, the dish size in units of wavelengths at the reference frequency.
        """
        try:
            return self.dish_size_in_lambda()
        except ValueError:
            raise AttributeError("dish_size_in_lambda_ref only exists when reference_freq is set")

    def area(self, freq=None):
        """Beam area (sr)"""
        pass

    def width(self, freq=None):
        """Beam width (rad)"""
        pass

    def first_null(self, freq=None):
        """An approximation of the first null of the beam"""
        pass

    def sq_area(self, freq=None):
        """The area of the beam^2"""
        pass

    def b_eff(self, freq=None):
        """Effective beam area (Parsons 2014)"""
        pass

    @classmethod
    def from_uvbeam(cls):
        raise NotImplementedError("coming soon to a computer near you!")


@attr.s(frozen=True)
class GaussianBeam(PrimaryBeam):
    def area(self, freq=None):
        return 1.13 * self.fwhm(freq) ** 2

    def width(self, freq=None):
        return 0.45 / self.dish_size_in_lambda(freq)

    def fwhm(self, freq=None):
        return 2.35 * self.width(freq)

    def sq_area(self, freq=None):
        return self.area(freq) / 2

    def b_eff(self, freq=None):
        return self.area(freq)**2 / self.sq_area(freq)

    def first_null(self, freq=None):
        # for an airy disk, even though beam model is Gaussian
        return 1.22 / self.dish_size_in_lambda(freq)


@attr.s(frozen=True, kw_only=True)
class Observatory(object):
    antpos = attr.ib(converter=ut.apply_or_convert_unit('m'))
    _beam = attr.ib(GaussianBeam, validator=vld.instance_of(PrimaryBeam))
    latitude = attr.ib(0, converter=ut.apply_or_convert_unit('rad'),
                       validator=ut.between(-np.pi * u.rad / 2, np.pi * u.rad / 2))

    _dish_size = attr.ib(converter=ut.apply_or_convert_unit("m", allow_unitless=True),
                         validator=ut.positive)

    reference_freq = attr.ib(None, converter=ut.apply_or_convert_unit("MHz"),
                             validator=vld.optional(ut.positive))

    @antpos.validator
    def _antpos_validator(self, att, val):
        assert len(val.shape) == 2
        assert val.shape[-1] in [2,3]
        assert val.shape[0] > 1

    @cached_property
    def beam(self):
        return self._beam(dish_size=self._dish_size, reference_freq=self.reference_freq)

    @cached_property
    def n_antennas(self):
        return len(self.antpos)

    def new(self, **kwargs):
        """Return a clone of this instance, but change kwargs"""
        return attr.evolve(self, **kwargs)

    @classmethod
    def from_uvdata(cls, uvdata, dish_size, beam=GaussianBeam, reference_freq=None):
        """Instantiate an Observatory from a pyuvdata.UVData object or compatible file"""
        try:
            import pyuvdata
        except ImportError:
            raise ImportError("cannot construct Observatory from uvdata object without "
                              "pyuvdata being installed!")

        if isinstance(uvdata, str):
            uv = pyuvdata.UVData()
            uv.read(uvdata)
        else:
            uv = uvdata

        return cls(
            antpos=uv.antenna_positions,
            beam=beam,
            dish_size=dish_size,
            reference_freq=reference_freq,
            latitude=uv.telescope_lat_lon_alt[0]
        )

    @staticmethod
    def beamgridder(xcen, ycen, size):
        cen = size // 2 - 0.5  # correction for centering
        xcen += cen
        ycen = -1 * ycen + cen
        beam = np.zeros((size, size))

        if round(ycen) > size - 1 or round(xcen) > size - 1 or ycen < 0.0 or xcen < 0.0:
            return beam
        else:
            beam[int(round(ycen)), int(round(xcen))] = 1.0  # single pixel gridder
            return beam

    def get_baselines_metres(self):
        """
        Calculate the raw baseline distances in metres for every pair of antennas.
        """
        # this does an "outer" subtraction, leaving the inner 2- or 3- length positions
        # as atomic quantities.
        return self.antpos[np.newaxis, :, :] - self.antpos[:, np.newaxis, :]

    @cached_property
    def rotation_matrix_eq2top_zenith(self):
        # This is the rotation matrix for converting equatorial co-ordinates
        # to topocentric co-ordinates at zenith
        # pulled from aipy.coord.eq2top_m where arguments are (0, self.lat).
        return self.rotation_matrix_eq2tops(0)

    def projected_baselines(self, lst=0):
        """The *projected* baseline lengths (in metres) phased to zenith"""
        baselines_metres = self.get_baselines_metres()

        # antpos could just be x,y positions, assuming that altitude is 0
        if baselines_metres.shape[-1] == 2:
            rotation_matrix = self.rotation_matrix_eq2tops(lst)[:2, :2]
        else:
            rotation_matrix = self.rotation_matrix_eq2tops(lst)

        baselines = np.tensordot(rotation_matrix, baselines_metres, (-1, 2)).T
        return baselines

    def get_redundant_baselines(self, bl_min=0, bl_max=np.inf, ref_fq=None, ndecimals=0,
                                report=False):
        """
        Determine all baseline pairs, grouping together redundant baselines.

        Parameters
        ----------
        bl_min : float or astropy.Quantity, optional
            The minimum baseline to consider, in metres (or compatible units)
        bl_max : float or astropy.Quantity, optional
            The maximum baseline to consider, in metres (or compatible units)
        ref_fq : float or astropy.Quantity, optional
            The frequency at which to calculate the baseline lengths (in wavelength
            units). If no units given, assumed to be MHz. Default is `reference_freq`,
            if that is available.
        ndecimals : int, optional
            The number of decimals to which the UV points must be the same to be
            considered redundant.
        report : bool, optional
            Whether to report information during the calculation.

        Returns
        -------
        dict: a dictionary in which keys are 3-tuples of (u,v, |u|) co-ordinates and
            values are lists of 2-tuples, where each 2-tuple consists of the indices
            of a pair of antennas with those co-ordinates.
        """
        uvbins = defaultdict()

        if ref_fq is None and self.reference_freq is not None:
            ref_fq = self.reference_freq
        elif ref_fq is None and self.reference_freq is None:
            raise ValueError("you must pass a reference frequency")

        # Ensure ref_fq is in MHz
        ref_fq = ut.apply_or_convert_unit("MHz")(ref_fq)
        bl_min = ut.apply_or_convert_unit("m")(bl_min)
        bl_max = ut.apply_or_convert_unit("m")(bl_max)

        m2lambda = (ref_fq / cnst.c).to("1/m")
        bl_min *= m2lambda
        bl_max *= m2lambda

        longest_baseline = 0
        # find redundant baselines
        projected_baselines = self.projected_baselines()
        for i in tqdm.tqdm(range(self.n_antennas-1), desc="finding redundancies",
                           unit='ants', disable=not report):
            for j in range(i+1, self.n_antennas):

                uvw = projected_baselines[i, j] * m2lambda
                u, v = uvw[0], uvw[1] # there may or may not be a "w" term.

                bl_len = np.sqrt(u ** 2 + v ** 2)
                if bl_len < bl_min or bl_len > bl_max:
                    continue

                if bl_len > longest_baseline:
                    longest_baseline = bl_len

                uvbin = (ut.trunc(u, ndecimals=ndecimals),
                         ut.trunc(v, ndecimals=ndecimals),
                         ut.trunc(bl_len.value, ndecimals=ndecimals))

                # add the uv point and its inverse to the redundant baseline dict.
                uvbins[uvbin].append((i, j))
                uvbins[(-uvbin[0], -uvbin[1], uvbin[2])].append((j, i))

        if report:
            print("There are %i baseline types" % len(uvbins))
            print(
                "The longest baseline being included is %.2f m"
                % (longest_baseline / m2lambda)
            )

        return bl_min, bl_max, uvbins

    def lsts_from_obs_int_time(self, integration_time, observation_duration=None, freq=None):
        """
        Compute a list of LSTs from a given observation duration and integration
        time. Without loss of (significant) generality, LSTs are always centred at zero.
        Parameters
        ----------
        observation_duration : float or astropy.Quantity
            Duration of full observation (for single night). Assumed to be in minutes.
        integration_time : float or astropy.Quantity
            Time for single snapshot.

        Returns
        -------
        array: LSTs
        """
        if observation_duration is None:
            if freq is None:
                if self.reference_freq is not None:
                    freq = self.reference_freq
                else:
                    raise ValueError("A frequency must be provided")

            observation_duration = (
                    60.0 * self.beam.fwhm(freq) / (15.0 * np.pi/180.0)
            )  # minutes it takes the sky to drift through beam FWHM

        observation_duration = ut.apply_or_convert_unit("min")(observation_duration)
        integration_time = ut.apply_or_convert_unit('s')(integration_time)
        assert integration_time < observation_duration

        # convert durations to radians (i.e. LSTs)
        duration_in_radians = observation_duration.to('sday').value * 2*np.pi
        inttime_in_radians = integration_time.to("sday").value * 2 * np.pi
        assert duration_in_radians < 2*np.pi

        return np.arange(-duration_in_radians/2, duration_in_radians/2, inttime_in_radians)

    def rotation_matrix_eq2tops(self, lst):
        """Return the 3x3 matrix converting equatorial coordinates to topocentric
        at the given LST

        Copied from aipy.coord
        """
        sin_H, cos_H = np.sin(lst), np.cos(lst)
        sin_d, cos_d = np.sin(self.latitude), np.cos(self.latitude)
        zero = np.zeros_like(lst)
        return np.array([[sin_H, cos_H, zero],
                        [-sin_d * cos_H, sin_d * sin_H, cos_d],
                        [cos_d * cos_H, -cos_d * sin_H, sin_d]])

    def grid_baselines(self, bl_max, integration_time,
                    uvbins, ref_fq=None, observation_duration=None, report=False):

        if ref_fq is None and self.reference_freq is not None:
            ref_fq = self.reference_freq
        elif ref_fq is None and self.reference_freq is None:
            raise ValueError("you must pass a reference frequency")

        # Ensure ref_fq is in MHz
        ref_fq = ut.apply_or_convert_unit("MHz")(ref_fq)

        lsts = self.lsts_from_obs_int_time(integration_time, observation_duration)
        m2lambda = (ref_fq / cnst.c).to("1/m")

        dish_size = self.beam.dish_size_in_lambda(ref_fq)
        # grid each baseline type into uv plane
        # round to nearest odd
        dim = int(np.round(bl_max / dish_size) * 2 + 1)

        uvsum = np.zeros((dim, dim)),
        quadsum = np.zeros((len(lsts), dim, dim))
        for i, t in enumerate(
                tqdm.tqdm(lsts, desc="gridding baselines", unit='intg. times', disable=not report)):
            uvw = self.projected_baselines(t) * m2lambda

            for cnt, antpairs in enumerate(
                    tqdm.tqdm(uvbins.values(), desc="gridding baselines", unit='uv-bins', disable=not report)
            ):
                bl = antpairs[0]
                nbls = len(antpairs)
                i, j = bl

                _beam = self.beamgridder(
                    xcen=uvw[i,j][0] / dish_size,
                    ycen=uvw[i,j][1] / dish_size,
                    size=dim,
                )

                uvsum += nbls * _beam
                quadsum[i] += nbls * _beam

        quadsum = np.sqrt(np.sum(quadsum**2, axis=0))

        return quadsum, uvsum
