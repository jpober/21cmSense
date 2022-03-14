"""
Module providing the definition of an Observatory.

This replaces the original usage of an aipy.AntennaArray with something much more
simple, and suited to the needs of this particular package.
"""
from __future__ import annotations

import attr
import collections
import logging
import numpy as np
import tqdm
from astropy import constants as cnst
from astropy import units as un
from astropy.io.misc import yaml
from attr import validators as vld
from cached_property import cached_property
from collections import defaultdict
from pathlib import Path

from . import _utils as ut
from . import beam, config
from . import types as tp

logger = logging.getLogger(__name__)


@attr.s(frozen=True, kw_only=True, order=False)
class Observatory:
    """
    A class defining an interferometric Observatory and its properties.

    Parameters
    ----------
    antpos : array
        An array with shape (Nants, 3) specifying the positions of the antennas.
        These should be in the ENU (East-North-Up) frame, relative to a central location
        given by `latitude`. If not a Quantity, units are assumed to be meters.
    beam : :class:`~py21cmsense.beam.PrimaryBeam` instance
        A beam, assumed to be homogeneous across antennas.
    latitude : float or Quantity, optional
        Latitude of the array center. If a float, assumed to be in radians.
        Note that longitude is not required, as we assume an isotropic sky.
    Trcv : float or Quantity
        Receiver temperature, assumed to be in mK unless otherwise defined.
    min_antpos, max_antpos
        The minimum/maximum radial distance to include antennas (from the origin
        of the array). Assumed to be in units of meters if no units are supplied.
        Can be used to limit antennas in arrays like HERA and SKA that
        have a "core" and "outriggers". The minimum is inclusive, and maximum exclusive.
    """

    _antpos: tp.Length = attr.ib(eq=attr.cmp_using(eq=np.array_equal))
    beam: beam.PrimaryBeam = attr.ib(validator=vld.instance_of(beam.PrimaryBeam))
    latitude: un.rad = attr.ib(
        0 * un.rad,
        validator=ut.between(-np.pi * un.rad / 2, np.pi * un.rad / 2),
    )
    Trcv: tp.Temperature = attr.ib(100 * un.K, validator=ut.nonnegative)
    max_antpos: tp.Length = attr.ib(
        default=np.inf * un.m, validator=(tp.vld_physical_type("length"), ut.positive)
    )
    min_antpos: tp.Length = attr.ib(
        default=0.0 * un.m, validator=(tp.vld_physical_type("length"), ut.nonnegative)
    )

    @_antpos.validator
    def _antpos_validator(self, att, val):
        tp.vld_physical_type("length")(self, att, val)
        assert val.ndim == 2
        assert val.shape[-1] == 3
        assert val.shape[0] > 1

    @cached_property
    def antpos(self) -> np.ndarray:
        """The positions of antennas in the array in units of metres."""
        # Mask out some antennas if a max_antpos is set in the YAML
        _n = len(self._antpos)
        sq_len = np.sum(np.square(self._antpos), axis=1)
        antpos = self._antpos[
            np.logical_and(
                sq_len >= self.min_antpos**2,
                sq_len < self.max_antpos**2,
            )
        ]

        if self.max_antpos < np.inf or self.min_antpos > 0:
            logger.info(
                f"Removed {_n - len(antpos)} antennas using given "
                f"max_antpos={self.max_antpos} m and min_antpos={self.min_antpos} m."
            )

        return antpos

    @property
    def frequency(self) -> un.Quantity[un.MHz]:
        """Central frequency of the observation."""
        return self.beam.frequency.to("MHz")

    @cached_property
    def n_antennas(self) -> int:
        """Number of antennas in the array."""
        return len(self.antpos)

    def clone(self, **kwargs) -> Observatory:
        """Return a clone of this instance, but change kwargs."""
        return attr.evolve(self, **kwargs)

    @classmethod
    def from_uvdata(cls, uvdata, beam: beam.PrimaryBeam, **kwargs) -> Observatory:
        """Instantiate an Observatory from a :class:`pyuvdata.UVData` object."""
        return cls(
            antpos=uvdata.antenna_positions,
            beam=beam,
            latitude=uvdata.telescope_location_lat_lon_alt[0],
            **kwargs,
        )

    @classmethod
    def from_yaml(cls, yaml_file: str | dict) -> Observatory:
        """Instantiate an Observatory from a compatible YAML config file."""
        if isinstance(yaml_file, (str, Path)):
            with open(yaml_file) as fl:
                data = yaml.load(fl)
        elif isinstance(yaml_file, collections.abc.Mapping):
            data = yaml_file
        else:
            raise ValueError(
                "yaml_file must be a string filepath or a raw dict from such a file."
            )

        # Mask out some antennas if a max_antpos is set in the YAML
        max_antpos = data.pop("max_antpos", np.inf * un.m)
        antpos = data.pop("antpos")
        _n = len(antpos)

        antpos = antpos[np.sum(np.square(antpos), axis=1) < max_antpos**2]

        if max_antpos < np.inf * un.m:
            logger.info(
                f"Removed {_n - len(antpos)} antennas using given max_antpos={max_antpos} m."
            )

        # If we get only East and North coords, add zeros for the UP direction.
        if antpos.shape[1] == 2:
            antpos = np.hstack((antpos, np.zeros((len(antpos), 1))))

        _beam = data.pop("beam")
        kind = _beam.pop("class")
        _beam = getattr(beam, kind)(**_beam)

        return cls(antpos=antpos, beam=_beam, **data)

    @cached_property
    def baselines_metres(self) -> tp.Meters:
        """Raw baseline distances in metres for every pair of antennas.

        Shape is ``(Nant, Nant, 3)``.
        """
        # this does an "outer" subtraction, leaving the inner 2- or 3- length positions
        # as atomic quantities.
        return (self.antpos[np.newaxis, :, :] - self.antpos[:, np.newaxis, :]).to(un.m)

    def projected_baselines(
        self, baselines: tp.Length | None = None, time_offset: tp.Time = 0 * un.hour
    ) -> np.ndarray:
        """The *projected* baseline lengths (in wavelengths).

        Phased to a point that has rotated off zenith by some time_offset.

        Parameters
        ----------
        baselines
            The baseline co-ordinates to project, assumed to be in metres.
            If not provided, uses all baselines of the observatory.
            Shape of the array can be (N,N,3) or (N, 3).
            The co-ordinates are expected to be in ENU.
        time_offset
            The amount of time elapsed since the phase center was at zenith.
            Assumed to be in days unless otherwise defined. May be negative.

        Returns
        -------
        An array the same shape as :attr:`baselines_metres`, but phased to the
        new phase centre.
        """
        if baselines is None:
            baselines = self.baselines_metres

        orig_shape = baselines.shape

        bl_wavelengths = baselines.reshape((-1, 3)) * self.metres_to_wavelengths

        out = ut.phase_past_zenith(time_offset, bl_wavelengths, self.latitude)

        out = out.reshape(*orig_shape[:-1], np.size(time_offset), orig_shape[-1])
        if np.size(time_offset) == 1:
            out = out.squeeze(-2)

        return out

    @cached_property
    def metres_to_wavelengths(self) -> un.Quantity[1 / un.m]:
        """Conversion factor for metres to wavelengths at fiducial frequency."""
        return (self.frequency / cnst.c).to("1/m")

    @cached_property
    def baseline_lengths(self) -> np.ndarray:
        """Lengths of baselines in units of wavelengths, shape (Nant, Nant)."""
        return np.sqrt(np.sum(self.projected_baselines() ** 2, axis=-1))

    @cached_property
    def shortest_baseline(self) -> float:
        """Shortest baseline in units of wavelengths."""
        return np.min(self.baseline_lengths[self.baseline_lengths > 0])

    @cached_property
    def longest_baseline(self) -> float:
        """Longest baseline in units of wavelengths."""
        return np.max(self.baseline_lengths)

    @cached_property
    def observation_duration(self) -> un.Quantity[un.day]:
        """The time it takes for the sky to drift through the FWHM."""
        return un.day * self.beam.fwhm / (2 * np.pi * un.rad)

    def get_redundant_baselines(
        self,
        bl_min: tp.Length = 0 * un.m,
        bl_max: tp.Length = np.inf * un.m,
        ndecimals: int = 1,
    ) -> dict[tuple[float, float, float], list[tuple[int, int]]]:
        """
        Determine all baseline groups.

        Parameters
        ----------
        bl_min : float or astropy.Quantity, optional
            The minimum baseline to consider, in metres (or compatible units)
        bl_max : float or astropy.Quantity, optional
            The maximum baseline to consider, in metres (or compatible units)
        ndecimals : int, optional
            The number of decimals to which the UV points must be the same to be
            considered redundant.

        Returns
        -------
        dict: a dictionary in which keys are 3-tuples of ``(u,v, |u|)`` co-ordinates and
            values are lists of 2-tuples, where each 2-tuple consists of the indices
            of a pair of antennas with those co-ordinates.
        """
        uvbins = defaultdict(list)

        bl_min = bl_min.to("m") * self.metres_to_wavelengths
        bl_max = bl_max.to("m") * self.metres_to_wavelengths

        uvw = self.projected_baselines()
        # group redundant baselines
        for i in tqdm.tqdm(
            range(self.n_antennas - 1),
            desc="finding redundancies",
            unit="ants",
            disable=not config.PROGRESS,
        ):
            for j in range(i + 1, self.n_antennas):

                bl_len = self.baseline_lengths[i, j]  # in wavelengths
                if bl_len < bl_min or bl_len > bl_max:
                    continue

                u, v = uvw[i, j][:2]

                uvbin = (
                    ut.trunc(u, ndecimals=ndecimals),
                    ut.trunc(v, ndecimals=ndecimals),
                    ut.trunc(bl_len, ndecimals=ndecimals),
                )

                # add the uv point and its inverse to the redundant baseline dict.
                uvbins[uvbin].append((i, j))
                uvbins[(-uvbin[0], -uvbin[1], uvbin[2])].append((j, i))

        return uvbins

    def time_offsets_from_obs_int_time(
        self, integration_time: tp.Time, observation_duration: tp.Time | None = None
    ):
        """Compute a list of time offsets within an LST-bin.

        The LSTs 'within a bin' are added coherently for a given baseline group.
        Time offsets are with respect to an arbitrary time, and describe the rotation of
        a hypothetical point through zenith.

        Parameters
        ----------
        integration_time
            Time for single snapshot.
        observation_duration
            Duration of the LST bin (for single night).

        Returns
        -------
        array :
            Time offsets (in julian days).
        """
        if observation_duration is None:
            observation_duration = self.observation_duration

        assert integration_time <= observation_duration

        return (
            np.arange(
                -observation_duration.to("day").value / 2,
                observation_duration.to("day").value / 2,
                integration_time.to("day").value,
            )
            << un.day
        )

    def baseline_coords_from_groups(self, baseline_groups) -> un.Quantity[un.m]:
        """Convert a dictionary of baseline groups to an array of ENU co-ordinates."""
        out = np.zeros((len(baseline_groups), 3)) * un.m
        for i, antpairs in enumerate(baseline_groups.values()):
            out[i] = self.baselines_metres[antpairs[0][0], antpairs[0][1]]
        return out

    @staticmethod
    def baseline_weights_from_groups(baseline_groups) -> np.ndarray:
        """Get number of baselines in each group.

        Parameters
        ----------
        baseline_groups
            A dictionary in the format output by :func:`get_redundant_baselines`.

        Returns
        -------
        weights
            An array containing the number of baselines in each group.
        """
        return np.array([len(antpairs) for antpairs in baseline_groups.values()])

    def grid_baselines(
        self,
        baselines: tp.Length | None = None,
        weights: np.ndarray | None = None,
        integration_time: tp.Time = 60.0 * un.s,
        bl_min: tp.Length = 0 * un.m,
        bl_max: tp.Length = np.inf * un.m,
        observation_duration: tp.Time | None = None,
        ndecimals: int = 1,
    ) -> np.ndarray:
        """
        Grid baselines onto a pre-determined uvgrid, accounting for earth rotation.

        Parameters
        ----------
        baselines : array_like, optional
            The baseline co-ordinates to project, assumed to be in metres.
            If not provided, calculates effective baselines by finding redundancies on
            all baselines in the observatory. Shape of the array can be (N,N,3) or (N, 3).
            The co-ordinates are expected to be in ENU. If `baselines` is provided,
            `weights` must also be provided.
        weights: array_like, optional
            An array of the same length as `baselines`, giving the number of independent
            baselines at each co-ordinate. If not provided, calculates effective
            baselines by finding redundancies on all baselines in the observatory.
            If `baselines` is provided, `weights` must also be provided.
        integration_time : float or Quantity, optional
            The amount of time integrated into a snapshot visibility, assumed
            to be in seconds.
        bl_min : float or Quantity, optional
            Minimum baseline length (in meters) to include in the gridding.
        bl_max : float or Quantity, optional
            Maximum baseline length (in meters) to include in the gridding.
        observation_duration : float or Quantity, optional
            Amount of time in a single (coherent) LST bin, assumed to be in minutes.
        ndecimals : int, optional
            Number of decimals to which baselines must match to be considered redundant.

        Returns
        -------
        array :
            Shape [n_baseline_groups, Nuv, Nuv]. The coherent sum of baselines within
            grid cells given by :attr:`ugrid`. One can treat different baseline groups
            independently, or sum over them.

        See Also
        --------
        grid_baselines_coherent :
            Coherent sum over baseline groups of the output of this method.
        grid_basleine_incoherent :
            Incoherent sum over baseline groups of the output of this method.
        """
        if baselines is not None:
            assert un.get_physical_type(baselines) == "length"
            assert baselines.ndim in (2, 3)

        assert un.get_physical_type(integration_time) == "time"
        assert un.get_physical_type(bl_min) == "length"
        assert un.get_physical_type(bl_max) == "length"
        if observation_duration is not None:
            assert un.get_physical_type(observation_duration) == "time"

        if baselines is None:
            baseline_groups = self.get_redundant_baselines(
                bl_min=bl_min, bl_max=bl_max, ndecimals=ndecimals
            )
            baselines = self.baseline_coords_from_groups(baseline_groups)
            weights = self.baseline_weights_from_groups(baseline_groups)

        if weights is None:
            raise ValueError(
                "If baselines are provided, weights must also be provided."
            )

        time_offsets = self.time_offsets_from_obs_int_time(
            integration_time, observation_duration
        )

        uvws = self.projected_baselines(baselines, time_offsets).reshape(
            baselines.shape[0], time_offsets.size, 3
        )

        # grid each baseline type into uv plane
        dim = len(self.ugrid(bl_max))
        edges = self.ugrid_edges(bl_max)

        uvsum = np.zeros((len(baselines), dim, dim))
        for cnt, (uvw, nbls) in enumerate(
            tqdm.tqdm(
                zip(uvws, weights),
                desc="gridding baselines",
                unit="baselines",
                disable=not config.PROGRESS,
                total=len(weights),
            )
        ):
            uvsum[cnt] = np.histogram2d(uvw[:, 0], uvw[:, 1], bins=edges)[0] * nbls

        return uvsum

    def longest_used_baseline(
        self, bl_max: tp.Length = np.inf * un.m
    ) -> un.Quantity[un.m]:
        """Determine the maximum baseline length kept in the array."""
        if np.isinf(bl_max):
            return self.longest_baseline

        # Note we don't do the conversion in-place!
        bl_max = bl_max * self.metres_to_wavelengths
        return np.max(self.baseline_lengths[self.baseline_lengths <= bl_max])

    def ugrid_edges(self, bl_max: tp.Length = np.inf * un.m) -> np.ndarray:
        """Get a uv grid out to the maximum used baseline smaller than given bl_max.

        The resulting array represents the *edges* of the grid (so the number of cells
        is one fewer than this).

        Parameters
        ----------
        bl_max : float or Quantity
            Include all baselines smaller than this number. Units of m.

        Returns
        -------
        array :
            1D array of regularly spaced un.
        """
        bl_max = self.longest_used_baseline(bl_max)

        # We're doing edges of bins here, and the first edge is at uv_res/2
        n_positive = int(
            np.ceil((bl_max - self.beam.uv_resolution / 2) / self.beam.uv_resolution)
        )

        # Grid from uv_res/2 to just past (or equal to) bl_max, in steps of resolution.
        positive = np.linspace(
            self.beam.uv_resolution / 2,
            self.beam.uv_resolution / 2 + n_positive * self.beam.uv_resolution,
            n_positive + 1,
        )
        return np.concatenate((-positive[::-1], positive))

    def ugrid(self, bl_max: tp.Length = np.inf * un.m) -> np.ndarray:
        """Centres of the UV grid plane."""
        # Shift the edges by half a cell, and omit the last one
        edges = self.ugrid_edges(bl_max)
        return (edges[1:] + edges[:-1]) / 2

    def grid_baselines_coherent(self, **kwargs) -> np.ndarray:
        """Get a UV grid of coherently gridded baselines.

        Different baseline groups are averaged coherently if they fall into the same
        UV bin.

        See :func:`grid_baselines` for parameter details.
        """
        grid = self.grid_baselines(**kwargs)
        return np.sum(grid, axis=0)

    def grid_baselines_incoherent(self, **kwargs) -> np.ndarray:
        """Get a UV grid of incoherently gridded baselines.

        Different baseline groups are averaged incoherently if they fall into the same
        UV bin.

        See :func:`grid_baselines` for parameter details.
        """
        grid = self.grid_baselines(**kwargs)
        return np.sqrt(np.sum(grid**2, axis=0))
