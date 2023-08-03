"""A module defining interferometric observation objects."""
from __future__ import annotations

import attr
import collections
import numpy as np
from astropy import units as un
from astropy.cosmology import LambdaCDM, Planck15
from astropy.io.misc import yaml
from attr import validators as vld
from collections import defaultdict
from functools import cached_property
from hickleable import hickleable
from os import path
from typing import Any

from . import _utils as ut
from . import conversions as conv
from . import observatory as obs
from . import types as tp


@hickleable(evaluate_cached_properties=True)
@attr.s(kw_only=True)
class Observation:
    """
    A class defining an interferometric Observation.

    Note that to achieve the same behaviour as the original 21cmSense's ``--track``
    option (i.e. track-scan instead of drift-scan) merely requires setting ``obs_duration``
    to the length of the tracked scan.

    Parameters
    ----------
    observatory : :class:`~py21cmsense.observatory.Observatory`
        An object defining attributes of the observatory itself (its location etc.)
    hours_per_day : float or Quantity, optional
        The number of good observing hours per day.  This corresponds to the size of a
        low-foreground region in right ascension for a drift scanning instrument. The
        total observing time is `n_days*hours_per_day`.  Default is 6.
        If simulating a tracked scan, `hours_per_day` should be a multiple of the length
        of the track (i.e. for two three-hour tracks per day, `hours_per_day` should be 6).
    track
        Tracked scan observation duration. This is an alias for ``obs_duration``, and
        is provided only for ease of use for those familiar with 21cmSense v1.
    obs_duration : float or Quantity, optional
        The time assigned to a single LST bin, by default the time it takes for a source
        to travel through the beam's FWHM. If a float, assumed to be in minutes.
    integration_time : float or Quantity, optional
        The amount of time integrated into a single visibility, by default a minute.
        If a float, assumed to be in seconds.
    n_channels : int, optional
        Number of channels used in the observation. Defaults to 82, which is equivalent
        to 1024 channels over 100 MHz of bandwidth. Sets maximum k_parallel that can be
        probed, but little to no overall effect on sensitivity.
    bandwidth : float or Quantity, optional
        The bandwidth used for the observation, assumed to be in MHz. Note this is not the total
        instrument bandwidth, but the redshift range that can be considered co-eval.
    n_days : int, optional
        The number of days observed (for the same set of LSTs). The default is 180, which is the
        maximum a particular R.A. can be observed in one year if one only observes at night.
        The total observing time is `n_days*hours_per_day`.
    bl_min : float, optional
        Set the minimum baseline (in meters) to include in the uv plane.
    bl_max : float, optional
        Set the maximum baseline (in meters) to include in the uv plane.
    redundancy_tol : int, optional
        The number of decimal places to which baseline vectors must match (in
        all dimensions) to be considered redundant.
    coherent : bool, optional
        Whether to add different baselines coherently if they are not instantaneously redundant.
    spectral_index : float, optional
        The spectral index of the foreground model. The foreground model is approximated as
        a spatially-independent power-law, and used only for generating sky noise temperature.
        The default is 2.6, based on Mozdzen et al. 2017: 2017MNRAS.464.4995M, figure 8,
        with Galaxy down (see also `tsky_amplitude` and `tsky_ref_freq`).
    tsky_amplitude : float or Quantity, optional
        The temperature of foregrounds at `tsky_ref_freq`. See `spectral_index`.
        Default assumed to be in mK.
    tsky_ref_freq : float or Quantity
        Frequency at which the foreground model is equal to `tsky_amplitude`.
        See `spectral_index`. Default assumed to be in MHz.
    use_approximate_cosmo : bool
        Whether to use approximate cosmological conversion factors. Doing so will give
        the same results as the original 21cmSense code, but non-approximate versions
        that use astropy are preferred.
    cosmo : LambdaCDM
        An astropy cosmology object to use.
    """

    observatory: obs.Observatory = attr.ib(validator=vld.instance_of(obs.Observatory))

    time_per_day: tp.Time = attr.ib(
        6 * un.hour,
        validator=(tp.vld_physical_type("time"), ut.between(0 * un.hour, 24 * un.hour)),
    )
    track: tp.Time | None = attr.ib(
        None,
        validator=attr.validators.optional(
            [tp.vld_physical_type("time"), ut.between(0, 24 * un.hour)]
        ),
    )
    obs_duration: tp.Time = attr.ib(
        validator=(tp.vld_physical_type("time"), ut.between(0, 24 * un.hour)),
    )
    integration_time: tp.Time = attr.ib(
        60 * un.second, validator=(tp.vld_physical_type("time"), ut.positive)
    )
    n_channels: int = attr.ib(82, converter=int, validator=ut.positive)
    bandwidth: tp.Frequency = attr.ib(
        8 * un.MHz, validator=(tp.vld_physical_type("frequency"), ut.positive)
    )
    n_days: int = attr.ib(default=180, converter=int, validator=ut.positive)
    bl_min: tp.Length = attr.ib(
        default=0 * un.m, validator=(tp.vld_physical_type("length"), ut.nonnegative)
    )
    bl_max: tp.Length = attr.ib(
        default=np.inf * un.m,
        validator=(tp.vld_physical_type("length"), ut.nonnegative),
    )
    redundancy_tol: int = attr.ib(default=1, converter=int, validator=ut.nonnegative)
    coherent: bool = attr.ib(default=True, converter=bool)

    # The following defaults are based on Mozdzen et al. 2017: 2017MNRAS.464.4995M,
    # figure 8, with galaxy down.
    spectral_index: float = attr.ib(
        default=2.6, converter=float, validator=ut.between(1.5, 4)
    )
    tsky_amplitude: tp.Temperature = attr.ib(
        default=260000 * un.mK,
        validator=ut.nonnegative,
    )
    tsky_ref_freq: tp.Frequency = attr.ib(default=150 * un.MHz, validator=ut.positive)
    use_approximate_cosmo: bool = attr.ib(default=False, converter=bool)
    cosmo: LambdaCDM = attr.ib(default=Planck15, converter=Planck15.from_format)

    @classmethod
    def from_yaml(cls, yaml_file):
        """Construct an :class:`Observation` from a YAML file."""
        if isinstance(yaml_file, str):
            with open(yaml_file) as fl:
                data = yaml.load(fl)
        elif isinstance(yaml_file, collections.abc.Mapping):
            data = yaml_file
        else:
            raise ValueError(
                "yaml_file must be a string filepath or a raw dict from such a file."
            )

        if (
            isinstance(data["observatory"], str)
            and isinstance(yaml_file, str)
            and not path.isabs(data["observatory"])
        ):
            data["observatory"] = path.join(
                path.dirname(yaml_file), data["observatory"]
            )

        observatory = obs.Observatory.from_yaml(data.pop("observatory"))
        return cls(observatory=observatory, **data)

    def __gethstate__(self) -> dict[str, Any]:
        """Get the hickle state."""
        d = attr.asdict(self, recurse=False)
        d["cosmo"] = d["cosmo"].to_format("mapping")
        del d["cosmo"]["cosmology"]  # The class.
        return d

    def __sethstate__(self, d: dict[str, Any]) -> None:
        """Set the hickle state."""
        d["cosmo"]["cosmology"] = type(Planck15)
        d["cosmo"] = Planck15.from_format(d["cosmo"])
        self.__dict__.update(d)

    @obs_duration.validator
    def _obs_duration_vld(self, att, val):
        if val > self.time_per_day:
            raise ValueError("obs_duration must be <= time_per_day")

    @integration_time.validator
    def _integration_time_vld(self, att, val):
        if val > self.obs_duration:
            raise ValueError("integration_time must be <= obs_duration")

    @obs_duration.default
    def _obstime_default(self):
        # time it takes the sky to drift through beam FWHM
        if self.track is not None:
            return self.track
        else:
            return self.observatory.observation_duration

    @bl_max.validator
    def _bl_max_vld(self, att, val):
        if val <= self.bl_min:
            raise ValueError(
                "bl_max must be greater than bl_min, got "
                f"bl_min={self.bl_min} and bl_max={val}"
            )

    @cached_property
    def baseline_groups(
        self,
    ) -> dict[tuple(float, float, float), list[tuple(int, int)]]:
        """A dictionary of redundant baseline groups.

        Keys are tuples of floats (X,Y,LENGTH), and
        values are lists of two-tuples of baseline antenna indices in that particular
        baseline group.
        """
        return self.observatory.get_redundant_baselines(
            bl_min=self.bl_min, bl_max=self.bl_max, ndecimals=self.redundancy_tol
        )

    def __getstate__(self):
        """Get state so that defaultdict is not used."""
        d = dict(self.__dict__.items())
        if "baseline_groups" in d:
            d["baseline_groups"] = dict(d["baseline_groups"])
        return d

    def __setstate__(self, state):
        """Set state so that defaultdict is restored."""
        if "baseline_groups" in state:
            state["baseline_groups"] = defaultdict(list, state["baseline_groups"])
        self.__dict__.update(state)

    @cached_property
    def baseline_group_coords(self) -> un.Quantity[un.m]:
        """Co-ordinates of baseline groups in metres."""
        return self.observatory.baseline_coords_from_groups(self.baseline_groups)

    @cached_property
    def baseline_group_counts(self) -> np.ndarray:
        """The number of baselines in each group."""
        return self.observatory.baseline_weights_from_groups(self.baseline_groups)

    @property
    def frequency(self) -> un.Quantity[un.MHz]:
        """Frequency of the observation."""
        return self.observatory.frequency

    @cached_property
    def uv_coverage(self) -> np.ndarray:
        # sourcery skip: assign-if-exp, swap-if-expression
        """A 2D array specifying the effective number of baselines in a grid of UV.

        Defined after earth rotation synthesis for a particular LST bin.
        The u-values on each side of the grid are given by :func:`ugrid`.
        """
        if not self.coherent:
            fnc = self.observatory.grid_baselines_incoherent
        else:
            fnc = self.observatory.grid_baselines_coherent

        return fnc(
            baselines=self.baseline_group_coords,
            weights=self.baseline_group_counts,
            integration_time=self.integration_time,
            bl_min=self.bl_min,
            bl_max=self.bl_max,
            observation_duration=self.obs_duration,
            ndecimals=self.redundancy_tol,
        )

    @cached_property
    def n_lst_bins(self) -> float:
        """
        Number of LST bins in the complete observation.

        An LST bin is considered a chunk of time that may be averaged
        coherently, so this is given by `hours_per_day/obs_duration`,
        where `obs_duration` is the time it takes for a source to travel
        through the beam FWHM.
        """
        return (self.time_per_day / self.obs_duration).to("").value

    @cached_property
    def Tsky(self) -> un.Quantity[un.K]:
        """Temperature of the sky at the default frequency."""
        return self.tsky_amplitude.to("K") * (self.frequency / self.tsky_ref_freq) ** (
            -self.spectral_index
        )

    @cached_property
    def Tsys(self) -> un.Quantity[un.K]:
        """System temperature (i.e. Tsky + Trcv)."""
        return self.Tsky.to("K") + self.observatory.Trcv.to("K")

    @cached_property
    def redshift(self) -> float:
        """Central redshift of the observation."""
        return conv.f2z(self.frequency)

    @cached_property
    def eta(self) -> un.Quantity[1 / un.MHz]:
        """The fourier dual of the frequencies of the observation."""
        return np.fft.fftfreq(
            self.n_channels, self.bandwidth.to("MHz") / self.n_channels
        )

    @cached_property
    def kparallel(self) -> un.Quantity[un.littleh / un.Mpc]:
        """1D array of kpar values, defined by the bandwidth and number of channels.

        Order of the values is the same as `fftfreq` (i.e. zero-first)
        """
        return (
            conv.dk_deta(
                self.redshift, self.cosmo, approximate=self.use_approximate_cosmo
            )
            * self.eta
        )

    @cached_property
    def total_integration_time(self) -> un.Quantity[un.s]:
        """The total (effective) integration time over UV bins for a particular LST bin.

        The u-values on each side of the grid are given by :func:`ugrid`.
        """
        return self.uv_coverage * self.n_days * self.integration_time.to(un.s)

    @cached_property
    def Trms(self) -> un.Quantity[un.K]:
        """Effective radiometric noise temperature per UV bin.

        (i.e. divided by bandwidth and integration time).
        The u-values on each side of the grid are given by :func:`ugrid`.
        """
        out = np.ones(self.total_integration_time.shape) * np.inf * self.Tsys.unit
        mask = self.total_integration_time > 0
        out[mask] = self.Tsys.to("K") / np.sqrt(
            2 * self.bandwidth * self.total_integration_time[mask]
        ).to("")
        return out

    @cached_property
    def ugrid(self) -> np.ndarray:
        """Centres of the linear grid which defines a side of the UV grid.

        The UV grid is defined by :func:`uv_coverage`.
        """
        return self.observatory.ugrid(self.bl_max)

    @cached_property
    def ugrid_edges(self) -> np.ndarray:
        """Edges of the linear grid which defines a side of the UV grid.

        The UV grid is defined by :func:`uv_coverage`.
        """
        return self.observatory.ugrid_edges(self.bl_max)

    def clone(self, **kwargs) -> Observation:
        """Create a clone of this instance, with arbitrary changes to parameters."""
        return attr.evolve(self, **kwargs)
