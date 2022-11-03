"""A module defining baseline filters.

While you can simply use a function that takes a single baseline (with three
co-ordinates) and returns a bool, this module provides standard kinds of filters (eg.
using baselines within a certain length range). It also enables loading the filters
from string names, useful for YAML files.
"""
import abc
import attr
import numpy as np
import warnings
from astropy import units as un
from pathlib import Path

from . import types as tp

_ALL_BASELINE_FILTERS = {}


class BaselineFilter(abc.ABC):
    """Abstract base class for theory models.

    Subclasses must implement the :meth:`delta_squared` method.
    """

    def __init_subclass__(cls) -> None:
        """Add the subclass to the plugin dict."""
        _ALL_BASELINE_FILTERS[cls.__name__] = cls
        return super().__init_subclass__()

    @abc.abstractmethod
    def __call__(self, bl: tp.Length) -> bool:
        """Determine whether a baseline should be included or not.

        Parameters
        ----------
        bl
            The baseline, which will be a length-3 array whose elements are
            quantities (in length units, typically metres).

        Returns
        -------
        bool
            True if the baseline should be included.
        """
        pass  # pragma: no cover


@attr.define
class BaselineRange(BaselineFilter):
    """Theory model from EOS2021 (https://arxiv.org/abs/2110.13919)."""

    bl_min: tp.Length = attr.field(
        default=0 * un.m, validator=tp.vld_physical_type("length")
    )
    bl_max: tp.Length = attr.field(
        default=np.inf * un.m, validator=tp.vld_physical_type("length")
    )
    direction: str = attr.field(
        default="mag", validator=attr.validators.in_(("ew", "ns", "mag"))
    )

    @bl_max.validator
    def _bl_max_vld(self, att, val):
        if val <= self.bl_min:
            raise ValueError(
                "bl_max must be greater than bl_min, got "
                f"bl_min={self.bl_min} and bl_max={val}"
            )

    def __call__(self, bl: tp.Length) -> bool:
        """Determine whether a baseline should be included or not.

        Parameters
        ----------
        bl
            The baseline, which will be a length-3 array whose elements are
            quantities (in length units, typically metres).

        Returns
        -------
        bool
            True if the baseline should be included.
        """
        if self.direction == "ew":
            return self.bl_min <= bl[0] < self.bl_max
        elif self.direction == "ns":
            return self.bl_min <= bl[1] < self.bl_max
        elif self.direction == "mag":
            blsize = np.sqrt(bl[0] ** 2 + bl[1] ** 2)
            return self.bl_min <= blsize < self.bl_max
