"""
Module providing the definition of an Observatory.

This replaces the original usage of an aipy.AntennaArray with something much more
simple, and suited to the needs of this particular package.
"""

from cached_property import cached_property
import attr
from attr import validators as vld
from astropy import constants as cnst
from . import _utils as ut
from astropy import units as u
import numpy as np


@attr.s(frozen=True, kw_only=True)
class Observatory(object):
    antpos = attr.ib()
    latitude = attr.ib(0, converter=ut.apply_or_convert_unit('rad'),
                       validator=ut.between(-np.pi*u.rad/2, np.pi*u.rad/2))

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
        if freq is None and self.reference_freq is not None:
            freq = self.reference_freq
        elif freq is None and self.reference_freq is None:
            raise ValueError("You must supply a frequency")

        freq = ut.apply_or_convert_unit('MHz')(freq)
        return self.dish_size / (cnst.c / freq)

    @cached_property
    def dish_size_in_lambda_ref(self):
        try:
            return self.dish_size_in_lambda()
        except ValueError:
            raise AttributeError("dish_size_in_lambda_ref only exists when reference_freq is set")

    def beam_width(self, freq=None):
        return 1.13 * (2.35 * (0.45 / self.dish_size_in_lambda(freq))) ** 2

    def first_null(self, freq=None):
        # for an airy disk, even though beam model is Gaussian
        return 1.22 / self.dish_size_in_lambda(freq)

    @cached_property
    def n_antennas(self):
        return len(self.antpos)

    def new(self, **kwargs):
        """Return a clone of this instance, but change kwargs"""
        return attr.evolve(self, **kwargs)