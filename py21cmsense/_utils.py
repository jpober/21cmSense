import numpy as np
from astropy import units as un

from . import config


class UnitError(ValueError):
    pass


def apply_or_convert_unit(unit, allow_unitless=False):
    """
    Return a function that converts numbers to quantities (or converts quantities to
    specified units).

    Parameters
    ----------
    unit : astropy unit or str

    Returns
    -------
    callable: function which converts its argument to a quantity
    """

    def converter(quantity):
        if hasattr(quantity, "unit"):
            return quantity.to(unit)
        else:
            if allow_unitless:
                return quantity

            if not config.ALLOW_DEFAULT_UNITS:
                raise UnitError("This value is required to have units convertible to {}".format(unit))

            if isinstance(unit, str):
                return quantity * getattr(un, unit)
            else:
                return quantity * unit

    return converter


def between(min, max):
    def converter(val):
        assert min <= val <= max

    return converter


def positive(x):
    assert x > 0, "must be positive"


def nonnegative(x):
    assert x >= 0, "must be non-negative"


# A function used for binning
def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx


def trunc(x, ndecimals=0):
    decade = 10 ** ndecimals
    return np.trunc(x * decade) / decade
