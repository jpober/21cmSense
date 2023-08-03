"""Module dealing with types and units throughout the package."""
from __future__ import annotations

import attr
import numpy as np
from astropy import constants as cnst
from astropy import units as un
from astropy.cosmology.units import littleh, redshift
from typing import Any, Callable, Type, Union

un.add_enabled_units([littleh, redshift])


class UnitError(ValueError):
    """An error pertaining to having incorrect units."""

    pass


Length = un.Quantity["length"]
Meters = un.Quantity["m"]
Time = un.Quantity["time"]
Frequency = un.Quantity["frequency"]
Temperature = un.Quantity["temperature"]
TempSquared = un.Quantity[un.get_physical_type("temperature") ** 2]
Wavenumber = un.Quantity[littleh / un.Mpc]
Delta = un.Quantity[un.mK**2]

time_as_distance = [
    (
        un.s,
        un.m,
        lambda x: cnst.c.to_value("m/s") * x,
        lambda x: x / cnst.c.to_value("m/s"),
    )
]


def vld_physical_type(unit: str) -> Callable[[Any, attr.Attribute, Any], None]:
    """Attr validator to check physical type."""

    def _check_type(self: Any, att: attr.Attribute, val: Any):
        if not isinstance(val, un.Quantity):
            raise UnitError(f"{att.name} must be an astropy Quantity!")
        if val.unit.physical_type != unit:
            raise un.UnitConversionError(
                f"{att.name} must have physical type of '{unit}'. "
                f"Got '{val.unit.physical_type}'"
            )

    return _check_type


def vld_unit(unit, equivalencies=()):
    """Attr validator to check unit equivalence."""

    def _check_unit(self, att, val):
        if not isinstance(val, un.Quantity):
            raise UnitError(f"{att.name} must be an astropy Quantity!")

        if not val.unit.is_equivalent(unit, equivalencies):
            raise un.UnitConversionError(
                f"{att.name} not convertible to {unit}. Got {val.unit}"
            )

    return _check_unit
