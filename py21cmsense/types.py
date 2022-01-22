"""Module dealing with types and units throughout the package."""
from __future__ import annotations

import attr
import numpy as np
from astropy import constants as cnst
from astropy import units as u
from astropy.cosmology.units import littleh
from typing import Any, Callable, Type, Union

u.add_enabled_units([littleh])


class UnitError(ValueError):
    """An error pertaining to having incorrect units."""

    pass


Length = u.Quantity["length"]
Meters = u.Quantity["m"]
Time = u.Quantity["time"]
Frequency = u.Quantity["frequency"]
Temperature = u.Quantity["temperature"]
TempSquared = u.Quantity[u.get_physical_type("temperature") ** 2]
Wavenumber = u.Quantity[littleh / u.Mpc]
Delta = u.Quantity[u.mK ** 2]

time_as_distance = [
    (
        u.s,
        u.m,
        lambda x: cnst.c.to_value("m/s") * x,
        lambda x: x / cnst.c.to_value("m/s"),
    )
]


def vld_physical_type(unit: str) -> Callable[[Any, attr.Attribute, Any], None]:
    """Attr validator to check physical type."""

    def _check_type(self: Any, att: attr.Attribute, val: Any):
        if not isinstance(val, u.Quantity):
            raise UnitError(f"{att.name} must be an astropy Quantity!")
        if val.unit.physical_type != unit:
            raise u.UnitConversionError(
                f"{att.name} must have physical type of '{unit}'. "
                f"Got '{val.unit.physical_type}'"
            )

    return _check_type


def vld_unit(unit, equivalencies=()):
    """Attr validator to check unit equivalence."""

    def _check_unit(self, att, val):
        if not isinstance(val, u.Quantity):
            raise UnitError(f"{att.name} must be an astropy Quantity!")

        if not val.unit.is_equivalent(unit, equivalencies):
            raise u.UnitConversionError(
                f"{att.name} not convertible to {unit}. Got {val.unit}"
            )

    return _check_unit
