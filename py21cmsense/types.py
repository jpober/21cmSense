from __future__ import annotations
from astropy import units as u
from typing import Union, Type, Callable, Any
import numpy as np
import attr
from astropy.cosmology.units import littleh
from astropy import constants as cnst
u.add_enabled_units([littleh])

class UnitError(ValueError):
    """An error pertaining to having incorrect units."""
    pass

Length = u.Quantity['length']
Meters = u.Quantity['m']
Time = u.Quantity['time']
Frequency = u.Quantity['frequency']
Temperature = u.Quantity['temperature']
TempSquared = u.Quantity[u.get_physical_type('temperature')**2]
Wavenumber = u.Quantity[littleh / u.Mpc]
Delta = u.Quantity[u.mK**2]

time_as_distance = [
    (u.s, u.m, lambda x: cnst.c.to_value('m/s') * x, lambda x: x / cnst.c.to_value('m/s'))
]

def vld_physical_type(unit: str) -> Callable[[attr.Attribute, Any], None]:
    def _check_type(self, att, val):
        if not isinstance(val, u.Quantity):
            raise UnitError(f"{att.name} must be an astropy Quantity!")
        if val.unit.physical_type != unit:
            raise u.UnitConversionError(
                f"{att.name} must have physical type of '{unit}'. "
                f"Got '{val.unit.physical_type}'"
            )
    return _check_type

def vld_unit(unit, equivalencies=tuple()):
    def _check_unit(self, att, val):
        if not isinstance(val, u.Quantity):
            raise UnitError(f"{att.name} must be an astropy Quantity!")
        
        if not val.unit.is_equivalent(unit, equivalencies):
            raise u.UnitConversionError(f"{att.name} not convertible to {unit}. Got {val.unit}")

    return _check_unit