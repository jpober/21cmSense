"""A package for calculate sensitivies of 21-cm interferometers."""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("21cmSense")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError

from . import data, yaml
from .antpos import hera
from .beam import GaussianBeam
from .observation import Observation
from .observatory import Observatory
from .sensitivity import PowerSpectrum
