import attr
from abc import ABC, abstractclassmethod, abstractmethod, abstractproperty
from astropy import constants as cnst
from astropy import units as un

from . import _utils as ut


@attr.s(frozen=True)
class PrimaryBeam(ABC):
    """
    A Base class defining a Primary Beam and the methods it requires to define.
    """

    frequency = attr.ib(
        converter=ut.apply_or_convert_unit("MHz"), validator=ut.positive
    )

    def new(self, **kwargs):
        """Return a clone of this instance, but change kwargs"""
        return attr.evolve(self, **kwargs)

    @abstractmethod
    def area(self, freq=None):
        """Beam area (sr)"""
        pass

    @abstractmethod
    def width(self, freq=None):
        """Beam width (rad)"""
        pass

    @abstractmethod
    def first_null(self, freq=None):
        """An approximation of the first null of the beam"""
        pass

    @abstractmethod
    def sq_area(self, freq=None):
        """The area of the beam^2"""
        pass

    @abstractmethod
    def b_eff(self, freq=None):
        """Effective beam area (Parsons 2014)"""
        pass

    @abstractproperty
    def uv_resolution(self):
        pass

    @abstractclassmethod
    def from_uvbeam(cls):
        pass


@attr.s(frozen=True)
class GaussianBeam(PrimaryBeam):
    """
    A simple Gaussian Primary beam.

    Parameters
    ----------
    frequency : float or Quantity
        The fiducial frequency at which the beam operates, assumed to be in MHz
        unless otherwise defined.
    dish_size : float or Quantity
        The size of the (assumed circular) dish, assumed to be in meters unless
        otherwise defined. This generates the beam size.
    """

    dish_size = attr.ib(converter=ut.apply_or_convert_unit("m"), validator=ut.positive)

    def dish_size_in_lambda(self, freq=None):
        """The dish size in units of wavelengths, for a given frequency.

        If frequency is not given, uses the instance's `frequency`
        """
        freq = ut.apply_or_convert_unit("MHz")(freq or self.frequency)
        return (self.dish_size / (cnst.c / freq)).to("").value

    def area(self, freq=None):
        """The integral of the beam over angle, in sr.

        If frequency is not given, uses the instance's `frequency`
        """
        return 1.13 * self.fwhm(freq) ** 2

    def width(self, freq=None):
        """The width of the beam (i.e. sigma), in radians

        If frequency is not given, uses the instance's `frequency`
        """
        return un.rad * 0.45 / self.dish_size_in_lambda(freq)

    def fwhm(self, freq=None):
        """The full-width half maximum of the beam

        If frequency is not given, uses the instance's `frequency`
        """
        return 2.35 * self.width(freq)

    def sq_area(self, freq=None):
        """The integral of the squared beam, in sr

        If frequency is not given, uses the instance's `frequency`
        """
        return self.area(freq) / 2

    def b_eff(self, freq=None):
        return self.area(freq) ** 2 / self.sq_area(freq)

    def first_null(self, freq=None):
        """The angle of the first null of the beam.

        .. note:: The Gaussian beam has no null, and in this case we use the first null
                  for an airy disk.

        If frequency is not given, uses the instance's `frequency`
        """
        return un.rad * 1.22 / self.dish_size_in_lambda(freq)

    @property
    def uv_resolution(self):
        """The appropriate resolution of a UV cell given the beam size"""
        return self.dish_size_in_lambda()

    @classmethod
    def from_uvbeam(cls):
        raise NotImplementedError("Coming Soon!")
