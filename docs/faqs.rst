===
FAQ
===

How do I use my own theoretical model?
--------------------------------------
The theoretical model is required to compute sample (or cosmic) variance.
By default, ``21cmSense`` uses a state-of-the-art model run with `21cmFAST <https://github.com/21cmFAST/21cmFAST>`_
from https://arxiv.org/abs/2110.13919. This model provides power spectra at multiple
redshifts and wavenumbers, and should be sufficient for most sensitivity calculations.

If you would like to provide your own theoretical model, you should provide your own
implementation of the :class:`py21cmsense.theory.TheoryModel` class. This class
has two required definitions: a class variable ``use_littleh``, which determines the
units of ``k`` that are passed to it, and the method ``delta_squared``. A simple
power-law theoretical model may be implemented like this::

    from py21cmsense.theory import TheoryModel
    from astropy import units as un

    class PowerLaw(TheoryModel):
        use_littleh = False

        def delta_squared(self, z: float, k: np.ndarray) -> un.Quantity[un.mK**2]:
            return ((1 + z)**2 * (k / 100.0)**-2) << un.mK**2

Note that the ``delta_squared`` method must exactly match this signature, as it is
called from within the :class:`py21cmsense.sensitivity.PowerSpectrum` class.
The default :class:`py21cmsense.theory.EOS2021` model loads a 2D array of values
in redshift and *k* and interpolates them. You could easily set up something similar to
this.

How do I use my custom theory function from the CLI?
----------------------------------------------------

If you have defined a custom theory model as in the previous question, you can use it
from the command line by adding a couple of lines to your sensitivity YAML
configuration. Imagine you wrote a package called ``my_theory``, within which there
was a module called ``module`` containing the ``PowerLaw`` class defined above (in
practice, whatever module ``PowerLaw`` resides in just has to be importable on the
``PYTHONPATH`` of your environment). You would then add the following lines to your
sensitivity YAML configuration::

    plugins:
      my_theory.module

    theory_model:
      PowerLaw


How do I change cosmology?
--------------------------

You can change the cosmology in the ``Obervation`` class::

    >>> from py21cmsense import Observation
    >>> from astropy.cosmology import WMAP9
    >>> obs = Observation(cosmo=WMAP9, ...)

In practice, the cosmology makes little difference in the calculcations.

How do I define a new ``PrimaryBeam``?
--------------------------------------

First, you should consider whether you really need one. ``21cmSense`` only takes into
account the frequency, beam width and UV resolution -- not the precise shape.
The beam width specifies the beam-crossing time, which determines how long one can
integrate coherently. The UV resolution is assumed to be a single number, by default
based on the inverse of the beam width. Unless these specific quantities are significantly
different for your desired beam, there is no benefit to being more precise than this.

If you do want to specify your own beam, you must sub-class from ``PrimaryBeam``, you
must define the same methods as defined by the following definition of the
``GaussianBeam``::

    @attr.s(frozen=True)
    class GaussianBeam(PrimaryBeam):
        """
        A simple Gaussian Primary beam.

        Parameters
        ----------
        frequency
            The fiducial frequency at which the beam operates, assumed to be in MHz
            unless otherwise defined.
        dish_size
            The size of the (assumed circular) dish, assumed to be in meters unless
            otherwise defined. This generates the beam size.
        """

        dish_size: tp.Length = attr.ib(
            validator=(tp.vld_physical_type("length"), ut.positive)
        )

        @property
        def dish_size_in_lambda(self) -> float:
            """The dish size in units of wavelengths."""
            return (self.dish_size / (cnst.c / self.frequency)).to("").value

        @property
        def uv_resolution(self) -> un.Quantity[1 / un.radian]:
            """The appropriate resolution of a UV cell given the beam size."""
            return self.dish_size_in_lambda

        @property
        def area(self) -> un.Quantity[un.steradian]:
            """The integral of the beam over angle, in sr."""
            return 1.13 * self.fwhm**2

        @property
        def width(self) -> un.Quantity[un.radian]:
            """The width of the beam (i.e. sigma), in radians."""
            return un.rad * 0.45 / self.dish_size_in_lambda

        @property
        def fwhm(self) -> un.Quantity[un.radians]:
            """The full-width half maximum of the beam."""
            return 2.35 * self.width

        @property
        def sq_area(self) -> un.Quantity[un.steradian]:
            """The integral of the squared beam, in sr."""
            return self.area / 2

        @property
        def first_null(self) -> un.Quantity[un.radians]:
            """The angle of the first null of the beam."""
            return un.rad * 1.22 / self.dish_size_in_lambda
