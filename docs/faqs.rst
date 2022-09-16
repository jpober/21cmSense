===
FAQ
===

How do I change cosmology?
--------------------------

You can change the cosmology via the ``config`` module, eg.::

    >>> from py21cmsense import config
    >>> from astropy.cosmology import WMAP9
    >>> config.COSMO = WMAP9
    >>> # Run standard 21cmSense calculations...

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
