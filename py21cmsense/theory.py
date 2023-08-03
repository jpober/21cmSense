"""A module defining an interface for theoretical predictions, eg. power spectra.

This is simply so that one can calculate sample variance values for a particular
theoretical model, with a uniform interface.

To register a class as a theory power spectrum, simply make it a subclass of
``TheoryModel``. You must define two things on the class: (1) a class attribute
"use_littleh" which says whether the function that evaluates the power spectrum expects
the wavenumber in h/Mpc or 1/Mpc, and (2) a ``__call__`` method, which takes in a float
redshift and an array of wavenumbers, and returns Delta^2 as an astropy Quantity with
units mK^2.
"""
import abc
import numpy as np
import warnings
from astropy import units as un
from pathlib import Path
from scipy.interpolate import InterpolatedUnivariateSpline, RectBivariateSpline

_ALL_THEORY_POWER_SPECTRA = {}


class TheoryModel(abc.ABC):
    """Abstract base class for theory models.

    Subclasses must implement the :meth:`delta_squared` method.
    """

    #: Whether the theory model uses little-h units for wavenumbers.
    use_littleh: bool = False

    def __init_subclass__(cls) -> None:
        """Add the subclass to the plugin dict."""
        _ALL_THEORY_POWER_SPECTRA[cls.__name__] = cls
        return super().__init_subclass__()

    @abc.abstractmethod
    def delta_squared(self, z: float, k: np.ndarray) -> un.Quantity[un.mK**2]:
        """Compute Delta^2(k, z) for the theory model.

        Parameters
        ----------
        z
            The redshift (should be a float).
        k
            The wavenumbers, either in units of 1/Mpc if use_littleh=False, or
            h/Mpc if use_littleh=True.

        Returns
        -------
        delta_squared
            An array of delta_squared values in units of mK^2.
        """
        pass  # pragma: no cover


class EOS2021(TheoryModel):
    """Theory model from EOS2021 (https://arxiv.org/abs/2110.13919)."""

    use_littleh = False

    def __init__(self):
        pth = Path(__file__).parent / "data/eos2021"
        z = np.fromfile(pth / "1pt5Gpc_EOS_coeval_pow_zlist.bin")
        # TODO: we divide by 2.5 here as the k values on the EOS2021 GDrive are wrong --
        # they are for the 600 Mpc box instead of the 1.5 Gpc box. Later when that's
        # fixed we should just fix the data here.
        self.k = np.fromfile(pth / "1pt5Gpc_EOS_coeval_pow_kbins.bin") / 2.5
        coeval_ps = np.fromfile(pth / "1pt5Gpc_EOS_coeval_pow_P21.bin").reshape(
            (z.size, self.k.size)
        )

        # Sort in order of ascending redshift
        order = np.argsort(z)
        self.z = z[order]
        self.coeval_ps = coeval_ps[order]

        self.spline = RectBivariateSpline(self.z, self.k, self.coeval_ps, ky=1)

    def delta_squared(self, z: float, k: np.ndarray) -> un.Quantity[un.mK**2]:
        """Compute Delta^2(k, z) for the theory model.

        Parameters
        ----------
        z
            The redshift (should be a float).
        k
            The wavenumbers, either in units of 1/Mpc if use_littleh=False, or
            h/Mpc if use_littleh=True.

        Returns
        -------
        delta_squared
            An array of delta_squared values in units of mK^2.
        """
        if np.any(k > self.k.max()):
            warnings.warn(
                f"Extrapolating above the simulated theoretical k: {k.max()} > {self.k.max()}",
                stacklevel=2,
            )
        if np.any(k < self.k.min()):
            warnings.warn(
                f"Extrapolating below the simulated theoretical k: {k.min()} < {self.k.min()}",
                stacklevel=2,
            )
        if not self.z.min() <= z <= self.z.max():
            warnings.warn(
                f"Extrapolating beyond simulated redshift range: {z} not in range ({self.z.min(), self.z.max()})",
                stacklevel=2,
            )

        return self.spline(z, k, grid=False) << un.mK**2


class Legacy21cmFAST(TheoryModel):
    """Simple 21cmFAST-based theory model explicitly for z=9.5, from 21cmSense v1."""

    use_littleh: bool = False

    def __init__(self) -> None:
        pth = (
            Path(__file__).parent
            / "data/ps_no_halos_nf0.521457_z9.50_useTs0_zetaX-1.0e+00_200_400Mpc_v2"
        )
        self.k, self.delta_squared_raw = np.genfromtxt(pth).T[:2]

        self.spline = InterpolatedUnivariateSpline(self.k, self.delta_squared_raw, k=1)

    def delta_squared(self, z: float, k: np.ndarray) -> un.Quantity[un.mK**2]:
        """Compute Delta^2(k, z) for the theory model.

        Parameters
        ----------
        z
            The redshift (should be a float).
        k
            The wavenumbers, either in units of 1/Mpc if use_littleh=False, or
            h/Mpc if use_littleh=True.

        Returns
        -------
        delta_squared
            An array of delta_squared values in units of mK^2.
        """
        if np.any(k > self.k.max()):
            warnings.warn(
                f"Extrapolating above the simulated theoretical k: {k.max()} > {self.k.max()}",
                stacklevel=2,
            )
        if np.any(k < self.k.min()):
            warnings.warn(
                f"Extrapolating below the simulated theoretical k: {k.min()} < {self.k.min()}",
                stacklevel=2,
            )
        if not 9 < z < 10:
            warnings.warn(
                f"Theory power corresponds to z=9.5, not z={z:.2f}",
                stacklevel=2,
            )

        return self.spline(k) << un.mK**2
