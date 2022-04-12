=========
Changelog
=========

Unreleased
==========

Fixed
-----
* Bug in ``dL_df`` (missing square).
* Use ``yaml.SafeLoader`` instead of ``yaml.FullyLoader``.

Changed
-------
* Many computations altered to use numpy vectorization over for loop implementations.
  Including but not limited to:

    * 2D sensitivity calculation.
    * 1D sensitivity calculation.
    * UVW calculation as a function of time.

* ``_utils.find_nearest`` can solve for the index of an array of inputs.

Features
--------
* Added a parameter ``systematics_mask`` to ``PowerSpectrum`` sensitivity, which enables
  arbitrary k-modes to be masked out in the sensitivity calculation.
* ``track`` option to ``Observation``. This is an alias for ``obs_duration`` but has
  a closer resemblance to the original 21cmSense v1.
* New ``calculate_sensitivity_2d_grid`` method that makes it easier to obtain a gridded
  cylindrical power spectrum sensitivity for arbitrary bins.

v2.0.0
======
A major overhaul of the code, making it object-oriented and modular.

Features
--------
* Python 3 compatibility.
* Class-based code, with specific objects for ``Observatory``, ``Observation``,
  and class templates for different ``Sensitivity`` calculations, including default
  ``PowerSpectrum``.
* Clean ``attrs``-based classes.
* ``click``-based CLI interface.
* Ability to pickle intermediate classes for re-use.
* Ability to specify observation frequency more freely (and self-consistently)
* Removal of all usage of ``aipy`` as it is far too complex for a simple calculation such as this.
* Ability to specify ``Observation`` using ``pyuvdata`` objects.
* Useful docstrings throughout.
* Explicit cosmological calculations based on astropy.
* More flexible: extra parameters for foreground model and integration time, among others.
* Agreement with previous versions (not exact agreement, because of the increase in accuracy
  from using astropy).
* Tools for plotting output -- and default 1D PS plot from the CLI.
* All quantities have appropriate units (from astropy).
* Example documentation, and example configuration files.
* Configuration files are no longer python files... they are YAML.
