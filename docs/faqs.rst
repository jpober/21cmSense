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
