=========
21cmSense
=========

A python package for calculating the expected sensitivities of 21cm experiments
to the Epoch of Reionization power spectrum.  For details of the observing
strategy assumed by this code, and other relevant scientific information, please
see

    Pober et al. 2013AJ....145...65P

and

    Pober et al. 2014ApJ...782...66P.

Installation
============
Clone/download the package and run ``pip install [-e] .`` in the top-level.

If you are a ``conda`` user (which we recommend), you may want to install the following
using ``conda`` rather than them being automatically installed with pip::

    $ conda install numpy scipy pyyaml astropy

Usage
=====
There are two ways to use this code: as a python library or via the CLI.
The simplest possible usage is by using the CLI as follows::

    $ sense calc-sense <SENSITIVITY_CONFIG_FILE.yml>

Other options to the ``calc-sense`` program can be read by using::

    $ sense calc-sense --help

An example config file is in this repository under ``example_configs/sensitivity_hera.yml``,
which details the various parameters that can be set. In all, three configuration files
are required -- one defining an ``observatory``, another defining an ``observation``, and the
``sensitivity`` one already mentioned.

The CLI can also be used in a two-step process, by running::

    $ sense grid-baselines <OBSERVATION_CONFIG_FILE.yml>

and then::

    $ sense calc-sense <SENSITIVITY_CONFIG_FILE.yml> --array-file=<ARRAY_FILE.pkl>

where the ``ARRAY_FILE`` is produced in the first step (and its location is printed during
the execution).



Acknowledgment
==============
If you use this code in any of your work, please acknowledge

    Pober et al. 2013AJ....145...65P

and

    Pober et al. 2014ApJ...782...66P

and provide a link to this repository.
