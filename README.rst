=========
21cmSense
=========

.. image:: https://travis-ci.org/steven-murray/21cmSense.svg?branch=master
    :target: https://travis-ci.org/steven-murray/21cmSense
.. image:: https://codecov.io/gh/steven-murray/21cmSense/branch/master/graph/badge.svg
  :target: https://codecov.io/gh/steven-murray/21cmSense
.. image:: https://img.shields.io/badge/License-GPLv3-blue.svg
  :target: https://www.gnu.org/licenses/gpl-3.0
.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
  :target: https://github.com/psf/black

A python package for calculating the expected sensitivities of 21cm experiments
to the Epoch of Reionization and/or Cosmic Dawn power spectrum.

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
For details of the observing strategy assumed by this code, and other relevant
scientific information, please see

    Pober et al. 2013AJ....145...65P

and

    Pober et al. 2014ApJ...782...66P

If you use this code in any of your work, please acknowledge these papers,
and provide a link to this repository.
