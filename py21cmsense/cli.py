#! /usr/bin/env python
"""
Creates an array file for use by sensitivity.py.  The main product is the uv coverage produced by the array during the
time it takes the sky to drift through the primary beam; other array parameters are also saved.
Array specific information comes from an aipy cal file.  If track is set, produces the uv coverage
for the length specified instead of that set by the primary beam.
"""
from __future__ import division
from __future__ import print_function

import os
import tempfile
import yaml

from os import path

import click
import numpy as np

from . import observation
from . import sensitivity as sense

import pickle

main = click.Group()


@main.command()
@click.argument(
    'configfile',
    type=click.Path(exists=True, dir_okay=False),
)
@click.option(
    '--direc', type=click.Path(exists=True, dir_okay=True, file_okay=False),
    help="directory to save output file", default='.'
)
def grid_baselines(configfile, direc):
    obs = observation.Observation.from_yaml(configfile)

    filepath = os.path.join(
        direc,
        "drift_blmin%0.f_blmax%0.f_%.3fGHz_arrayfile.pkl" % (obs.bl_min.value, obs.bl_max.value,
                                                             obs.frequency.to("GHz").value)
    )

    with open(filepath, 'wb') as fl:
        pickle.dump(obs, fl)

    print("There are {} baseline types".format(len(obs.baseline_groups)))
    print("Saving array file as {}".format(filepath))


@main.command()
@click.argument(
    "configfile",
    type=click.Path(exists=True, dir_okay=False),
)
@click.option(
    '--array-file', default=None,
    type=click.Path(exists=True, dir_okay=False, file_okay=True),
    help="array file created with grid-baselines",
)
@click.option(
    '--direc', type=click.Path(exists=True, dir_okay=True, file_okay=False),
    help="directory to save output file", default='.'
)
@click.option(
    '--fname', default=None, type=click.Path(), help="filename to save output file"
)
@click.option(
    '--thermal/--no-thermal', default=True, help="whether to include thermal noise"
)
@click.option(
    '--samplevar/--no-samplevar', default=True,  help="whether to include sample variance"
)
@click.option(
    '--write-significance/--no-significance',
    default=True,
    help="whether to write the significance of the PS to screen"
)
def calc_sense(configfile, array_file, direc, fname, thermal, samplevar, write_significance):
    # If given an array-file, overwrite the "observation" parameter
    # in the config with the pickled array file, which has already
    # calculated the uv_coverage, hopefully.
    if array_file is not None:
        with open(configfile) as fl:
            cfg = yaml.load(fl, Loader=yaml.FullLoader)
        cfg['observation'] = path.abspath(array_file)

        configfile = tempfile.mktemp()
        with open(configfile, 'w') as fl:
            yaml.dump(cfg, fl)

    sensitivity = sense.PowerSpectrum.from_yaml(configfile)

    out = {}
    if thermal:
        print("Getting Thermal Variance")
        out['thermal'] = sensitivity.calculate_sensitivity_1d(sample=False)
    if samplevar:
        print("Getting Sample Variance")
        out['sample'] = sensitivity.calculate_sensitivity_1d(thermal=False, sample=True)
    if thermal and samplevar:
        print("Getting Combined Variance")
        out['sample+thermal'] = sensitivity.calculate_sensitivity_1d(thermal=True, sample=True)

    # save results to output npz
    if fname is None:
        fname = "{model}_{freq:.3f}.npz".format(model=sensitivity.foreground_model,
                                                 freq=sensitivity.observation.frequency)

    np.savez(
        os.path.join(direc, fname),
        ks=sensitivity.k1d,
        **out
    )

    print(out)
    if write_significance:
        sig = sensitivity.calculate_significance(thermal=thermal, sample=samplevar) #... ?
        print("Significance of detection: ", sig)


#@click.option(
#     "-m",
#     "--model",
#     default="moderate",
#     type=click.Choice(['pessimistic', 'moderate', 'optimistic']),
#     help="""
#     The model of the foreground wedge to use.  Three options are 'pess' (all k
#     modes inside horizon + buffer are excluded, and all baselines are added
#     incoherently), 'mod' (all k modes inside horizon + buffer are excluded, but
#     all baselines within a uv pixel are added coherently), and 'opt' (all modes
#     k modes inside the primary field of view are excluded).  See Pober et al.
#     2014 for more details.
#     """,
# )
# @click.option(
#     "--buffer",
#     default=0.1,
#     type=float,
#     help="""
#     The size of the additive buffer outside the horizon to exclude in the
#     pessimistic and moderate models.
#     """,
# )
# @click.option(
#     "--eor-ps-file",
#     type=click.Path(exists=True, dir_okay=False),
#     default=None,
#     help="""The model epoch of reionization power spectrum.  The code is built
#     to handle output power spectra from 21cmFAST.
#     """,
# )
# @click.option(
#     "--n-days",
#     default=180.0,
#     type=float,
#     help="""
#     The total number of days observed.  The default is 180, which is the maximum
#     a particular R.A. can be observed in one year if one only observes at night.
#     The total observing time is ndays*n_per_day.
#     """,
# )
# @click.option(
#     "--n-per-day",
#     default=6.0,
#     type=float,
#     help="""The number of good observing hours per day.  This corresponds to the
#     size of a low-foreground region in right ascension for a drift scanning
#     instrument.  The total observing time is ndays*n_per_day.  Default is 6.
#     If simulating a tracked scan, n_per_day should be a multiple of the length
#     of the track (i.e. for two three-hour tracks per day, n_per_day should be 6).
#     """,
# )
# @click.option(
#     "-b",
#     "--bandwidth",
#     default=0.008,
#     type=float,
#     help="""
#     Cosmological bandwidth in GHz.  Note this is not the total instrument
#     bandwidth, but the redshift range that can be considered co-eval.
#     Default is 0.008 (8 MHz).
#     """,
# )
# @click.option(
#     "--n-chan",
#     default=82,
#     type=int,
#     help="""
#     Integer number of channels across cosmological bandwidth.  Defaults to 82,
#     which is equivalent to 1024 channels over 100 MHz of bandwidth.
#     Sets maximum k_parallel that can be probed, but little to no overall effect
#     on sensitivity.
#     """,
# )
# @click.option(
#     "--ns/--no-ns",
#     default=True,
#     help="""
#     Remove pure north/south baselines (u=0) from the sensitivity calculation.
#     These baselines can potentially have higher systematics, so excluding them
#     represents a conservative choice.
#     """,
# )
# @click.option(
#     "--hubble",
#     default=0.7,
#     type=float,
#     help="Hubble parameter."
# )