#! /usr/bin/env python
"""
Creates an array file for use by sensitivity.py.  The main product is the uv coverage produced by the array during the
time it takes the sky to drift through the primary beam; other array parameters are also saved.
Array specific information comes from an aipy cal file.  If track is set, produces the uv coverage
for the length specified instead of that set by the primary beam.
"""
from __future__ import division, print_function

import click
import logging
import numpy as np
import os
import pickle
import tempfile
import yaml
from os import path

from . import observation
from . import sensitivity as sense

try:
    import matplotlib.pyplot as plt

    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False

main = click.Group()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("py21cmsense")


@main.command()
@click.argument("configfile", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--direc",
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
    help="directory to save output file",
    default=".",
)
def grid_baselines(configfile, direc):
    obs = observation.Observation.from_yaml(configfile)

    filepath = os.path.join(
        direc,
        "drift_blmin%0.f_blmax%0.f_%.3fGHz_arrayfile.pkl"
        % (obs.bl_min.value, obs.bl_max.value, obs.frequency.to("GHz").value),
    )

    with open(filepath, "wb") as fl:
        pickle.dump(obs, fl)

    print("There are {} baseline types".format(len(obs.baseline_groups)))
    print("Saving array file as {}".format(filepath))


@main.command()
@click.argument("configfile", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--array-file",
    default=None,
    type=click.Path(exists=True, dir_okay=False, file_okay=True),
    help="array file created with grid-baselines",
)
@click.option(
    "--direc",
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
    help="directory to save output file",
    default=".",
)
@click.option(
    "--fname", default=None, type=click.Path(), help="filename to save output file"
)
@click.option(
    "--thermal/--no-thermal", default=True, help="whether to include thermal noise"
)
@click.option(
    "--samplevar/--no-samplevar",
    default=True,
    help="whether to include sample variance",
)
@click.option(
    "--write-significance/--no-significance",
    default=True,
    help="whether to write the significance of the PS to screen",
)
@click.option(
    "-p/-P",
    "--plot/--no-plot",
    default=True,
    help="whether to plot the 1D power spectrum uncertainty",
)
@click.option(
    "--plot-title", default=None, type=str, help="title for the output 1D plot"
)
@click.option(
    "--prefix", default="", type=str, help="string prefix for all output files"
)
def calc_sense(
    configfile,
    array_file,
    direc,
    fname,
    thermal,
    samplevar,
    write_significance,
    plot,
    plot_title,
    prefix,
):
    # If given an array-file, overwrite the "observation" parameter
    # in the config with the pickled array file, which has already
    # calculated the uv_coverage, hopefully.
    if array_file is not None:
        with open(configfile) as fl:
            cfg = yaml.load(fl, Loader=yaml.FullLoader)
        cfg["observation"] = path.abspath(array_file)

        configfile = tempfile.mktemp()
        with open(configfile, "w") as fl:
            yaml.dump(cfg, fl)

    sensitivity = sense.PowerSpectrum.from_yaml(configfile)

    out = {}
    if thermal:
        print("Getting Thermal Variance")
        out["thermal"] = sensitivity.calculate_sensitivity_1d(sample=False)
    if samplevar:
        print("Getting Sample Variance")
        out["sample"] = sensitivity.calculate_sensitivity_1d(thermal=False, sample=True)
    if thermal and samplevar:
        print("Getting Combined Variance")
        out["sample+thermal"] = sensitivity.calculate_sensitivity_1d(
            thermal=True, sample=True
        )

    # save results to output npz
    if fname is None:
        fname = "{pfx}{model}_{freq:.3f}.npz".format(
            pfx=prefix + "_" if prefix else "",
            model=sensitivity.foreground_model,
            freq=sensitivity.observation.frequency,
        )

    np.savez(os.path.join(direc, fname), ks=sensitivity.k1d.value, **out)

    if write_significance:
        sig = sensitivity.calculate_significance(
            thermal=thermal, sample=samplevar
        )  # ... ?
        print("Significance of detection: ", sig)

    if plot and HAVE_MPL:
        for key, value in out.items():
            plt.plot(sensitivity.k1d, value, label=key)
        plt.xscale("log")
        plt.yscale("log")
        plt.legend()
        plt.title(plot_title)
        plt.savefig(
            "{pfx}{model}_{freq:.3f}.png".format(
                pfx=prefix + "_" if prefix else "",
                model=sensitivity.foreground_model,
                freq=sensitivity.observation.frequency,
            )
        )
