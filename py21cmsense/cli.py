"""CLI routines for 21cmSense."""
from __future__ import division, print_function

import click
import logging
import os
import pickle
import tempfile
import yaml
from os import path
from pathlib import Path
from rich.logging import RichHandler

from . import observation
from . import sensitivity as sense

try:
    import matplotlib.pyplot as plt

    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False

main = click.Group()

FORMAT = "%(message)s"
logging.basicConfig(
    level=logging.INFO, format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)

logger = logging.getLogger("py21cmsense")


@main.command()
@click.argument("configfile", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--direc",
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
    help="directory to save output file",
    default=".",
)
@click.option(
    "--outfile",
    "-o",
    type=click.Path(exists=False, dir_okay=False, file_okay=True),
    help="filename of output file",
    default=None,
)
def grid_baselines(configfile, direc, outfile):
    """Grid baselines according to CONFIGFILE."""
    obs = observation.Observation.from_yaml(configfile)

    if outfile is None:
        outfile = Path(direc) / (
            f"drift_blmin{obs.bl_min.value:.3f}_blmax{obs.bl_max.value:.3f}_"
            f"{obs.frequency.to('GHz').value:.3f}GHz_arrayfile.pkl"
        )
    elif not Path(outfile).is_absolute():
        outfile = Path(direc) / outfile

    with open(outfile, "wb") as fl:
        pickle.dump(obs, fl)

    logger.info(f"There are {len(obs.baseline_groups)} baseline types")
    logger.info(f"Saving array file as {outfile}")


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
    """Calculate the sensitivity of an array.

    This is the primary command of 21cmSense, and can be run independently for a
    complete sensitivity calculation.
    """
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
    logger.info(
        f"Used {len(sensitivity.k1d)} bins between "
        f"{sensitivity.k1d.min()} and {sensitivity.k1d.max()}"
    )
    sensitivity.write(filename=fname, thermal=thermal, sample=samplevar, prefix=prefix)

    if write_significance:
        sig = sensitivity.calculate_significance(thermal=thermal, sample=samplevar)
        logger.info(f"Significance of detection: {sig}")

    if plot and HAVE_MPL:
        fig = sensitivity.plot_sense_1d(thermal=thermal, sample=samplevar)
        if plot_title:
            plt.title(plot_title)
        prefix + "_" if prefix else ""
        fig.savefig(
            f"{prefix}{sensitivity.foreground_model}_"
            f"{sensitivity.observation.frequency:.3f}.png"
        )
