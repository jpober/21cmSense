#! /usr/bin/env python
"""
Creates an array file for use by calc_sense.py.  The main product is the uv coverage produced by the array during the
time it takes the sky to drift through the primary beam; other array parameters are also saved.
Array specific information comes from an aipy cal file.  If track is set, produces the uv coverage
for the length specified instead of that set by the primary beam.
"""
from __future__ import division
from __future__ import print_function

import os

import click
import numpy as np

from . import calc_sense as cs
from . import gridding

main = click.Group()


@main.command()
@click.argument(
    'array-config',
    type=click.Path(exists=True, dir_okay=False),
)
@click.option(
    '--track', type=float, default=None,
    help="If set, calculate sensitivity for a tracked observation of this duration in hours; otherwise, calculate for a drift scan.",
)
@click.option(
    '--bl-min',
    default=0.0, type=float,
    help="Set the minimum baseline (in meters) to include in the uv plane.",
)
@click.option(
    "--bl_max", default=None, type=float,
    help="Set the maximum baseline (in meters) to include in the uv plane.  Use to exclude outriggers with little EoR sensitivity to speed up calculation.",
)
@click.option(
    "-f", "--freq",
    default=0.135, type=float,
    help="The central frequency of the observation in GHz.  Default is 0.135 GHz, corresponding to z = 9.5, which matches the default model power spectrum used in calc_sense.py.",
)
@click.option(
    '--direc', type=click.Path(exists=True, dir_okay=True, file_okay=False),
    help="directory to save output file", default='.'
)
def mk_array_fl(array_config, track, bl_min, bl_max, freq, direc):
    t_int = 60.0  # hardcoded...
    # ==============================READ ARRAY PARAMETERS=========================

    (bl_len_max, bl_len_min, dish_size_in_lambda, obs_duration,
     prms, quadsum, uvsum) = gridding.get_array_specs(
        bl_max=bl_max, bl_min=bl_min, calfile=array_config, freq=freq, track=track, t_int=t_int, report=True
    )

    if track:
        name = prms["name"] + "track_%.1fhr" % track
    else:
        name = prms["name"] + "drift"

    filepath = os.path.join(
        direc,
        "%s_blmin%0.f_blmax%0.f_%.3fGHz_arrayfile.npz" % (name, bl_len_min, bl_len_max, freq)
    )
    print("Saving array file as {}".format(filepath))

    np.savez(
        filepath,
        uv_coverage=uvsum,
        uv_coverage_pess=quadsum,
        name=name,
        obs_duration=obs_duration,
        dish_size_in_lambda=dish_size_in_lambda,
        Trx=prms["Trx"],
        t_int=t_int,
        freq=freq,
    )


@main.command()
@click.argument(
    "array-file",
    type=click.Path(exists=True, dir_okay=False),
)
@click.option(
    "-m",
    "--model",
    default="moderate",
    type=click.Choice(['pessimistic', 'moderate', 'optimistic']),
    help="""
    The model of the foreground wedge to use.  Three options are 'pess' (all k 
    modes inside horizon + buffer are excluded, and all baselines are added 
    incoherently), 'mod' (all k modes inside horizon + buffer are excluded, but 
    all baselines within a uv pixel are added coherently), and 'opt' (all modes 
    k modes inside the primary field of view are excluded).  See Pober et al. 
    2014 for more details.
    """,
)
@click.option(
    "--buffer",
    default=0.1,
    type=float,
    help="""
    The size of the additive buffer outside the horizon to exclude in the 
    pessimistic and moderate models.
    """,
)
@click.option(
    "--eor-ps-file",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="""The model epoch of reionization power spectrum.  The code is built 
    to handle output power spectra from 21cmFAST.
    """,
)
@click.option(
    "--n-days",
    default=180.0,
    type=float,
    help="""
    The total number of days observed.  The default is 180, which is the maximum 
    a particular R.A. can be observed in one year if one only observes at night.  
    The total observing time is ndays*n_per_day.
    """,
)
@click.option(
    "--n-per-day",
    default=6.0,
    type=float,
    help="""The number of good observing hours per day.  This corresponds to the 
    size of a low-foreground region in right ascension for a drift scanning 
    instrument.  The total observing time is ndays*n_per_day.  Default is 6.  
    If simulating a tracked scan, n_per_day should be a multiple of the length 
    of the track (i.e. for two three-hour tracks per day, n_per_day should be 6).
    """,
)
@click.option(
    "-b",
    "--bandwidth",
    default=0.008,
    type=float,
    help="""
    Cosmological bandwidth in GHz.  Note this is not the total instrument 
    bandwidth, but the redshift range that can be considered co-eval.  
    Default is 0.008 (8 MHz).
    """,
)
@click.option(
    "--n-chan",
    default=82,
    type=int,
    help="""
    Integer number of channels across cosmological bandwidth.  Defaults to 82, 
    which is equivalent to 1024 channels over 100 MHz of bandwidth.  
    Sets maximum k_parallel that can be probed, but little to no overall effect 
    on sensitivity.
    """,
)
@click.option(
    "--ns/--no-ns",
    default=True,
    help="""
    Remove pure north/south baselines (u=0) from the sensitivity calculation.  
    These baselines can potentially have higher systematics, so excluding them 
    represents a conservative choice.
    """,
)
@click.option(
    "--hubble",
    default=0.7,
    type=float,
    help="Hubble parameter."
)
@click.option(
    '--direc', type=click.Path(exists=True, dir_okay=True, file_okay=False),
    help="directory to save output file", default='.'
)
@click.option(
    '-w',
    '--write-significance/--no-significance',
    default=True,
    help="whether to write the significance of the PS to screen"
)
def calc_sense(array_file, model, buffer, eor_ps_file, n_days, n_per_day, bandwidth, n_chan, ns, hubble, direc,
               write_significance):
    # Load in data from array file; see mk_array_file.py for definitions of the parameters
    array = np.load(array_file)
    name = array["name"]
    obs_duration = array["obs_duration"]
    dish_size_in_lambda = array["dish_size_in_lambda"]
    Trx = float(array["Trx"])
    t_int = array["t_int"]
    if model == "pessimistic":
        uv_coverage = array["uv_coverage_pess"]
    else:
        uv_coverage = array["uv_coverage"]

    if eor_ps_file is None:
        eor_ps_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/ps_no_halos_nf0.521457_z9.50_useTs0_zetaX-1.0e+00_200_400Mpc_v2")
    eor_ps = cs.get_eor_ps(eor_ps_file, h=hubble)

    kpls, sense, Tsense = cs.calculate_sensitivity_2d(
        uv_coverage=uv_coverage,
        freq=array['freq'],
        p21=eor_ps,
        n_per_day=n_per_day,
        obs_duration=obs_duration,
        dish_size_in_lambda=dish_size_in_lambda,
        t_int=t_int,
        Trx=Trx,
        n_channels=n_chan,
        bandwidth=bandwidth,
        n_days=n_days,
        horizon_buffer=buffer,
        foreground_model=model,
        no_ns_baselines=not ns,
        report=True
    )

    kmag, sense1d, Tsense1d = cs.average_sensitivity_to_1d(
        sense, Tsense,
        maxk=eor_ps.x.max(),
        bandwidth=bandwidth,
        freq=array['freq'],
        kpls=kpls,
        report=True
    )

    # save results to output npz
    np.savez(
        os.path.join(
            direc,
            "{name}_{model}_{freq:.3f}.npz".format(name=name, model=model, freq=array["freq"])
        ),
        ks=kmag,
        errs=sense1d,
        T_errs=Tsense1d,
    )

    if write_significance:
        sig = cs.calculate_significance(sense1d, eor_ps, kmag)
        print("Significance of detection: ", sig)
