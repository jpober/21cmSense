"""
Gridding baselines onto UV plane.
"""
from __future__ import division
from __future__ import print_function

from builtins import range

import aipy
import numpy as np
import yaml

from . import antpos
from . import array_definition as arrdef
import tqdm

def beamgridder(xcen, ycen, size):
    cen = size // 2 - 0.5  # correction for centering
    xcen += cen
    ycen = -1 * ycen + cen
    beam = np.zeros((size, size))

    if round(ycen) > size - 1 or round(xcen) > size - 1 or ycen < 0.0 or xcen < 0.0:
        return beam
    else:
        beam[int(round(ycen)), int(round(xcen))] = 1.0  # single pixel gridder
        return beam


def get_redundant_baselines(aa, bl_max, bl_min, obs_zen, ref_fq, report=False):
    uvbins = {}
    nants = len(aa)
    cnt = 0

    # find redundant baselines
    bl_len_min = bl_min / (aipy.const.c / (ref_fq * 1e11))  # converts meters to lambda
    bl_len_max = 0.0
    for i in tqdm.tqdm(range(nants), desc="finding redundancies", unit='ants', disable=not report):
        for j in range(nants):
            if i == j:
                continue  # no autocorrelations
            u, v, w = aa.gen_uvw(i, j, src=obs_zen)
            bl_len = np.sqrt(u ** 2 + v ** 2)
            if bl_len > bl_len_max:
                bl_len_max = bl_len
            if bl_len < bl_len_min:
                continue
            uvbin = "%.1f,%.1f" % (u, v)
            cnt += 1
            if uvbin not in uvbins:
                uvbins[uvbin] = ["%i,%i" % (i, j)]
            else:
                uvbins[uvbin].append("%i,%i" % (i, j))
    if report:
        print("There are %i baseline types" % len(list(uvbins.keys())))
        print(
            "The longest baseline is %.2f meters"
            % (bl_len_max * (aipy.const.c / (ref_fq * 1e11)))
        )  # 1e11 converts from GHz to cm

    if bl_max:
        # units of wavelength
        bl_len_max = bl_max / (aipy.const.c / (ref_fq * 1e11))

        if report:
            print(
                "The longest baseline being included is %.2f m"
                % (bl_len_max * (aipy.const.c / (ref_fq * 1e11)))
            )

    return bl_len_max, bl_len_min, uvbins


def grid_baselines(aa, bl_len_max, dish_size_in_lambda, obs_zen, times, uvbins, report=False):
    # grid each baseline type into uv plane
    # round to nearest odd
    dim = int(np.round(bl_len_max / dish_size_in_lambda) * 2 + 1)
    uvsum, quadsum = (
        np.zeros((dim, dim)),
        np.zeros((dim, dim)),
    )  # quadsum adds all non-instantaneously-redundant baselines incoherently

    for cnt, uvbin in enumerate(
            tqdm.tqdm(uvbins, desc="gridding baselines", unit='uv-bins', disable=not report)
    ):
        uvplane = np.zeros((dim, dim))
        for t in times:
            aa.set_jultime(t)
            obs_zen.compute(aa)
            bl = uvbins[uvbin][0]
            nbls = len(uvbins[uvbin])
            i, j = bl.split(",")
            i, j = int(i), int(j)
            u, v, w = aa.gen_uvw(i, j, src=obs_zen)
            _beam = beamgridder(
                xcen=u[0,0] / dish_size_in_lambda,
                ycen=v[0,0] / dish_size_in_lambda,
                size=dim,
            )

            uvplane += nbls * _beam
            uvsum += nbls * _beam
        quadsum += (uvplane) ** 2
    quadsum = quadsum ** 0.5

    return quadsum, uvsum


def read_config(config_file):
    # load cal file
    with open(config_file) as fl:
        prms = yaml.load(fl)

    # Get antenna pos
    if isinstance(prms['antpos'], str):
        # could be either a function name, or a filename.
        try:
            pos = np.load(prms['antpos'])
        except FileNotFoundError:
            try:
                fnc = getattr(antpos, prms['antpos'])
            except AttributeError:
                raise AttributeError("antenna position generator function {} does not exist!".format(fnc.__name__))

            try:
                pos = fnc(**prms.get("antpos_kw", {}))
            except:
                print("antpos_kw was poorly defined for the given antpos function")
                raise

    else:
        # must be a list of actual positions
        pos = prms['antpos']
        assert hasattr(pos, "__len__")
        assert all([len(x) == 3 for x in pos])

    prms['antpos'] = pos

    # hardcode this for now.
    prms['beam'] = aipy.fit.Beam2DGaussian

    # Better default for loc
    loc = prms['loc']
    loc = loc.split("(")[-1].split(")")[0].split(",")

    if len(loc) == 2:
        loc = loc + [0]
    else:
        loc[-1] = float(loc[-1])

    prms['loc'] = loc

    return prms


def get_array_specs(calfile, bl_max, bl_min, freq, t_int=60.0, track=None, report=False):
    prms = read_config(calfile)
    aa = arrdef.get_aa(np.array([0.150]), prms)
    prms = aa.get_arr_params()

    if track:
        obs_duration = 60.0 * track
    else:
        # scales observing time linearly with frequency to account for change in beam FWHM
        obs_duration = prms["obs_duration"] * (0.15 / freq)

    dish_size_in_lambda = prms["dish_size_in_lambda"]

    # ==========================FIDUCIAL OBSERVATION PARAMETERS===================
    # while poor form to hard code these to arbitrary values, they have very little effect on the end result
    # observing time
    cen_jd = 2454600.90911
    start_jd = cen_jd - (1.0 / 24) * (obs_duration / t_int / 2)
    end_jd = cen_jd + (1.0 / 24) * (obs_duration - 1) / t_int / 2
    times = np.arange(start_jd, end_jd, (1.0 / 24 / t_int))
    if report:
        print("Observation duration:", start_jd, end_jd)

    ref_fq = 0.150  # hardcoded... bad?

    # ================================MAIN CODE===================================
    aa.set_jultime(cen_jd)
    obs_lst = aa.sidereal_time()
    obs_zen = aipy.phs.RadioFixedBody(obs_lst, aa.lat)
    obs_zen.compute(aa)  # observation is phased to zenith of the center time of the drift

    bl_len_max, bl_len_min, uvbins = get_redundant_baselines(aa, bl_max, bl_min, obs_zen, ref_fq, report=report)
    quadsum, uvsum = grid_baselines(aa, bl_len_max, dish_size_in_lambda, obs_zen, times, uvbins, report=report)

    return bl_len_max, bl_len_min, dish_size_in_lambda, obs_duration, prms, quadsum, uvsum
