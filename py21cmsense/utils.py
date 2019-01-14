#! usr/bin/env python
from __future__ import print_function
from builtins import range
from scipy.interpolate import interp2d
import numpy as np, re

"""
Utilities for working with the output of 21cmSense CLI functions, i.e. 
reading/writing of data files.
"""


def load_noise_files(files, verbose=False, polyfit_deg=3, kmin=0.01, kmax=0.51):
    """
    Loads input 21cmsense files and clips nans and infs
    reduces k-range to [.01,.51] hMpc^-1
    returns freqs, ks, noise_values
    """
    if not files:
        raise ValueError("files must be a single string, or list of strings.")

    # wrap single files for iteration
    single_file = False
    if len(np.shape(files)) == 0:
        files = [files]
        single_file = True

    files.sort()
    re_f = re.compile("(\d+\.\d+)")  # glob should load them in increasing freq order

    noises = []
    noise_ks = []
    noise_freqs = []
    nk_grid = np.linspace(kmin, kmax)

    for noisefile in files:
        data = np.load(noisefile)
        noise = data["T_errs"]
        noise_k = data["ks"]

        bad = np.logical_or(np.isinf(noise), np.isnan(noise))
        noise = noise[np.logical_not(bad)]
        noise_k = noise_k[np.logical_not(bad)]

        # keep only the points that fall in our desired k range
        noise = noise[noise_k < nk_grid.max()]
        noise_k = noise_k[noise_k < nk_grid.max()]
        if verbose:
            print(noisefile, "Max k:", np.max(noise), end=" ")

        # adds values at k=0 to help interpolator function
        # noise approaches infinite at low K
        noise_k = np.insert(noise_k, 0, 0)
        noise = np.insert(noise, 0, np.min([1e3, np.median(noise)]))

        tmp = np.polyfit(noise_k, noise, polyfit_deg, full=True)
        noise = np.poly1d(tmp[0])(nk_grid)
        noises.append(noise)
        noise_ks.append(nk_grid)

        small_name = noisefile.split("/")[-1].split(".npz")[0].split("_")[-1]
        f = float(re_f.match(small_name).groups()[0]) * 1e3  # sensitivity freq in MHz
        if verbose:
            print(f)
        noise_freqs.append(f)

    if single_file:
        noise_freqs = np.squeeze(noise_freqs)
        noise_ks = np.squeeze(noise_ks)
        noises = np.squeeze(noises)
    return noise_freqs, noise_ks, noises


def noise_interp2d(
    noise_freqs,
    noise_ks,
    noises,
    interp_kind="linear",
    **kwargs
):
    """
    Builds 2d interpolator from loaded data, default interpolation: linear
    interpolator inputs k (in hMpci), freq (in MHz)n
    """

    noise_k_range = [np.min(np.concatenate(noise_ks)), np.max(np.concatenate(noise_ks))]

    if np.min(noise_k_range) == np.max(noise_k_range):
        raise ValueError("k range only contains one value")

    nk_count = np.mean([len(myks) for myks in noise_ks])
    nks = np.linspace(noise_k_range[0], noise_k_range[1], num=nk_count)
    noise_interp = np.array(
        [np.interp(nks, noise_ks[i], noises[i]) for i in range(len(noises))]
    )
    NK, NF = np.meshgrid(nks, noise_freqs)
    noise_interp = interp2d(NK, NF, noise_interp, kind=interp_kind, **kwargs)

    return noise_interp
