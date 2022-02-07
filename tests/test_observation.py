import pytest

import copy
import numpy as np
import pickle
from astropy import units

from py21cmsense import GaussianBeam, Observation, Observatory


@pytest.fixture(scope="module")
def bm():
    return GaussianBeam(150.0 * units.MHz, dish_size=14 * units.m)


@pytest.fixture(scope="module")
def observatory(bm):
    return Observatory(
        antpos=np.array([[0, 0, 0], [14, 0, 0], [28, 0, 0], [70, 0, 0]]) * units.m,
        beam=bm,
    )


def test_units(observatory):
    obs = Observation(observatory=observatory)

    assert obs.time_per_day.unit == units.hour
    assert obs.obs_duration.to("min").unit == units.min
    assert obs.integration_time.to("s").unit == units.s
    assert obs.bandwidth.to("MHz").unit == units.MHz
    assert obs.bl_min.to("m").unit == units.m
    assert obs.bl_max.to("m").unit == units.m
    assert obs.tsky_amplitude.to("mK").unit == units.mK
    assert obs.tsky_ref_freq.to("MHz").unit == units.MHz

    assert obs.frequency == observatory.frequency
    assert obs.n_lst_bins > 1
    assert obs.Tsky.to("mK").unit == units.mK
    assert obs.Tsys.to("mK").unit == units.mK
    assert obs.Trms.to("mK").unit == units.mK
    assert 6 < obs.redshift < 12
    assert obs.kparallel.unit == units.littleh / units.Mpc
    assert obs.total_integration_time.to("s").unit == units.s
    assert len(obs.ugrid_edges) == len(obs.ugrid) + 1
    assert obs.clone() == obs


def test_pickle(observatory):
    obs = Observation(observatory=observatory)

    string_rep = pickle.dumps(obs)
    obs2 = pickle.loads(string_rep)
    assert obs == obs2


def test_uvcov(observatory):
    coherent_obs = Observation(observatory=observatory, coherent=True)

    incoherent_obs = Observation(observatory=observatory, coherent=False)

    assert np.all(coherent_obs.uv_coverage >= incoherent_obs.uv_coverage)


def test_equality(observatory):
    new_observatory = copy.deepcopy(observatory)
    assert new_observatory == observatory
