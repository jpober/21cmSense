import hickle
import numpy as np
from astropy import units as un

from py21cmsense import GaussianBeam, Observation, Observatory

beam = GaussianBeam(frequency=150 * un.MHz, dish_size=14 * un.m)
obs = Observatory(beam=beam, antpos=np.random.random((25, 3)) * 30 * un.m)
observation = Observation(observatory=obs)


def test_beam(tmpdirec):
    hickle.dump(beam, tmpdirec / "beam.h5")
    new = hickle.load(tmpdirec / "beam.h5")

    assert new == beam


def test_observatory(tmpdirec):
    hickle.dump(obs, tmpdirec / "observatory.h5")
    new = hickle.load(tmpdirec / "observatory.h5")
    assert new == obs


def test_observation(tmpdirec):
    hickle.dump(observation, tmpdirec / "observation.h5")
    new = hickle.load(tmpdirec / "observation.h5")
    assert new == observation
