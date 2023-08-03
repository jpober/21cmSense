import pytest

import numpy as np

from py21cmsense.theory import EOS2021, Legacy21cmFAST


def test_eos_extrapolation():
    eos = EOS2021()

    with pytest.warns(
        UserWarning, match="Extrapolating above the simulated theoretical k"
    ):
        eos.delta_squared(15, np.array([0.1, 1e6]))

    with pytest.warns(
        UserWarning, match="Extrapolating below the simulated theoretical k"
    ):
        eos.delta_squared(15, np.array([0.0001, 0.1]))

    with pytest.warns(
        UserWarning, match="Extrapolating beyond simulated redshift range"
    ):
        eos.delta_squared(50, np.array([0.1]))


def test_legacy():
    theory = Legacy21cmFAST()
    assert theory.delta_squared(9.1, 1.0) == theory.delta_squared(9.9, 1.0)

    with pytest.warns(UserWarning, match="Theory power corresponds to z=9.5, not z"):
        theory.delta_squared(1.0, 1.0)

    with pytest.warns(
        UserWarning, match="Extrapolating above the simulated theoretical k"
    ):
        theory.delta_squared(9.5, np.array([0.1, 1e6]))

    with pytest.warns(
        UserWarning, match="Extrapolating below the simulated theoretical k"
    ):
        theory.delta_squared(9.5, np.array([0.0001, 0.1]))
