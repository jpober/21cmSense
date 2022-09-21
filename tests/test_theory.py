import pytest

import numpy as np

from py21cmsense.theory import EOS2021


def test_extrapolation():
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
