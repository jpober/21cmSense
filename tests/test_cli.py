import pytest

import glob
import traceback
from astropy.io.misc import yaml
from click.testing import CliRunner
from os import path
from yaml import dump

from py21cmsense import cli

here = path.dirname(path.abspath(__file__))
example_configs = path.join(here, "../example_configs")


@pytest.fixture(scope="module")
def runner():
    return CliRunner()


@pytest.fixture(scope="module")
def observatory_config() -> str:
    return path.join(example_configs, "observatory_hera.yml")


@pytest.fixture(scope="module")
def observation_config(tmpdirec, observatory_config):
    with open(path.join(example_configs, "observation_hera.yml")) as fl:
        observation = yaml.load(fl)

    observation["observatory"] = observatory_config

    with open(path.join(tmpdirec, "observation.yml"), "w") as fl:
        yaml.dump(observation, fl)

    return path.join(tmpdirec, "observation.yml")


@pytest.fixture(scope="module")
def sensitivity_config(tmpdirec, observation_config):
    with open(path.join(example_configs, "sensitivity_hera.yml")) as fl:
        sensitivity = yaml.load(fl)

    sensitivity["observation"] = observation_config

    with open(path.join(tmpdirec, "sensitivity.yml"), "w") as fl:
        yaml.dump(sensitivity, fl)

    return path.join(tmpdirec, "sensitivity.yml")


def test_gridding_baselines(runner, observation_config, tmpdirec):
    output = runner.invoke(
        cli.main, ["grid-baselines", observation_config, "--direc", str(tmpdirec)]
    )
    if output.exception:
        traceback.print_exception(*output.exc_info)

    assert output.exit_code == 0


def test_calc_sense(runner, sensitivity_config, tmpdirec):
    output = runner.invoke(
        cli.main, ["calc-sense", sensitivity_config, "--direc", str(tmpdirec)]
    )
    if output.exception:
        traceback.print_exception(*output.exc_info)

    assert output.exit_code == 0


def test_both(runner, tmpdirec, observation_config, sensitivity_config):
    output = runner.invoke(
        cli.main,
        [
            "grid-baselines",
            observation_config,
            "--direc",
            tmpdirec,
            "--outfile",
            "arrayfile.h5",
        ],
    )

    if output.exception:
        traceback.print_exception(*output.exc_info)

    output = runner.invoke(
        cli.main,
        [
            "calc-sense",
            sensitivity_config,
            "--array-file",
            path.join(tmpdirec, "arrayfile.h5"),
            "--plot-title",
            "MYTITLE",
        ],
    )
    if output.exception:
        traceback.print_exception(*output.exc_info)

    assert output.exit_code == 0
