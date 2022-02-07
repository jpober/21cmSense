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


@pytest.fixture(scope="module")
def sensitivity_config_defined_p21(tmpdirec, observation_config, sensitivity_config):
    with open(sensitivity_config) as fl:
        sensitivity = yaml.load(fl)

    pfile = path.join(
        example_configs,
        "../py21cmsense/data/ps_no_halos_nf0.521457_z9.50_useTs0_zetaX-1.0e+00_200_400Mpc_v2",
    )

    sensitivity["observation"] = observation_config

    with open(path.join(tmpdirec, "sensitivity_with_p21.yml"), "w") as fl:
        dump(sensitivity, fl)

        fl.write(f"p21: !txt {pfile}\n")

    return path.join(tmpdirec, "sensitivity_with_p21.yml")


def test_gridding_baselines(runner, observation_config):

    output = runner.invoke(cli.main, ["grid-baselines", observation_config])
    if output.exception:
        traceback.print_exception(*output.exc_info)

    assert output.exit_code == 0


def test_calc_sense(runner, sensitivity_config):
    output = runner.invoke(cli.main, ["calc-sense", sensitivity_config])
    if output.exception:
        traceback.print_exception(*output.exc_info)

    assert output.exit_code == 0


def test_calc_sense_with_p21(runner, sensitivity_config_defined_p21):
    output = runner.invoke(cli.main, ["calc-sense", sensitivity_config_defined_p21])
    if output.exception:
        traceback.print_exception(*output.exc_info)

    assert output.exit_code == 0

    # ensure a plot was created
    assert glob.glob("*.png")


def test_both(runner, tmpdirec, observation_config, sensitivity_config):
    output = runner.invoke(
        cli.main,
        [
            "grid-baselines",
            observation_config,
            "--direc",
            tmpdirec,
            "--outfile",
            "arrayfile.pkl",
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
            path.join(tmpdirec, "arrayfile.pkl"),
            "--plot-title",
            "MYTITLE",
        ],
    )
    if output.exception:
        traceback.print_exception(*output.exc_info)

    assert output.exit_code == 0
