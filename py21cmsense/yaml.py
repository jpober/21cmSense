"""Module defining new YAML tags for py21cmsense."""
import inspect
import numpy as np
import pickle
import yaml
from astropy.io.misc.yaml import AstropyLoader
from functools import wraps

_DATA_LOADERS = {}


class LoadError(IOError):
    """Error raised on trying to load data from YAML files."""

    pass


def data_loader(tag=None):
    """A decorator that turns a function into a YAML tag for loading external datafiles."""

    def inner(fnc):
        _DATA_LOADERS[fnc.__name__] = fnc

        new_tag = tag or fnc.__name__.split("_loader")[0]
        fnc.tag = new_tag

        # ensure it only takes path to data.
        assert len(inspect.signature(fnc).parameters) == 1

        @wraps(fnc)
        def wrapper(data):
            try:
                return fnc(data)
            except OSError:
                raise
            except Exception as e:
                raise LoadError(str(e))

        def yaml_fnc(loader, node):
            return wrapper(node.value)

        yaml.add_constructor(f"!{new_tag}", yaml_fnc, Loader=yaml.FullLoader)
        yaml.add_constructor(f"!{new_tag}", yaml_fnc, Loader=yaml.Loader)
        yaml.add_constructor(f"!{new_tag}", yaml_fnc, Loader=AstropyLoader)

        return wrapper

    return inner


@data_loader("pkl")
def pickle_loader(data):
    """YAML tag for loading pickle files."""
    with open(data, "rb") as f:
        data = pickle.load(f)
    return data


@data_loader()
def npz_loader(data):
    """YAML tag for loading npz files."""
    return dict(np.load(data))


@data_loader()
def npy_loader(data):
    """YAML tag for loading npy files."""
    return np.load(data)


@data_loader()
def txt_loader(data):
    """YAML tag for loading ASCII files."""
    return np.genfromtxt(data)


def yaml_func(tag=None):
    """A decorator that turns a function into a YAML tag."""

    def inner(fnc):
        new_tag = tag or fnc.__name__
        fnc.tag = new_tag

        def yaml_fnc(loader, node):
            kwargs = loader.construct_mapping(node)
            return fnc(**kwargs)

        yaml.add_constructor(f"!{new_tag}", yaml_fnc, Loader=yaml.FullLoader)
        yaml.add_constructor(f"!{new_tag}", yaml_fnc, Loader=yaml.Loader)
        yaml.add_constructor(f"!{new_tag}", yaml_fnc, Loader=AstropyLoader)

        return fnc

    return inner
