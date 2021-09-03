#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from __future__ import absolute_import, print_function

from setuptools import find_packages, setup

import io
import re
from os.path import dirname, join


def read(*names, **kwargs):
    return io.open(
        join(dirname(__file__), *names), encoding=kwargs.get("encoding", "utf8")
    ).read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


docs_req = ["sphinx>=1.3", "sphinx-rtd-theme", "numpydoc", "nbsphinx", "ipython"]
test_req = ["pre-commit", "pytest", "matplotlib"]
dev_rq = docs_req + test_req

setup_args = {
    "name": "py21cmsense",
    "data_files": [("", ["LICENSE.rst"]), ("", ["CHANGELOG.rst"])],
    "version": find_version("py21cmsense", "__init__.py"),
    "license": read("LICENSE.rst"),
    "long_description": "%s\n%s"
    % (
        re.compile("^.. start-badges.*^.. end-badges", re.M | re.S).sub(
            "", read("README.rst")
        ),
        re.sub(":[a-z]+:`~?(.*?)`", r"``\1``", read("CHANGELOG.rst")),
    ),
    "author": "Jonathan Pober",
    "url": "https://github.com/jpober/21cmSense",
    "packages": find_packages(),
    "include_package_data": True,
    "zip_safe": False,
    "classifiers": [
        # complete classifier list: http://pypi.python.org/pypi?%3Aaction=list_classifiers
        "Development Status :: 5 - Production/Stable",
        "License :: OSI Approved :: GPL",
        "Operating System :: Unix",
        "Operating System :: POSIX",
        "Programming Language :: Python",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    "install_requires": [
        "numpy",
        "scipy",
        "future",
        "click",
        "tqdm",
        "pyyaml",
        "astropy",
        "methodtools",
        "pyuvdata",
        "cached_property",
        "rich",
        "h5py",
        "attrs>=21.1.0",
    ],
    "extras_require": {"docs": docs_req, "test": test_req, "dev": dev_rq},
    "package_data": {"py21cmsense": ["data/*"]},
    "entry_points": {"console_scripts": ["sense = py21cmsense.cli:main"]},
}

if __name__ == "__main__":
    setup(*(), **setup_args)
