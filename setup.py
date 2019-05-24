#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import io
import re
from os.path import dirname
from os.path import join
from setuptools import find_packages
from setuptools import setup


def read(*names, **kwargs):
    return io.open(
        join(dirname(__file__), *names),
        encoding=kwargs.get('encoding', 'utf8')
    ).read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")



setup_args = dict(
    name='py21cmsense',
    version=find_version("py21cmsense", "__init__.py"),
    license=read("LICENSE.rst"),
    long_description='%s\n%s' % (
        re.compile('^.. start-badges.*^.. end-badges', re.M | re.S).sub('', read('README.rst')),
        re.sub(':[a-z]+:`~?(.*?)`', r'``\1``', read('CHANGELOG.rst'))
    ),
    author='Jonathan Pober',
    url='https://github.com/jpober/21cmSense',
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        # complete classifier list: http://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Development Status :: 5 - Production/Stable',
        'License :: OSI Approved :: GPL',
        'Operating System :: Unix',
        'Operating System :: POSIX',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    install_requires=[
        'numpy',
        'scipy',
        'future',
        'click',
        'tqdm',
        'pyyaml',
        'astropy'
    ],
    package_data={"py21cmsense":['data/*']},
    entry_points={
        'console_scripts': [
            'sense = py21cmsense.cli:main',
        ]
    },
)

if __name__ == "__main__":
    setup(*(), **setup_args)
