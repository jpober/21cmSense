from distutils.core import setup

__version__ = '0.0.1'

setup_args = {
    'name': 'py21cmsense',
    'package_dir': {'py21cmsense': 'py21cmsense'},
    'packages': ['py21cmsense'],
    'py_modules': ['py21cmsense.utils'],
    'version': __version__,
    'package_data': {'py21cmsense': ['tests/test_data/*.npz', 'tests/*.py']},
}

if __name__ == '__main__':
    apply(setup, (), setup_args)
