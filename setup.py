from setuptools import setup

from bitmix import __version__

setup(
    name='bitmix',
    version=__version__,

    url='https://github.com/iCAS-Lab/bitmix',
    author='Brendan Reidy',
    author_email='brendanreidy16@gmail.com',

    py_modules=['bitmix'],
)

install_requires=[
    'torch',
],