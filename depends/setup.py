from setuptools import setup
from setuptools import find_packages
from subprocess import STDOUT, check_call
import os

#install libhdf5-dev
check_call(['apt-get', 'install', '-y', 'libhdf5-dev'], stdout=open(os.devnull,'wb'), stderr=STDOUT)
setup(name='my_depends',
      version='0.0.1',
      install_requires=['keras', 'h5py'],
      packages=find_packages())