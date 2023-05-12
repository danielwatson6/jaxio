import os
import re
from setuptools import setup


with open(os.path.join(os.path.dirname(__file__), 'requirements.txt')) as f:
  requirements = re.sub(r'\n+', '\n', f.read()).strip().split('\n')

setup(
    name='jaxio',
    version='0.0.1',
    description='Input pipelines for JAX, in JAX',
    long_description='An attempt to do input pipelines purely relying on JAX, with support for jitting iterators.',
    author='Daniel Watson',
    packages=['jaxio'],
    install_requires=requirements,
    license_files=('LICENSE.txt',),
)
