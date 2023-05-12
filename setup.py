import os
import re
from setuptools import setup


with open(os.path.join(os.path.dirname(__file__), 'requirements.txt')) as f:
  requirements = re.sub(r'\n+', '\n', f.read()).strip().split('\n')

setup(
  name='datax',
  version='0.0.1',
  description='Input pipelines for JAX, in JAX',
  author='Daniel Watson',
  packages=['datax'],
  install_requires=requirements,
)
