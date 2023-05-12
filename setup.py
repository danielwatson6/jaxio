from setuptools import setup


setup(
    name='jaxio',
    version='0.0.2',
    description='Input pipelines for JAX, in JAX',
    long_description='An attempt to do input pipelines purely relying on JAX, with support for jitting iterators.',
    author='Daniel Watson',
    packages=['jaxio'],
    install_requires=['jax', 'jaxlib'],
    url='https://github.com/danielwatson6/jaxio',
    license_files=('LICENSE.txt',),
)
