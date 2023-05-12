# jaxio

An attempt to do input pipelines purely relying on JAX, with support for jitting iterators. Very very heavily inspired by the `tf.data.Dataset` API, since this is
what most jax users currently use.

## Quick example

```python
import jax.numpy as jnp
import jaxio

d = jaxio.Dataset.from_pytree_slices(jnp.arange(10))
d = d.as_jit_compatible()  # -- jit boundary --
d = d.batch(3)             # <-- will be jitted
d = d.map(jnp.square)      # <-- will be jitted
d = d.map(lambda x: -x)    # <-- will be jitted
d = d.jit()                # -- jit boundary --
d = d.prefetch(1)

for el in d:
  print(el)

# [ 0 -1 -4]
# [ -9 -16 -25]
# [-36 -49 -64]
```

## Installation

```bash
pip install jaxio
```

## Development

**These instructions are only intended for those interested in contributing to jaxio directly.**

One-time setup:

```bash
virtualenv .venv
virtualenv .venv-docs
source .venv-docs/bin/activate
pip install --upgrade pip
pip install -e .
pip install -r docs/requirements.txt
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
pip install pytest
```

To test the package locally:

```bash
pytest
```

To re-generate the documentation pages:

```bash
source .venv-docs/bin/activate
pip install -e .
rm -rf docs/_build
cd docs && make html && cd -
source .venv/bin/activate
```
