"""Tests for `jaxio.src.dataset`."""

# TODO(danielwatson6): test peek
# TODO(danielwatson6): test shuffle
# TODO(danielwatson6): test sleep
# TODO(danielwatson6): test unbatch
# TODO(danielwatson6): test from_pytree_slices with actual pytree

from functools import partial
import time

import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import pytest

import jaxio


def test_constructor():
  data = range(10)

  expected = list(data)

  d = jaxio.Dataset(data)
  assert not d._is_jittable
  # Nit: don't make the assert line contain a generator expression (on failures,
  # pytest won't expand generators, but it will print arrays), easier to debug.
  result = list(d)
  assert result == expected


def test_from_pytree_slices():
  data = jnp.arange(10)

  d = jaxio.Dataset.from_pytree_slices(data)
  assert not d._is_jittable
  result = jnp.stack(list(d))
  assert jnp.all(result == data)


def test_as_jit_compatible():
  data = jnp.arange(10)

  d = jaxio.Dataset.from_pytree_slices(data)
  d = d.as_jit_compatible()
  assert d._is_jittable
  result = jnp.stack(list(d))
  assert jnp.all(result == data)


# TODO(danielwatson6): test batch axis.
# TODO(danielwatson6): test nondeterministic
def test_batch():
  n = 20
  b = 3

  expected = jnp.arange((n // b) * b).reshape(n // b, b)

  d = jaxio.Dataset(range(n))
  d = d.batch(b)
  assert not d._is_jittable
  d_list = list(d)
  # TODO(danielwatson6): re-write this test to preserve type?
  assert all(isinstance(x, np.ndarray) for x in d_list)
  assert jnp.all(jnp.stack(d_list) == expected)

  # jit-compatible batch
  d = jaxio.Dataset.from_pytree_slices(jnp.arange(n))
  d = d.as_jit_compatible().batch(b)
  assert d._is_jittable
  d_list = list(d)
  assert all(isinstance(x, jax.Array) for x in d_list)
  assert jnp.all(jnp.stack(d_list) == expected)

  # jit batch
  d = jaxio.Dataset.from_pytree_slices(jnp.arange(n))
  d = d.as_jit_compatible().batch(b).jit()
  result = jnp.stack(list(d))
  assert jnp.all(result == expected)


# TODO(danielwatson6): test jit
def test_enumerate():
  data = range(10)

  expected = list(enumerate(data))

  d = jaxio.Dataset(data)
  d = d.enumerate()
  assert not d._is_jittable
  result = list(d)
  assert result == expected


# TODO(danielwatson6): test jit
def test_filter():
  data = range(10)
  filter_fn = lambda x: x % 2 == 0

  expected = list(filter(filter_fn, data))

  d = jaxio.Dataset(data)
  d = d.filter(filter_fn)
  assert not d._is_jittable
  result = list(d)
  assert result == expected


def test_jit():
  n = 10

  d = jaxio.Dataset(range(n))
  with pytest.raises(ValueError):
    d.jit()

  d = jaxio.Dataset.from_pytree_slices(jnp.arange(n))
  d = d.as_jit_compatible().jit()
  assert d._is_jittable
  result = jnp.stack(list(d))
  assert jnp.all(result == jnp.arange(n))


def test_map():
  n = 10
  f = lambda x: x * 2

  expected = [f(x) for x in range(n)]

  d = jaxio.Dataset(range(n))
  d = d.map(f)
  assert not d._is_jittable
  result = list(d)
  assert result == expected

  # jit-compatible map
  d = jaxio.Dataset.from_pytree_slices(jnp.arange(n))
  d = d.as_jit_compatible().map(f)
  assert d._is_jittable
  result = jnp.stack(list(d))
  assert jnp.all(result == jnp.stack(expected))

  # jit map
  d = jaxio.Dataset.from_pytree_slices(jnp.arange(n))
  d = d.as_jit_compatible().map(f)
  assert d._is_jittable
  d = d.jit()
  result = jnp.stack(list(d))
  assert jnp.all(result == jnp.stack(expected))


# TODO(danielwatson6): maybe we should not do benchmarks on tests, slow and
# nondeterministic.
def test_prefetch():
  n = 100
  dataset_sleep_secs = 0.01
  # simulate costly computation using data. must be slower than next(d) for the
  # benefits of prefetching to materialize.
  costly_computation_sleep_secs = 0.05

  expected = list(range(n))

  slow_io_time = 0.0
  slow_result = []
  d = jaxio.Dataset(range(n)).sleep(dataset_sleep_secs)
  assert not d._is_jittable
  for _ in range(n):
    tic = time.time()
    slow_result.append(next(d))
    slow_io_time += time.time() - tic
    time.sleep(costly_computation_sleep_secs)
  assert slow_result == expected

  fast_io_time = 0.0
  fast_result = []
  d = jaxio.Dataset(range(n)).sleep(dataset_sleep_secs).prefetch()
  assert not d._is_jittable
  for _ in range(n):
    tic = time.time()
    fast_result.append(next(d))
    slow_io_time += time.time() - tic
    time.sleep(costly_computation_sleep_secs)
  assert fast_result == expected

  assert slow_io_time > fast_io_time * 10.0


# TODO(danielwatson6): test jit
# TODO(danielwatson6): test memory usage
def test_repeat():
  n = 3

  d = jaxio.Dataset(range(n))
  d = d.repeat()
  assert not d._is_jittable
  for _ in range(n):
    next(d)
  assert next(d) == 0
  for _ in range(n - 1):
    next(d)
  assert next(d) == 0

  d = jaxio.Dataset(range(n))
  d = d.repeat(2)
  for _ in range(n):
    next(d)
  assert next(d) == 0
  for _ in range(n - 1):
    next(d)
  with pytest.raises(StopIteration):
    next(d)


# TODO
# def test_shuffle():
#   seed = 42
#   n = 10
#   b = 3

#   rng = jrandom.PRNGKey(seed)
#   perms = [
#       jrandom.permutation(jrandom.fold_in(rng, i), jnp.arange(n))
#       for i in range(n // b)
#   ]
#   shuffle_with_perm_fn = partial(jnp.take, axis=axis)
#   jtu.tree_map(partial(gather_fn, indices=perm), el)


def test_transform():
  data = jnp.arange(10)

  transform = lambda next_fn: (lambda: next_fn() * 2)
  expected = jnp.stack([transform(lambda: x)() for x in list(data)])

  d = jaxio.Dataset(data)
  d = d.transform(transform)
  assert not d._is_jittable
  result = jnp.stack(list(d))
  assert jnp.all(result == expected)

  # jit-compatible transform
  d = jaxio.Dataset(data)
  d = d.as_jit_compatible().transform(transform)
  assert d._is_jittable
  result = jnp.stack(list(d))
  assert jnp.all(result == expected)

  # jit transform
  d = jaxio.Dataset(data)
  d = d.as_jit_compatible().transform(transform)
  assert d._is_jittable
  d = d.jit()
  result = jnp.stack(list(d))
  assert jnp.all(result == expected)
