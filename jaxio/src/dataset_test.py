import time

import jax.numpy as jnp
import pytest

import jaxio


def test_constructor():
  data = range(10)

  d = jaxio.Dataset(data)
  assert not d._is_jittable
  assert list(d) == list(data)


def test_from_pytree_slices():
  data = jnp.arange(10)

  d = jaxio.Dataset.from_pytree_slices(data)
  assert not d._is_jittable
  assert jnp.all(jnp.stack(list(d)) == data)


def test_as_jit_compatible():
  data = jnp.arange(10)

  d = jaxio.Dataset.from_pytree_slices(data)
  d = d.as_jit_compatible()
  assert d._is_jittable
  assert jnp.all(jnp.stack(list(d)) == data)


# TODO(danielwatson6): test batch axis.
def test_batch():
  n = 20
  b = 3

  expected = jnp.arange((n // b) * b).reshape(n // b, b)

  d = jaxio.Dataset(range(n))
  d = d.batch(b)
  assert not d._is_jittable
  d_list = list(d)
  # (non-jit) batch calls stack, so we expect a list of jax arrays.
  assert all(isinstance(x, jnp.ndarray) for x in d_list)
  assert jnp.all(jnp.stack(d_list) == expected)

  # jit-compatible batch
  d = jaxio.Dataset.from_pytree_slices(jnp.arange(n))
  d = d.as_jit_compatible().batch(b)
  assert d._is_jittable
  assert jnp.all(jnp.stack(list(d)) == expected)

  # jit batch
  d = jaxio.Dataset.from_pytree_slices(jnp.arange(n))
  d = d.as_jit_compatible().batch(b).jit()
  assert jnp.all(jnp.stack(list(d)) == expected)


def test_enumerate():
  data = range(10)

  d = jaxio.Dataset(data)
  d = d.enumerate()
  assert not d._is_jittable
  assert list(d) == list(enumerate(data))


def test_filter():
  data = range(10)
  filter_fn = lambda x: x % 2 == 0

  d = jaxio.Dataset(data)
  d = d.filter(filter_fn)
  assert not d._is_jittable
  assert list(d) == list(filter(filter_fn, data))


def test_fmap():
  data = jnp.arange(10)

  transform = lambda next_fn: (lambda: next_fn() * 2)
  expected = jnp.stack([transform(lambda: x)() for x in list(data)])

  d = jaxio.Dataset(data)
  d = d.fmap(transform)
  assert not d._is_jittable
  assert jnp.all(jnp.stack(list(d)) == expected)

  # jit-compatible fmap
  d = jaxio.Dataset(data)
  d = d.as_jit_compatible().fmap(transform)
  assert d._is_jittable
  assert jnp.all(jnp.stack(list(d)) == expected)

  # jit fmap
  d = jaxio.Dataset(data)
  d = d.as_jit_compatible().fmap(transform)
  assert d._is_jittable
  d = d.jit()
  assert jnp.all(jnp.stack(list(d)) == expected)


def test_jit():
  n = 10

  d = jaxio.Dataset(range(n))
  with pytest.raises(ValueError):
    d.jit()

  d = jaxio.Dataset.from_pytree_slices(jnp.arange(n))
  d = d.as_jit_compatible().jit()
  assert d._is_jittable
  assert jnp.all(jnp.stack(list(d)) == jnp.arange(n))


def test_map():
  n = 10
  f = lambda x: x * 2

  expected = [f(x) for x in range(n)]

  d = jaxio.Dataset(range(n))
  d = d.map(f)
  assert not d._is_jittable
  assert list(d) == expected

  # jit-compatible map
  d = jaxio.Dataset.from_pytree_slices(jnp.arange(n))
  d = d.as_jit_compatible().map(f)
  assert d._is_jittable
  assert jnp.all(jnp.stack(list(d)) == jnp.stack(expected))

  # jit map
  d = jaxio.Dataset.from_pytree_slices(jnp.arange(n))
  d = d.as_jit_compatible().map(f)
  assert d._is_jittable
  d = d.jit()
  assert jnp.all(jnp.stack(list(d)) == jnp.stack(expected))


def test_prefetch():
  n = 100
  dataset_sleep_secs = 0.01
  # simulate costly computation using data. must be slower than next(d) for the
  # benefits of prefetching to materialize.
  costly_computation_sleep_secs = 0.05

  slow_io_time = 0.0
  slow_result = []
  d = jaxio.Dataset(range(n)).sleep(dataset_sleep_secs)
  assert not d._is_jittable
  for _ in range(n):
    tic = time.time()
    slow_result.append(next(d))
    slow_io_time += time.time() - tic
    time.sleep(costly_computation_sleep_secs)
  assert slow_result == list(range(n))

  fast_io_time = 0.0
  fast_result = []
  d = jaxio.Dataset(range(n)).sleep(dataset_sleep_secs).prefetch()
  assert not d._is_jittable
  for _ in range(n):
    tic = time.time()
    fast_result.append(next(d))
    slow_io_time += time.time() - tic
    time.sleep(costly_computation_sleep_secs)
  assert fast_result == list(range(n))

  assert slow_io_time > fast_io_time * 10.0


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
