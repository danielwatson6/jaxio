from typing import Any, Callable, Sequence

import concurrent.futures
from functools import partial
import itertools
import logging
import queue
import time

import jax
import jax.tree_util as jtu
from jaxlib import xla_extension


PyTree = Any
NextFn = Callable[[], PyTree]


def tree_starmap(
    f: Callable[[Sequence[PyTree]], PyTree], xs: Sequence[jnp.ndarray]
) -> jnp.ndarray:
  """Tree map a sequence, avoids the sequence being treated as one pytree."""
  return jtu.tree_map(lambda *xs: f(xs), *xs)


def unstack(x: jnp.ndarray, axis: int = 0) -> jnp.ndarray:
  """Identical to `tf.unstack`."""
  # This is faster than the nicer `tuple(jnp.moveaxis(x, axis, 0))` because it
  # avoids an XLA transpose.
  return jtu.tree_map(
      partial(jnp.squeeze, axis=axis), jnp.split(x, x.shape[axis], axis=axis)
  )


class Dataset:

  def __init__(self, it: PyTree):
    self._it = iter(it)
    self._is_jittable = False

  def __iter__(self):
    return self

  # TODO(danielwatson6): what happens when we iterate over a previous dataset?
  # Do we need to tee the iterator?
  def __next__(self):
    while True:
      try:
        return next(self._it)
      except (StopIteration, xla_extension.XlaRuntimeError, RuntimeError) as e:
        logging.info('Dataset::__next__: stopping')
        logging.debug('Dataset::__next__: caught error: %r', e)
        raise StopIteration

  @classmethod
  def from_next_fn(cls, next_fn: NextFn) -> 'Dataset':
    return cls(next_fn() for _ in iter(int, 1))

  @classmethod
  def from_pytree_slices(cls, pytree: PyTree, axis: int = 0):
    return cls(
        x for x in jtu.tree_map(partial(unstack, axis=axis), pytree)
    )

  def as_jit_compatible(self) -> 'Dataset':
    """Enable JIT compatibility.

    This is achieved by wrapping the next_fn in a jax io_callback to allow
    jitting it later.

    NOTE: this assumes the iterator always returns pytrees with the same
    structure. This might otherwise lead to unexpected behavior.
    """
    if self._is_jittable:
      logging.warning('Dataset::as_jit_compatible: already jittable, returning self.')
      return self
    head = next(self)
    logging.info('Dataset::as_jit_compatible: head == %r', head)
    next_fn = itertools.chain([head], self).__next__
    j_next_fn = lambda: jax.experimental.io_callback(next_fn, head)
    d = self.from_next_fn(j_next_fn)
    d._is_jittable = True
    return d

  def batch(self, batch_size: int, axis: int = 0) -> 'Dataset':
    """Yield batches of data of specified batch size.

    NOTE: this drops the last batch if it is not full.
    """
    if self._is_jittable:
      def next_fn():
        # TODO(danielwatson6): try implementing with vmap (disabling ordering
        # required) and see if it's faster.
        _, batch = jax.lax.scan(
            lambda carry, _: (carry, self.__next__()),
            jnp.zeros(()),
            jnp.zeros((batch_size,)),
            length=batch_size,
        )
        if axis != 0:
          batch = tree_starmap(partial(jnp.moveaxis, 0, axis), batch)
        return batch
      d = self.from_next_fn(next_fn)
      d._is_jittable = True
      return d

    def next_fn():
      batch = []
      for _ in range(batch_size):
        batch.append(self.__next__())
      return tree_starmap(partial(jnp.stack, axis=axis), batch)
    return self.from_next_fn(next_fn)

  def filter(self, predicate: Callable[[PyTree], bool]) -> 'Dataset':
    """Get a new dataset whose next_fn filters out elements.

    NOTE: this disables jit compatibility.
    """
    next_fn = lambda: next(filter(predicate, iter(self)))
    d = self.from_next_fn(next_fn)
    if self._is_jittable:
      logging.info('Dataset::filter: disabling jit compatibility.')
      logging.warning('Dataset::filter: filtering a jittable dataset might be leading to more calls to `as_jit_compatible` than needed.')
    assert not d._is_jittable
    return d

  def fmap(self, transform: Callable[[NextFn], NextFn]) -> 'Dataset':
    """Get a new dataset whose next_fn is a transform of the current next_fn.

    This probably has no connections to functional programming, don't hate me🥺
    """
    d = self.from_next_fn(transform(self.__next__))
    if self._is_jittable:
      d._is_jittable = True
    return d

  def jit(self, device: jax.Device | None = None, **jit_kwargs) -> 'Dataset':
    """Get a new dataset jitting the `next_fn`."""
    assert self._is_jittable, 'Dataset::jit: dataset is not jittable.'
    # Pin computation to the CPU by default.
    if device is None:
      device = jax.devices("cpu")[0]
    jit_kwargs['device'] = device
    return self.fmap(partial(jax.jit, **jit_kwargs))

  def map(self, f: Callable[[PyTree], PyTree]) -> 'Dataset':
    """Get a new dataset applying an element-wise transformation."""
    d = self.from_next_fn(lambda: f(self.__next__()))
    if self._is_jittable:
      d._is_jittable = True
    return d

  def padded_batch(self, batch_size: int) -> 'Dataset':
    raise NotImplementedError

  def prefetch(self, bufsize: int) -> 'Dataset':
    """Prefetch elements from the dataset via multithreading.

    NOTE: this disables jit compatibility.
    """
    prefetch_next = lambda it, q: q.put(it.__next__())
    def it():
      q = queue.Queue(maxsize=bufsize)
      futures = []
      dataset_consumed = False
      with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        for _ in range(bufsize):
          futures.append(executor.submit(prefetch_next, self, q))
        while True:
          if dataset_consumed and q.empty():
            break
          if not dataset_consumed:
            futures.append(executor.submit(prefetch_next, self, q))
          yield q.get()
          if any(isinstance(f.exception(), StopIteration) for f in futures):
            logging.info('Dataset::prefetch: dataset_consumed == True')
            dataset_consumed = True
          futures = [f for f in futures if not f.done()]
          assert len(futures) <= bufsize
    return self.__class__(it())

  def sleep(self, seconds: int | float) -> 'Dataset':
    def next_fn():
      time.sleep(seconds)
      return self.__next__()
    d = self.from_next_fn(next_fn)
    assert not d._is_jittable
    return d


# TODO(danielwatson6): move to its own test module.
if __name__ == '__main__':
  logging.basicConfig(level=logging.INFO)
  import collections
  import json
  import jax.numpy as jnp

  d = Dataset.from_pytree_slices(jnp.arange(100))
  d = d.batch(8)
  d = d.sleep(0.1)
  d = d.map(lambda x: x * 2)
  d = d.as_jit_compatible()
  d = d.jit()
  d = d.prefetch(1)

  stats = collections.defaultdict(lambda: 0)
  while True:
    try:
      tic = time.time()
      batch = next(d)
      stats['time: batch = next(d)'] += time.time() - tic
      print(batch)
      tic = time.time()
      time.sleep(0.5)
      stats['time: <expensive thing with batch>'] += time.time() - tic
    except StopIteration:
      break

  print(json.dumps(stats, indent=2, sort_keys=True))


