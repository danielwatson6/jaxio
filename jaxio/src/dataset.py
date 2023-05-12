"""JAX datasets."""

from typing import Any, Callable, Iterable, Sequence

import concurrent.futures
from functools import partial
import itertools
import logging
import queue
import time

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxlib import xla_extension


PyTree = Any
NextFn = Callable[[], PyTree]


def tree_starmap(
    f: Callable[[Sequence[PyTree]], PyTree], xs: Sequence[jnp.ndarray]
) -> jnp.ndarray:
  """Tree map a sequence, avoids the sequence being treated as one pytree.

  Args:
    f: the function to be applied.
    xs: the sequence where we wish to tree-map each element.
  Returns:
    The tree-mapped sequence.
  """
  return jtu.tree_map(lambda *xs: f(xs), *xs)


def unstack(x: jnp.ndarray, axis: int = 0) -> jnp.ndarray:
  """Identical to `tf.unstack`.

  Args:
    x: the array to unstack.
    axis: the axis to unstack along.

  Returns:
    The unstacked array.
  """
  # This is faster than the nicer `tuple(jnp.moveaxis(x, axis, 0))` because it
  # avoids an XLA transpose.
  return jtu.tree_map(
      partial(jnp.squeeze, axis=axis), jnp.split(x, x.shape[axis], axis=axis)
  )


class Dataset:
  """JAX dataset.

  `jaxio` datasets are just iterators, compatible with Python's native `iter`
  and `next` builtins, but with handy methods to transform them, very very
  heavily inspired by `tf.data.Dataset`.

  The vanilla constructor can be thought of as analgous to
  `tf.data.Dataset.from_generator`.

  .. note::
    JAX datasets are designed assuming the iterators always return pytrees
    of the same structure. If this is not the case, unexpected behavior might be
    encountered.

  .. warning::
      All datasets created with the `jaxio` API are NOT jit compatible by
      default. The user should instead call `as_jit_compatible` once, and as
      early as possible in the pipeline, to explicitly control the boundary of
      the jax io callback.

  Args:
    it: an arbitrary python iterable, will be converted to a python iterator
      automatically.
  """

  def __init__(self, it: Iterable[Any]) -> None:
    """Constructor.

    Args:
      it: an arbitrary python iterable, will be converted to a python iterator
        automatically.
    """
    self._it = iter(it)
    self._is_jittable = False

  def __iter__(self) -> 'Dataset':
    return self

  # TODO(danielwatson6): what happens when we iterate over a previous dataset?
  # Do we need to tee the iterator?
  def __next__(self) -> PyTree:
    while True:
      try:
        return next(self._it)
      # Sometimes we get these junk errors when the iterator is exhausted.
      except (StopIteration, xla_extension.XlaRuntimeError, RuntimeError) as e:
        logging.info('Dataset::__next__: stopping')
        logging.debug('Dataset::__next__: caught error: %r', e)
        raise StopIteration

  @classmethod
  def from_next_fn(cls, next_fn: NextFn) -> 'Dataset':
    """Create a dataset that infinitely yields fresh calls to `next_fn`.

    Args:
      next_fn: callable that takes no arguments and returns a pytree.
    Returns:
      A new dataset yielding the retunred values of fresh calls to `next_fn`.
    """
    return cls(next_fn() for _ in iter(int, 1))

  @classmethod
  def from_pytree_slices(cls, pytree: PyTree, axis: int = 0):
    """Create a dataset yields the slices of a pytree along a given axis.

    This is mostly useful for debugging, as the whole data lives in memory.

    Args:
      pytree: the pytree whose leaves are to be sliced.
      axis: the axis to slice along.
    Returns:
      A new dataset yielding the slices of the pytree along the given axis.
    """
    return cls(x for x in jtu.tree_map(partial(unstack, axis=axis), pytree))

  def as_jit_compatible(self) -> 'Dataset':
    """Enable JIT compatibility.

    This is achieved by wrapping the next_fn in a jax io_callback to allow
    jitting it later.

    Returns:
      A new dataset that is jit compatible.
    """
    if self._is_jittable:
      logging.warning('Dataset::as_jit_compatible: already jittable, returning self.')
      return self
    head = next(self)
    logging.info('Dataset::as_jit_compatible: head == %r', head)
    next_fn = itertools.chain([head], self).__next__
    j_next_fn = lambda: jax.experimental.io_callback(next_fn, head)
    d = self.from_next_fn(j_next_fn)
    logging.info('Dataset::as_jit_compatible: enabling jit compatibility')
    d._is_jittable = True
    return d

  def batch(self, batch_size: int, axis: int = 0) -> 'Dataset':
    """Yield batches of data of specified batch size.

    The new dataset will use a more efficient batching (compatible with jit) if
    the current dataset is jit compatible.

    .. note::
      This drops the last batch if it is not full.

    Args:
      batch_size: the size of the batches to yield.
      axis: the axis to stack the batches along.
    Returns:
      A new dataset that yields batches of data.
    """
    if self._is_jittable:
      logging.info('Dataset::batch: jit compatible')
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

    logging.info('Dataset::batch: not jit compatible')
    def next_fn():
      batch = []
      for _ in range(batch_size):
        batch.append(self.__next__())
      return tree_starmap(partial(jnp.stack, axis=axis), batch)
    return self.from_next_fn(next_fn)

  def enumerate(self) -> 'Dataset':
    """Yield (index, element) pairs.

    .. warning::
      The result will not be jit compatible.

    Returns:
      A new dataset that yields (index, element) pairs.
    """
    d = self.__class__(enumerate(self))
    if self._is_jittable:
      logging.info('Dataset::prefetch: disabling jit compatibility.')
    assert not d._is_jittable
    return d

  def filter(self, f: Callable[[PyTree], bool]) -> 'Dataset':
    """Get a new dataset whose next_fn filters out elements.

    .. warning::
      The result will not be jit compatible.

    Args:
      f: a callable that takes a pytree and returns whether to keep it.
    Returns:
      A new dataset that filters out elements.
    """
    d = self.__class__(filter(f, self))
    if self._is_jittable:
      logging.info('Dataset::filter: disabling jit compatibility.')
    assert not d._is_jittable
    return d

  def fmap(self, transform: Callable[[NextFn], NextFn]) -> 'Dataset':
    """Get a new dataset whose next_fn is a transform of the current next_fn.

    This probably has no connections to functional programming, don't hate me🥺

    Args:
      transform: a callable that takes a next_fn and returns a new next_fn.
    Returns:
      A new dataset whose next_fn is a transform of the current next_fn.
    """
    d = self.from_next_fn(transform(self.__next__))
    if self._is_jittable:
      d._is_jittable = True
    return d

  def jit(self, **jit_kwargs) -> 'Dataset':
    """Get a new dataset jitting the `next_fn`.

    .. warning::
      This does NOT pin the computation to the CPU by default. The user
      should use `jax.default_device` context managers to do this.

    Args:
      jit_kwargs: kwargs to pass to `jax.jit`.
    Returns:
      A new dataset whose next_fn is jitted.
    """
    if not self._is_jittable:
      raise ValueError('Dataset::jit: dataset is not jittable.')
    return self.fmap(partial(jax.jit, **jit_kwargs))

  def map(self, f: Callable[[PyTree], PyTree]) -> 'Dataset':
    """Get a new dataset applying an element-wise transformation.

    Args:
      f: a callable that takes a pytree and returns a new pytree.
    Returns:
      A new dataset applying `f` to each element of the current dataset.
    """
    d = self.from_next_fn(lambda: f(self.__next__()))
    if self._is_jittable:
      d._is_jittable = True
    return d

  def prefetch(self, bufsize: int = 1) -> 'Dataset':
    """Prefetch elements from the dataset into a queue of given size.

    This is achieved by letting a thread pool executor (with a single worker)
    make calls to the current next_fn and putting the results in a queue.

    .. warning::
      The result will not be jit compatible.

    Args:
      bufsize: the size of the queue to prefetch into.
    Returns:
      A new dataset that prefetches elements into a queue.
    """
    prefetch_next = lambda it, q: q.put(it.__next__())
    def it():
      q = queue.Queue(maxsize=bufsize)
      # We need to keep track of the dispatched futures to know when the dataset
      # is consumed, otherwise we will wait forever for the next element. This
      # is ok, because we can guarantee that we never have more than `bufsize`
      # futures in flight.
      futures = []
      dataset_consumed = False
      with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        for _ in range(bufsize):
          futures.append(executor.submit(prefetch_next, self, q))
        while True:
          if dataset_consumed and q.empty():
            break
          # The order here is crucial, must dispatch before yielding.
          if not dataset_consumed:
            futures.append(executor.submit(prefetch_next, self, q))
          yield q.get()
          # Now we can update the active futures and check if we're done.
          if any(isinstance(f.exception(), StopIteration) for f in futures):
            logging.info('Dataset::prefetch: dataset_consumed == True')
            dataset_consumed = True
          futures = [f for f in futures if not f.done()]
          assert len(futures) <= bufsize
    d = self.__class__(it())
    if self._is_jittable:
      logging.info('Dataset::prefetch: disabling jit compatibility.')
    assert not d._is_jittable
    return d

  def repeat(self, n: int | None = None) -> 'Dataset':
    """Repeat the dataset n times (or infinitely if left unspecified).

    .. warning::
      The result will not be jit compatible.

    Args:
      n: the number of times to repeat the dataset. If None or not specified,
        the dataset will be repeated infinitely.
    Returns:
      A new dataset repeating the current dataset.
    """
    assert n is None or n > 0
    # `itertools.cycle` unfortunately caches the first cycle.
    def it():
      i = 0
      it1 = self
      while (n is None or i < n):
        it1, it2 = itertools.tee(it1)
        yield from it2
        i += 1
    d = self.__class__(it())
    if self._is_jittable:
      logging.info('Dataset::repeat: disabling jit compatibility.')
    assert not d._is_jittable
    return d

  def sleep(self, seconds: int | float) -> 'Dataset':
    """Get a new dataset that sleeps for `seconds` before yielding an element.

    Especially useful for debugging prefetch performance.

    .. warning::
      The result will not be jit compatible.

    Args:
      seconds: the number of seconds to sleep before yielding an element.
    Returns:
      A new dataset that sleeps for `seconds` before yielding an element.
    """
    def next_fn():
      time.sleep(seconds)
      return self.__next__()
    d = self.from_next_fn(next_fn)
    if self._is_jittable:
      logging.info('Dataset::repeat: disabling jit compatibility.')
    assert not d._is_jittable
    return d
