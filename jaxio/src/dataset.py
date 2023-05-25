"""JAX datasets."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import Any, Generic, Type, TypeVar

import concurrent.futures
import functools
import itertools
import logging
import operator as op
import queue
import time

import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.tree_util as jtu
from jaxlib import xla_extension
import numpy as np


PyTree = Any

S = TypeVar('S', PyTree)
T = TypeVar('T', PyTree)
U = TypeVar('U', PyTree)

NextFn = Callable[[], T]


at = lambda x, indices: x[indices]
compose2 = lambda f, g: (lambda *a, **kw: f(g(*a, **kw)))
compose = lambda *fs: functools.reduce(compose2, fs)
partial = functools.partial
ret = lambda x: (lambda: x)


def scanzero(
    scan_fn: Callable[[T], tuple[T, jax.Array]],
    init: T,
    length: int = 1,
) -> tuple[T, jax.Array]:
  """Scan over a function that takes no input sequence elements.

  Args:
    scan_fn: the scan function to be applied (only depends on the carry).
    init: the initial carry.
    length: the length of the scan.
  Returns:
    The final carry.
  """
  return jax.lax.scan(
      lambda carry, _: scan_fn(carry), init, jnp.zeros((length,))
  )


def tree_starmap(f: Callable[[Sequence[T]], U], xs: Sequence[T]) -> Sequence[U]:
  """Tree map a sequence, avoiding the sequence being treated as one pytree.

  Args:
    f: the function to be applied.
    xs: the sequence where we wish to tree-map each element.
  Returns:
    The tree-mapped sequence.
  """
  return jtu.tree_map(lambda *xs: f(xs), *xs)


def unstack(x: jax.Array, axis: int = 0) -> jax.Array:
  """Identical to `tf.unstack` but in JAX.

  Args:
    x: the array to unstack.
    axis: the axis to unstack along.

  Returns:
    The unstacked array.
  """
  # This is faster than the nicer `tuple(jnp.moveaxis(x, axis, 0))` because it
  # avoids an XLA transpose:
  # https://github.com/google/jax/blob/main/jax/_src/numpy/lax_numpy.py#L896
  return jtu.tree_map(
      partial(jnp.squeeze, axis=axis), jnp.split(x, x.shape[axis], axis=axis)
  )


def vmapzero(f: Callable[[], PyTree], batch_size: int) -> Callable[[], PyTree]:
  """Transform a function that takes no arguments with `jax.vmap`."""
  return partial(jax.vmap(lambda _: f()), jnp.zeros((batch_size,)))


def vscanzero(f: Callable[[], PyTree], batch_size: int) -> Callable[[], PyTree]:
  """Identical to `vmapzero` but deterministic and sequential."""
  vf = partial(scanzero, lambda _: (_, f()), jnp.zeros(()), length=batch_size)
  return compose(op.itemgetter(1), vf)


class Dataset(Generic[T]):
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

  def __init__(self: Dataset[T], it: Iterable[T]) -> None:
    self._it = iter(it)
    self._is_jittable = False
    self._cardinality = -1
    self._unusable = False

  def __iter__(self: Dataset[T]) -> Dataset[T]:
    return self

  def __next__(self: Dataset[T]) -> T:
    if self._unusable:
      raise ValueError(
          'dataset::Dataset.__next__: attempting to call a dataset that has '
          'been transformed, this can lead to unexpected behavior'
      )
    try:
      return self.next_fn()
    # Sometimes we get this junk when the iterator is exhausted.
    except (StopIteration, xla_extension.XlaRuntimeError, RuntimeError) as e:
      logging.info('dataset::Dataset.__next__: stopping')
      logging.debug('dataset::Dataset.__next__: caught error: %r', e)
      raise StopIteration

  def __len__(self: Dataset[T]) -> int:
    return self._cardinality

  def __lshift__(self: Dataset[T], f: Callable[[T], Dataset[U]]) -> Dataset[U]:
    return self.flat_map(f)

  def __rrshift__(self: Dataset[T], f: Callable[[T], Dataset[U]]) -> Dataset[U]:
    return self.flat_map(f)

  @property
  def next_fn(self: Dataset[T]) -> NextFn[T]:
    """The next function of the dataset without checks or other side effects."""
    return (
        self._it.next_fn if isinstance(self._it, Dataset) else self._it.__next__
    )

  @classmethod
  def from_iter(cls: Type[Dataset[T]], it: Iterable[T]) -> Dataset[T]:
    """Create a dataset from an iterable.

    Args:
      it: the iterable to be wrapped.
    Returns:
      A new dataset yielding the elements of the iterable.
    """
    return cls(iter(it))

  @classmethod
  def from_pytree(cls: Type[Dataset[T]], pytree: U) -> Dataset[U]:
    """Create a single-element dataset from a pytree.

    Args:
      pytree: the pytree to be wrapped.
    Returns:
      A new dataset yielding the pytree.
    """
    return cls(pytree for _ in range(1))

  @classmethod
  def from_pytree_slices(
      cls: Type[Dataset[T]], pytree: U, axis: int = 0
  ) -> Dataset[U]:
    """Create a dataset yields the slices of a pytree along a given axis.

    This is mostly useful for debugging, as the whole data lives in memory.

    .. warning::
      The result will not be jit compatible.

    Args:
      pytree: the pytree whose leaves are to be sliced.
      axis: the axis to slice along.
    Returns:
      A new dataset yielding the slices of the pytree along the given axis.
    """
    # WIP(danielwatson6): i think this will be wrong for actual pytrees
    # return cls(x for x in jtu.tree_map(partial(unstack, axis=axis), pytree))
    return cls.unit(pytree)

  @classmethod
  def unit(cls: Type[Dataset[T]], pytree: U) -> Dataset[U]:
    """Monadic alias for ``Dataset.from_pytree(pytree)``."""
    return cls.from_pytree(pytree)

  def as_jit_compatible(self: Dataset[T]) -> Dataset[T]:
    """Enable JIT compatibility.

    This is achieved by wrapping the next_fn in a jax io_callback to allow
    jitting it later.

    Returns:
      A new dataset that is jit compatible.
    """
    if self._is_jittable:
      logging.warning(
          'dataset::Dataset.as_jit_compatible: already jit compatible, '
          'returning unchanged dataset'
      )
      return self

    head, d = self.peek()
    if not all(isinstance(x, jnp.ndarray) for x in jtu.tree_leaves(head)):
      raise ValueError(
          'dataset::Dataset.as_jit_compatible: elements must all be JAX '
          f'arrays, got type {type(head)}: {head}'
      )
    iocall = partial(jax.experimental.io_callback, result_shape_dtypes=head)
    d = d.transform(partial(partial, iocall))
    logging.info(
        'dataset::Dataset.as_jit_compatible: enabling jit compatibility'
    )
    d._is_jittable = True
    return d

  def batch(
      self: Dataset[T],
      batch_size: int,
      axis: int = 0,
      deterministic: bool = True,
  ) -> Dataset[U]:
    """Yield batches of data of specified batch size.

    The new dataset will use a more efficient batching (compatible with jit) if
    the current dataset is jit compatible. We can't use it otherwise because
    ``jax.lax.scan`` compiles.

    .. note::
      This drops the last batch if it is not full.

    .. note::
      When not jit compatible, this will convert pytree leaves to numpy arrays.

    Args:
      batch_size: the size of the batches to yield.
      axis: the axis to stack the batches along.
      deterministic: if ``False``, will use vmap, which will result in
        nondeterministic ordering of batch elements.
    Returns:
      A new dataset that yields batches of data.
    """
    if self._is_jittable:
      logging.info('dataset::Dataset.batch: jit compatible')
      if deterministic:
        logging.debug('dataset::Dataset.batch: deterministic')
        transform = vscanzero
      else:
        logging.warning(
            'dataset::Dataset.batch: nondeterministic batching might change in '
            'the future due to weird JAX semantics of, e.g., vmap of '
            'scan/while_loop of io_callback. See: '
            'https://jax.readthedocs.io/en/latest/notebooks/external_callbacks.html'
        )
        transform = vmapzero
      d = self.transform(partial(transform, batch_size=batch_size))
      if axis != 0:
        moveaxis = partial(jnp.moveaxis, source=0, dest=axis)
        d = d.map(partial(jtu.tree_map, moveaxis))
      return d

    logging.info('dataset::Dataset.batch: not jit compatible')
    def makeit() -> Iterator[U]:
      while True:
        batch = tuple(itertools.islice(self, batch_size))
        if len(batch) != batch_size:
          return
        yield tree_starmap(partial(np.stack, axis=axis), batch)
    d = self.__class__(makeit())
    self._unusable = True
    return d

  def cache(self: Dataset[T]) -> Dataset[T]:
    raise NotImplementedError

  def concatenate(self: Dataset[T], *ds: Dataset[T]) -> Dataset[T]:
    datasets = (self, *ds)
    d = self.from_pytree_slices(jnp.arange(len(datasets)))
    return partial(at, datasets) >> d

  def enumerate(self: Dataset[T]) -> Dataset[tuple[int, T]]:
    """Yield (index, element) pairs.

    Returns:
      A new dataset that yields (index, element) pairs.
    """
    # return self.scan(lambda i, el: (i + 1, el), jnp.zeros((), dtype=jnp.int32))

    if self._is_jittable:
      logging.info('dataset::Dataset.enumerate: jit compatible')
      transform = lambda next_fn: partial(
          scanzero,
          lambda i: (i + 1, next_fn()),
          jnp.zeros((), dtype=jnp.int32),
      )
      return self.transform(transform)

    logging.info('dataset::Dataset.enumerate: not jit compatible')
    d = self.__class__(enumerate(self))
    self._unusable = True
    return d

  def filter(self: Dataset[T], f: Callable[[T], bool]) -> Dataset[T]:
    """Get a new dataset whose next_fn filters out elements.

    Args:
      f: a callable that takes a pytree and returns whether to keep it.
    Returns:
      A new dataset that filters out elements.
    """
    if self._is_jittable:
      logging.info('dataset::Dataset.filter: jit compatible')
      transform = lambda next_fn: partial(
          jax.lax.while_loop,
          compose(jnp.logical_not, f),
          lambda _: next_fn(),
          next_fn(),
      )
      return self.transform(transform)

    logging.info('dataset::Dataset.filter: not jit compatible')
    d = self.__class__(filter(f, self))
    self._unusable = True
    return d

  def flat_map(self: Dataset[T], f: Callable[[T], Dataset[U]]) -> Dataset[U]:
    """Convert elements into datasets and flatten the result into one dataset.

    This is just ``interleave`` with ``cycle_length=1`` and ``block_length=1``.
    Dataset objects also allow to do this with monadic notation:

    .. code-block:: python

        d = f >> d  # same as `d = d.flat_map(f)`
        d = d << f  # same as `d = d.flat_map(f)`

    Args:
      f: a callable that takes a pytree and returns a dataset.
    Returns:
      Flattened dataset.
    """
    return self.interleave(f, cycle_length=1, block_length=1)

  def jit(self: Dataset[T], **jit_kwargs) -> Dataset[T]:
    """Get a new dataset jitting the `next_fn`.

    .. warning::
      This does NOT pin the computation to the CPU by default.

    Args:
      jit_kwargs: kwargs to pass to `jax.jit`.
    Returns:
      A new dataset whose next_fn is jitted.
    """
    if not self._is_jittable:
      raise ValueError('dataset::Dataset.jit: dataset is not jit compatible')
    return self.transform(partial(jax.jit, **jit_kwargs))

  # TODO(danielwatson6): implement
  # TODO(danielwatson6): docs
  # TODO(danielwatson6): can we make this work with jit?
  def interleave(
      self: Dataset[T],
      f: Callable[[T], Dataset[U]],
      cycle_length: int = 1,
      block_length: int = 1,
  ) -> Dataset[U]:
    """Map elements to datasets and interleave the results.

    This is a generalization of flat map (i.e., `f >> d`), that allows to

    Args:
      f: a callable that takes a pytree and returns a dataset.
      cycle_length: the number of original elements to process concurrently.
      block_length: the number of consecutive elements to produce from each
        original input element before cycling to another input element.
    Returns:
      Interleaved dataset.
    """

    def makeit() -> Iterator[U]:
      ...

    d = self.__class__(makeit())
    if self._is_jittable:
      logging.warning(
          'dataset::Dataset.interleave: disabling jit compatibility'
      )
    assert not d._is_jittable
    self._unusable = True
    return d

  def map(self: Dataset[T], f: Callable[[T], U]) -> Dataset[U]:
    """Get a new dataset applying an element-wise transformation.

    Args:
      f: a callable that takes a pytree and returns a new pytree.
    Returns:
      A new dataset applying `f` to each element of the current dataset.
    """
    return self.transform(partial(compose, f))

  def peek(self: Dataset[T]) -> tuple[T, Dataset[T]]:
    head = next(self)

    if self._is_jittable:
      logging.info('dataset::Dataset.peek: jit compatible')
      def transform(next_fn: NextFn[T]) -> NextFn[T]:
        scan_fn = lambda is_first: (
            jnp.zeros_like(is_first), jax.lax.cond(is_first, ret(head), next_fn)
        )
        return partial(scanzero, scan_fn, jnp.array(True))
      d = self.transform(transform)

    else:
      logging.info('dataset::Dataset.peek: not jit compatible')
      d = self.__class__(itertools.chain([head], self))
      self._unusable = True

    return head, d

  def prefetch(self: Dataset[T], bufsize: int = 1) -> Dataset[T]:
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
    prefetch_next = lambda it, q: q.put(next(it))

    def makeit() -> Iterator[T]:
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
            logging.debug('dataset::Dataset.prefetch: dataset_consumed == True')
            dataset_consumed = True
          futures = [f for f in futures if not f.done()]
          assert len(futures) <= bufsize

    d = self.__class__(makeit())
    if self._is_jittable:
      # Logging level is intentional, prefetching from a jit compatible dataset
      # is an expected use case.
      logging.info('dataset::Dataset.prefetch: disabling jit compatibility')
    assert not d._is_jittable
    self._unusable = True
    return d

  # TODO(danielwatson6): can we make this work with jit?
  # TODO(danielwatson6): memory usage? what does tf do?
  # https://github.com/tensorflow/tensorflow/blob/v2.12.0/tensorflow/core/kernels/data/repeat_dataset_op.cc
  def repeat(self: Dataset[T], n: int | None = None) -> Dataset[T]:
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
    def makeit() -> Iterator[T]:
      i = 0
      it1 = self
      while (n is None or i < n):
        # this also copies rippppp
        it1, it2 = itertools.tee(it1)
        yield from it2
        i += 1

    d = self.__class__(makeit())
    if self._is_jittable:
      logging.warning('dataset::Dataset.repeat: disabling jit compatibility')
    assert not d._is_jittable
    self._unusable = True
    return d

  def scan(
      self: Dataset[T],
      f: Callable[[S, T], tuple[S, U]],
      init: S,
      length: int = 1,
  ) -> Dataset[tuple[S, U]]:
    if self._is_jittable:
      logging.info('dataset::Dataset.scan: jit compatible')
      def transform(next_fn: NextFn[T]) -> NextFn[tuple[S, U]]:
        ...
      return self.transform()

    logging.info('dataset::Dataset.scan: not jit compatible')
    ...

  def shuffle(
      self: Dataset[T], base_rng: jrandom.KeyArray, bufsize: int
  ) -> Dataset[T]:
    """Shuffle the dataset with buffers of given size.

    Args:
      base_rng: the base random key to use for shuffling.
      bufsize: the size of the buffer to use for shuffling.
    Returns:
      A new dataset that shuffles the current dataset.
    """
    assert bufsize > 1

    def tree_shuffle(el: tuple[int, U]) -> U:
      i, data = el
      rng = jrandom.fold_in(base_rng, i)
      leaves = jtu.tree_leaves(data)
      n = leaves[0].shape[0]
      assert all(x.shape[0] == n for x in leaves[1:])
      perm = jrandom.permutation(rng, jnp.arange(n))
      return jtu.tree_map(partial(at, indices=perm), data)

    return self.batch(bufsize).enumerate().map(tree_shuffle).unbatch()

  # TODO
  def skip(self: Dataset[T]) -> Dataset[T]:
    raise NotImplementedError

  def sleep(self: Dataset[T], seconds: int | float) -> Dataset[T]:
    """Get a new dataset that sleeps for `seconds` before yielding an element.

    Especially useful for debugging prefetch performance.

    Args:
      seconds: the number of seconds to sleep before yielding an element.
    Returns:
      A new dataset that sleeps for `seconds` before yielding an element.
    """
    def sleep_fn() -> jax.Array:
      time.sleep(seconds)
      return jnp.zeros(())

    if self._is_jittable:
      logging.info('dataset::Dataset.sleep: jit compatible')
      sleep_fn = partial(jax.experimental.io_callback, sleep_fn, jnp.zeros(()))
    else:
      logging.info('dataset::Dataset.sleep: not jit compatible')

    def transform(next_fn: NextFn[T]) -> NextFn[T]:
      def sleepy_next_fn() -> T:
        sleep_fn()
        return next_fn()
      return sleepy_next_fn
    return self.transform(transform)

  def take(self: Dataset[T]) -> Dataset[T]:
    raise NotImplementedError

  def transform(
      self: Dataset[T], f: Callable[[NextFn[T]], NextFn[U]]
  ) -> Dataset[U]:
    """Get a new dataset whose next_fn is a transform of the current next_fn.

    Args:
      f: a callable that takes a next_fn and returns a new next_fn.
    Returns:
      A new dataset whose next_fn is a transform of the current next_fn.
    """
    # Important: do not use `self.__next__`, it will trigger an `_unusable`
    # check because it will call upstream `__next__` methods.
    next_fn = f(self.next_fn)
    # Python doesn't have `(_ while True)`, we use `iter(int, 1)` as infinity.
    d = self.__class__(next_fn() for _ in iter(int, 1))
    d._is_jittable = self._is_jittable
    self._unusable = True
    return d

  # TODO
  def unbatch(self: Dataset[T], axis: int = 0) -> Dataset[U]:
    """Get a new dataset that unbatches along the given axis.

    Args:
      axis: the axis to unbatch along.
    Returns:
      A new dataset that unbatch the current dataset.
    """
    head, d = self.peek()
    if self._is_jittable:
      logging.info('dataset::Dataset.unbatch: jit compatible')
      def transform_fn(next_fn: NextFn[T]) -> NextFn[U]:
        def scan_fn(carry: tuple[int, T]) -> tuple[tuple[int, T], jax.Array]:
          i, batch = carry
          should_call_next = i >= head.shape[axis]
          batch = jax.lax.cond(should_call_next, next_fn, ret(batch))
          i = jnp.where(should_call_next, 0, i)
          el = jtu.tree_map(partial(jnp.take, indices=i, axis=axis), batch)
          return (i + 1, batch), el
        init = (0, jnp.zeros_like(head))
        # TODO(danielwatson6): don't we need to get rid of the carry?
        return partial(scanzero, scan_fn, init)
      return d.transform(transform_fn)

    logging.info('dataset::Dataset.unbatch: not jit compatible')

    # TODO
    d = self.__class__(
        el for batch in d for el in jtu.tree_map(np_unstack, batch)
    )
    self._unusable = True
    return d
