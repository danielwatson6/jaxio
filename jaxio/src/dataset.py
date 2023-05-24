"""JAX datasets."""

from typing import Any, Callable, Iterable, Iterator, Sequence

import concurrent.futures
from functools import partial
import itertools
import logging
import queue
import time

import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.tree_util as jtu
from jaxlib import xla_extension
import numpy as onp


PyTree = Any
NextFn = Callable[[], PyTree]


def scanzero(
    scan_fn: Callable[[PyTree], tuple[PyTree, PyTree]],
    init: PyTree,
    length: int = 1,
) -> tuple[PyTree, jax.Array]:
  """Scan with zeroed-out x's.

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


def tree_starmap(
    f: Callable[[Sequence[PyTree]], PyTree], xs: Sequence[PyTree]
) -> PyTree:
  """Tree map a sequence, avoids the sequence being treated as one pytree.

  Args:
    f: the function to be applied.
    xs: the sequence where we wish to tree-map each element.
  Returns:
    The tree-mapped sequence.
  """
  return jtu.tree_map(lambda *xs: f(xs), *xs)


def unstack(x: jax.Array, axis: int = 0) -> jax.Array:
  """Identical to `tf.unstack`.

  Args:
    x: the array to unstack.
    axis: the axis to unstack along.

  Returns:
    The unstacked array.
  """
  # This is faster than the nicer `tuple(jnp.moveaxis(x, 0, axis))` because it
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
        logging.info('dataset::Dataset.__next__: stopping')
        logging.debug('dataset::Dataset.__next__: caught error: %r', e)
        raise StopIteration

  @classmethod
  def from_pytree_slices(cls, pytree: PyTree, axis: int = 0):
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
    return cls(x for x in jtu.tree_map(partial(unstack, axis=axis), pytree))

  def as_jit_compatible(self) -> 'Dataset':
    """Enable JIT compatibility.

    This is achieved by wrapping the next_fn in a jax io_callback to allow
    jitting it later.

    Returns:
      A new dataset that is jit compatible.
    """
    if self._is_jittable:
      logging.warning('dataset::Dataset.as_jit_compatible: already jit compatible, returning unchanged dataset.')
      return self

    head, d = self.peek()
    assert all(isinstance())
    transform = lambda next_fn: partial(
        jax.experimental.io_callback, next_fn, head
    )
    d = d.transform(transform)
    logging.info('dataset::Dataset.as_jit_compatible: enabling jit compatibility')
    d._is_jittable = True
    return d

  def batch(self, batch_size: int, axis: int = 0) -> 'Dataset':
    """Yield batches of data of specified batch size.

    The new dataset will use a more efficient batching (compatible with jit) if
    the current dataset is jit compatible. We can't use it otherwise because
    `jax.lax.scan` compiles.

    .. note::
      This drops the last batch if it is not full.

    Args:
      batch_size: the size of the batches to yield.
      axis: the axis to stack the batches along.
    Returns:
      A new dataset that yields batches of data.
    """
    if self._is_jittable:
      logging.info('dataset::Dataset.batch: jit compatible')
      def transform(next_fn: NextFn) -> NextFn:
        return lambda: scanzero(
            lambda zero: (zero, next_fn()),
            jnp.zeros(()),
            length=batch_size,
        )[1]
      d = self.transform(transform)
      if axis != 0:
        tree_moveaxis = partial(jtu.tree_map, partial(jnp.moveaxis, 0, axis))
        d = d.map(tree_moveaxis)
      return d

    logging.info('dataset::Dataset.batch: not jit compatible')
    def it() -> Iterator[PyTree]:
      while True:
        batch = tuple(itertools.islice(self, batch_size))
        if len(batch) != batch_size:
          return
        # TODO(danielwatson6): would be nice to return original types.
        yield tree_starmap(partial(onp.stack, axis=axis), batch)
    return self.__class__(it())

  def enumerate(self) -> 'Dataset':
    """Yield (index, element) pairs.

    Returns:
      A new dataset that yields (index, element) pairs.
    """
    if self._is_jittable:
      logging.info('dataset::Dataset.enumerate: jit compatible')
      transform = lambda next_fn: partial(
          scanzero,
          lambda i: (i + 1, next_fn()),
          jnp.zeros(()),
      )
      return self.transform(transform)

    logging.info('dataset::Dataset.enumerate: not jit compatible')
    return self.__class__(enumerate(self))

  def filter(self, f: Callable[[PyTree], bool]) -> 'Dataset':
    """Get a new dataset whose next_fn filters out elements.

    Args:
      f: a callable that takes a pytree and returns whether to keep it.
    Returns:
      A new dataset that filters out elements.
    """
    if self._is_jittable:
      logging.info('dataset::Dataset.filter: jit compatible.')
      transform = lambda next_fn: partial(
          jax.lax.while_loop,
          lambda el: jnp.logical_not(f(el)),
          lambda _: next_fn(),
          next_fn(),
      )
      return self.transform(transform)

    logging.info('dataset::Dataset.filter: not jit compatible.')
    return self.__class__(filter(f, self))

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
      raise ValueError('dataset::Dataset.jit: dataset is not jit compatible.')
    return self.transform(partial(jax.jit, **jit_kwargs))

  def map(self, f: Callable[[PyTree], PyTree]) -> 'Dataset':
    """Get a new dataset applying an element-wise transformation.

    Args:
      f: a callable that takes a pytree and returns a new pytree.
    Returns:
      A new dataset applying `f` to each element of the current dataset.
    """
    return self.transform(lambda next_fn: (lambda: f(next_fn())))

  def peek(self) -> tuple[PyTree, 'Dataset']:
    head = next(self)

    if self._is_jittable:
      logging.info('dataset::Dataset.peek: jit compatible')
      def transform(next_fn: NextFn) -> NextFn:
        scan_fn = lambda is_first: (
            jnp.zeros_like(is_first),
            jax.lax.cond(is_first, lambda: head, next_fn)
        )
        return partial(scanzero, scan_fn, jnp.array(True))
      d = self.transform(transform)

    else:
      logging.info('dataset::Dataset.peek: not jit compatible')
      d = self.__class__(itertools.chain([head], self))

    return head, d

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
    prefetch_next = lambda it, q: q.put(next(it))

    def it() -> Iterator[PyTree]:
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

    d = self.__class__(it())
    if self._is_jittable:
      logging.warning('dataset::Dataset.prefetch: disabling jit compatibility.')
    assert not d._is_jittable
    return d

  # TODO(danielwatson6): is there a way to make this jittable?
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
    def it() -> Iterator[PyTree]:
      i = 0
      it1 = self
      while (n is None or i < n):
        it1, it2 = itertools.tee(it1)
        yield from it2
        i += 1

    d = self.__class__(it())
    if self._is_jittable:
      logging.warning('dataset::Dataset.repeat: disabling jit compatibility.')
    assert not d._is_jittable
    return d

  def shuffle(
      self, base_rng: jrandom.KeyArray, bufsize: int, axis: int = 0
  ) -> 'Dataset':
    if self._is_jittable:
      logging.info('dataset::Dataset.shuffle: jit compatible')
      gather_fn = partial(jnp.take, axis=axis)
    else:
      logging.info('dataset::Dataset.shuffle: not jit compatible')
      # TODO(danielwatson6): would be nice to return original types.
      gather_fn = partial(onp.take, axis=axis)

    def tree_shuffle(buf: PyTree) -> PyTree:
      i, el = buf
      rng = jrandom.fold_in(base_rng, i)
      leaves = jtu.tree_leaves(el)
      n = leaves[0].shape[axis]
      assert all(a.shape[axis] == n for a in leaves[1:])
      perm = jrandom.permutation(rng, jnp.arange(n))
      return jtu.tree_map(partial(gather_fn, indices=perm), el)

    return self.batch(bufsize).enumerate().map(tree_shuffle).unbatch()

  def sleep(self, seconds: int | float) -> 'Dataset':
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

    def transform(next_fn: NextFn) -> NextFn:
      def sleepy_next_fn() -> PyTree:
        sleep_fn()
        return next_fn()
      return sleepy_next_fn
    return self.transform(transform)

  def transform(self, f: Callable[[NextFn], NextFn]) -> 'Dataset':
    """Get a new dataset whose next_fn is a transform of the current next_fn.

    Args:
      f: a callable that takes a next_fn and returns a new next_fn.
    Returns:
      A new dataset whose next_fn is a transform of the current next_fn.
    """
    next_fn = f(self.__next__)
    d = self.__class__(next_fn() for _ in iter(int, 1))
    d._is_jittable = self._is_jittable
    return d

  def unbatch(self, axis: int = 0) -> 'Dataset':
    """Get a new dataset that unbatches along the given axis.

    Args:
      axis: the axis to unbatch along.
    Returns:
      A new dataset that unbatch the current dataset.
    """
    head, d = self.peek()
    if self._is_jittable:
      logging.info('dataset::Dataset.unbatch: jit compatible')
      def transform_fn(next_fn: NextFn) -> NextFn:
        def scan_fn(
            carry: tuple[int, PyTree]
        ) -> tuple[tuple[int, PyTree], PyTree]:
          i, batch = carry
          should_call_next = i >= head.shape[axis]
          batch = jax.lax.cond(should_call_next, next_fn, lambda: batch)
          i = jnp.where(should_call_next, 0, i)
          el = jtu.tree_map(partial(jnp.take, indices=i, axis=axis), batch)
          return (i + 1, batch), el
        init = (0, jnp.zeros_like(head))
        return partial(scanzero, scan_fn, init)
      return d.transform(transform_fn)

    logging.info('dataset::Dataset.unbatch: not jit compatible')
    def it() -> Iterator[PyTree]:
      for batch in d:
        # TODO(danielwatson6): would be nice to return original types.
        yield from jtu.tree_map(
            lambda x: tuple(onp.moveaxis(x, 0, axis)), batch
        )
    return self.__class__(it())
