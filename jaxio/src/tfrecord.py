"""TFRecord utilities."""

from typing import Any, Iterator

import binascii
import contextlib
from functools import partial
import threading


BYTE_ORDER = 'little'
# https://github.com/tensorflow/tensorflow/blob/v2.12.0/tensorflow/tsl/lib/hash/crc32c.h#L44
MASK_DELTA = 0xa282ead8


def uint32(x: int) -> int:
  return x & 0xffffffff


def mask_crc32(crc: int) -> int:
  return uint32((uint32(crc >> 15) | uint32(crc << 17)) + MASK_DELTA)


def unmask_crc32(masked_crc: int) -> int:
  rot = uint32(masked_crc - MASK_DELTA)
  return uint32(rot >> 17) | uint32(rot << 15)


# TODO(danielwatson6): parallel read support
def read(path: str, thread_safe: bool = False) -> Iterator[bytes]:
  """Read a tfrecord file and yield records from it."""
  with open(path, 'rb') as fp:
    while True:
      # If reading in parallel, we must not overlap reads.
      with threading.Lock() if thread_safe else contextlib.nullcontext():
        # Each record has the following format:
        #   uint64 length
        #   uint32 length_mcrc32
        #   byte   raw_bytes[length]
        #   uint32 raw_bytes_mcrc32
        # https://github.com/tensorflow/tensorflow/blob/v2.12.0/tensorflow/tsl/lib/io/record_writer.cc#L104
        header = fp.read(8 + 4)
        if len(header) == 0:
          break
        if len(header) != 12:
          raise ValueError('tfrecord::read: bad remainder bytes')
        length_uint64 = header[:8]
        length_mcrc32_uint32 = header[8:]
        length_mcrc32 = int.from_bytes(length_mcrc32_uint32, BYTE_ORDER)
        assert unmask_crc32(length_mcrc32) == binascii.crc32(length_uint64)
        length = int.from_bytes(length_uint64, BYTE_ORDER)
        raw_bytes = fp.read(length)
        if len(raw_bytes) != length:
          raise ValueError('tfrecord::read: bad remainder bytes')
        raw_bytes_mcrc32_uint32 = fp.read(4)

      if len(raw_bytes_mcrc32_uint32) != 4:
        raise ValueError('tfrecord::read: bad remainder bytes')
      raw_bytes_mcrc32 = int.from_bytes(raw_bytes_mcrc32_uint32, BYTE_ORDER)
      assert unmask_crc32(raw_bytes_mcrc32) == binascii.crc32(raw_bytes)
      yield raw_bytes


@contextlib.contextmanager
def writer(
  path: str, append_mode: bool = False, thread_safe: bool = False
) -> Iterator[Any]:
  """Context manager yielding a `write_fn` to amend/write in tfrecord format.

  Args:
    path: path to the tfrecord file.
    append_mode: whether to append to the file or overwrite it.
    thread_safe: whether to use a lock to ensure thread safety.
  Returns:
    A context manager yielding a `write_fn`, accepts bytes to write as argument.
  """
  def write_fn(fp, raw_bytes: bytes) -> None:
    # Each record has the following format:
    #   uint64 length
    #   uint32 length_mcrc32
    #   byte   raw_bytes[length]
    #   uint32 raw_bytes_mcrc32
    # https://github.com/tensorflow/tensorflow/blob/v2.12.0/tensorflow/tsl/lib/io/record_writer.cc#L104
    length_uint64 = len(raw_bytes).to_bytes(8, BYTE_ORDER)
    length_mcrc32 = mask_crc32(binascii.crc32(length_uint64))
    length_mcrc32_uint32 = length_mcrc32.to_bytes(4, BYTE_ORDER)
    data_mcrc32 = mask_crc32(binascii.crc32(raw_bytes))
    data_mcrc32_uint32 = data_mcrc32.to_bytes(4, BYTE_ORDER)
    tfrecord_bytes = b''.join(
        [length_uint64, length_mcrc32_uint32, raw_bytes, data_mcrc32_uint32]
    )
    with threading.Lock() if thread_safe else contextlib.nullcontext():
      fp.write(tfrecord_bytes)

  try:
    fp = open(path, 'ab' if append_mode else 'wb')
    yield partial(write_fn, fp)
  finally:
    fp.close()
