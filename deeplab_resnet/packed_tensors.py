from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


__all__ = [
    "PackedTensors",
]


class PackedTensors(object):
  """Packed representation of compressed tensors.

  This class can pack and unpack several tensor values into a single string. It
  can also optionally store a model identifier.

  The tensors currently must be rank 1 (vectors) and either have integer or
  string type.
  """

  def __init__(self, string=None):
    self._example = tf.train.Example()
    if string:
      self.string = string

  @property
  def model(self):
    """A model identifier."""
    buf = self._example.features.feature["MD"].bytes_list.value[0]
    return buf.decode("ascii")

  @model.setter
  def model(self, value):
    self._example.features.feature["MD"].bytes_list.value[:] = [
        value.encode("ascii")]

  @model.deleter
  def model(self):
    del self._example.features.feature["MD"]

  @property
  def string(self):
    """The string representation of this object."""
    return self._example.SerializeToString()

  @string.setter
  def string(self, value):
    self._example.ParseFromString(value)

  def pack(self, tensors, arrays):
    """Packs `Tensor` values into this object."""
    if len(tensors) != len(arrays):
      raise ValueError("`tensors` and `arrays` must have same length.")
    i = 1
    for tensor, array in zip(tensors, arrays):
      feature = self._example.features.feature[chr(i)]
      feature.Clear()
      if array.ndim != 1:
        raise RuntimeError("Unexpected tensor rank: {}.".format(array.ndim))
      if tensor.dtype.is_integer:
        feature.int64_list.value[:] = array
      elif tensor.dtype == tf.string:
        feature.bytes_list.value[:] = array
      else:
        raise RuntimeError(
            "Unexpected tensor dtype: '{}'.".format(tensor.dtype))
      i += 1
    # Delete any remaining, previously set arrays.
    while chr(i) in self._example.features.feature:
      del self._example.features.feature[chr(i)]
      i += 1

  def unpack(self, tensors):
    """Unpacks `Tensor` values from this object."""
    arrays = []
    for i, tensor in enumerate(tensors):
      feature = self._example.features.feature[chr(i + 1)]
      #np_dtype = tensor.dtype.as_numpy_dtype
      dtype = tensor.dtype
      if dtype.is_integer:
        arrays.append(tf.constant(np.array(feature.int64_list.value), dtype=dtype))
        #arrays.append(np.array(feature.int64_list.value, dtype=np_dtype))
      elif dtype == tf.string:
        arrays.append(tf.constant(np.array(feature.bytes_list.value), dtype=dtype))
        #arrays.append(np.array(feature.bytes_list.value, dtype=np_dtype))
      else:
        raise RuntimeError(
            "Unexpected tensor dtype: '{}'.".format(dtype))
    return arrays