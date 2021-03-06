from linear import _Linear

import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variable_scope as vs
'''most of code are from  https://github.com/flrngel/sru-tensorflow/blob/master/sru.py, 
   and use GRUCell format in "tensorflow.python.ops.rnn_cell_impl.py"'''

class SRUCell(RNNCell):
  """Simple recurrent unit cell.
  The implementation is based on: https://arxiv.org/abs/1709.02755.
  """

  def __init__(self, num_units,
                activation=None, reuse=None):
    """Initialize the basic SRU cell.
    Args:
      num_units: int, The number of units in the LSTM cell.
      activation: Activation function of the inner states.  Default: `tanh`.
      reuse: (optional) Python boolean describing whether to reuse variables
        in an existing scope.  If not `True`, and the existing scope already has
        the given variables, an error is raised.
      When restoring from CudnnLSTM-trained checkpoints, must use
      CudnnCompatibleLSTMCell instead.
    """
    super(SRUCell, self).__init__(_reuse=reuse)

    self._num_units = num_units
    self._activation = activation or math_ops.tanh
    self._gate_linear = None
    self._bias_initializer = None
    self._kernel_initializer = None
    
  @property
  def state_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  def call(self, inputs, state):
    """Single recurrent unit cell (SRU).
    Args:
      inputs: `2-D` tensor with shape `[batch_size x input_size]`.
      state: `2-D` tensor with shape `[batch_size x _num_units]`
    Returns:
      A pair containing the new hidden state, and the new state.
    """
    sigmoid = math_ops.sigmoid
    # Parameters of gates are concatenated into one multiply for efficiency.
    c = state

    if self._bias_initializer is None:
      bias_ones = init_ops.constant_initializer(1.0, dtype=inputs.dtype)

    with vs.variable_scope(scope or type(self).__name__):
      self._gate_linear = _Linear(
          [inputs],
          3 * self._num_units,
          False,
          None,
          kernel_initializer=self._kernel_initializer)

      #add the biases
      b_f = vs.get_variable('b_f', [self._num_units],initializer=bias_ones)
      b_r = vs.get_variable('b_r', [self._num_units],initializer=bias_ones)

    x, f, r = array_ops.split(
          value=self._gate_linear([inputs]), num_or_size_splits=3, axis=1)

    f = sigmoid(f + b_f)
    r = sigmoid(r + b_r)

    new_c = f * c + (1 - f) * x
    # here should be new_h = r * self._activation(new_c)+(1-r)*inputs
    # but the shape of "(1-r)" and "inputs" are not match, so if you
    # use "new_h = r * self._activation(new_c)+(1-r)*inputs", then
    # it will raise "ValueError: Dimensions must be equal..."
    new_h = r * self._activation(new_c)

    return new_h, new_c
