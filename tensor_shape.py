# Taken from effectivemachinelearning.com for study purpose

import tensorflow as tf
a = tf.placeholder(tf.float32, [None, 128])

static_shape = a.shape.as_list() # return static shape
dynamic_shape = tf.shape(a)

# the static shape can be set with the following
a.set_shape([32, 128]) # static shape is [32, 128]
a.set_shape([None, 128]) # first dimension of a is determined dynamically
# dynamically stands during runtime

# we can also reshape a given tensor dynamically using tf.reshape function

# General purpose reshape function can be written as follows:
import tensorflow as tf
import numpy as np

def reshape(tensor, dim_list):
  shape = get_shape(tensor)
  dims_prod = []
  
  for dims in dims_list:
    if isinstance(dims, int):
      dims_prod.append(shape[dims])
    elif all([isinstance(shape[d], int) for d in dims]):
      dims_prod.append(np.prod([shpe[d] for d in dims]))
    else:
      dims_prod.append(tf.reduce_prod([shape[d]for d in dims]))
   tensor = tf.reshape(tensor, dims_prod)
   return tensor 
