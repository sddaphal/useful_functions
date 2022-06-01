import numpy as np
import tensorflow as tf

# creates the representative dataset useful for uint8/int8 qunatization options
def representative_dataset():
  for _ in range(100):
    data = np.random.rand(1, 28, 28)
    yield [data.astype(np.float32)]
    
model = tf.keras.models.Sequential([
  tf.keras.layers.Input(shape=(28,28), name='input'),
  tf.keras.layers.LSTM(20, time_major=False, return_sequences=True),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(10, activation='softmax', name='output')
])
