import numpy as np
import tensorflow as tf

def representative_dataset():
  for _ in range(100):
    data = np.random.rand(1, 28, 28)
    yield [data.astype(np.float32)]
    
model = tf.keras.models.Sequential([
  tf.keras.layers.Input(shape=(28,28), name='input'),
  

])
