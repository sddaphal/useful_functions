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

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics='accuracy')
model.summary()

# Load the mnist dataset for the problem in hand
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# Normalize the input data for the better results
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train.astype(np.float32)
x_test =  x_test.astype(np.float32)

_FAST_TRAIN = False
_EPOCHS = 5

if _FAST_TRAIN:
  _EPOCHS = 1
  _TRAINING_DATA_COUNT = 1000
  x_train = x_train[:_TRAINING_DATA_COUNT]
  y_train = y_train[:_TRAINING_DATA_COUNT]
  
model.fit(x_train, y_train, epochs=_EPOCHS)
model.evaluate(x_test, y_test, verbose=0)

