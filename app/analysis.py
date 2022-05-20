import numpy as np
import tensorflow as tf

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
assert x_train.shape == (60000, 28, 28)
assert x_test.shape == (10000, 28, 28)
assert y_train.shape == (60000,)
assert y_test.shape == (10000,)

classes = np.unique(y_train)
x_train = np.reshape(x_train, (60000, 784))

print('Classes:', classes)
print("Features' shape:", x_train.shape)
print("Target's shape:", y_train.shape)
print(f'min: {np.amin(x_train)}, max: {np.amax(x_train)}')
