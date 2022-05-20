import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
assert x_train.shape == (60000, 28, 28)
assert x_test.shape == (10000, 28, 28)
assert y_train.shape == (60000,)
assert y_test.shape == (10000,)

x_train = np.reshape(x_train, (60000, 784))[:6000]
y_train = y_train[:6000]
classes = np.unique(y_train)

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.3, random_state=40)
proportions = pd.Series(y_train).value_counts(normalize=True)

print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)
print('Proportion of samples per class in train set:', proportions, sep='\n')
