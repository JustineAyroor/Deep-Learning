# Assignmet 6
# Justine Ayroor UCID: ja573

from keras.models import load_model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.utils import np_utils, generic_utils, to_categorical
from keras import backend as K 
K.set_image_dim_ordering('th')
import numpy as np
import keras
import sys

nb_classes = 10
X_test = np.load(sys.argv[1])
Y_test = np.load(sys.argv[2])

print('X_test shape:', X_test.shape)
print(X_test.shape[0], 'test samples')

print('Y_test shape:', Y_test.shape)

Y_test = to_categorical(Y_test, nb_classes)#convert label into one-hot vector

print('Y_test shape:', Y_test.shape)

model = load_model(sys.argv[3])

model.compile(loss='categorical_crossentropy',
              optimizer='adamax',
              metrics=['accuracy'])
scores = model.evaluate(X_test, Y_test)
print("Accuracy of the Model:")
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
print("%s: %.2f%%" % ("Error: ", 100 - scores[1]*100))
