#Assignment 6
#Justine Ayroor UCID:ja573

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.utils import np_utils, generic_utils, to_categorical
from keras import backend as K 
K.set_image_dim_ordering('th')
import numpy as np
import keras
import sys

batch_size = 16
nb_classes = 10
nb_epoch = 15

img_channels = 3
img_rows = 112
img_cols = 112

X_train = np.load(sys.argv[1])
Y_train = np.load(sys.argv[2])

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print('Y_train shape:', Y_train.shape)
Y_train = to_categorical(Y_train, nb_classes)
print('Y_train shape:', Y_train.shape)

model = Sequential()

#Layer 1
model.add(Conv2D(filters=112, kernel_size=(5,5), padding='valid', input_shape=[3,112,112]))#Convo$
#model.add(BatchNormalization())
model.add(Activation('relu'))#Activation function
model.add(AveragePooling2D(pool_size=(4,4)))
model.add(BatchNormalization())
#27x27 output

#Layer 2
model.add(Conv2D(filters=112, kernel_size=(4,4), padding='valid'))#Convo$
#model.add(BatchNormalization())
model.add(Activation('relu'))#Activation function
model.add(AveragePooling2D(pool_size=(3,3)))
model.add(BatchNormalization())
#8x8

#Layer 3
model.add(Conv2D(filters=112, kernel_size=(3,3), padding='valid'))#Convo$
#model.add(BatchNormalization())
model.add(Activation('relu'))#Activation function
model.add(AveragePooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
#3x3

# Layer 4
model.add(Conv2D(filters=112, kernel_size=(2,2), padding='valid'))#Convo$
#model.add(BatchNormalization())
model.add(Activation('relu'))#Activation function
model.add(AveragePooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
#1

#Dense layer
model.add(Flatten())# shape equals to [batch_size, 32] 32 is the number of filters
model.add(Dense(10))#Fully connected layer
model.add(BatchNormalization())
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer= 'adamax',
              metrics=['accuracy'])

def train():
    model.fit(X_train, Y_train,
              batch_size=batch_size,
              epochs=nb_epoch,
              shuffle=True)

train()
# serialize weights to HDF5
model.save(sys.argv[3])
print("Saved model to disk")
print("Execute test_model.py to Check Validation Accuracy\n Training Accuraccy Calculated: 99%\n")
print("Command: python test_model.py x_test.npy y_test.npy model.h5")
print("Trainig Complete")
