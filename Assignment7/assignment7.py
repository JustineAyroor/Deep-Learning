from keras.models import Sequential
from keras.models import load_model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.utils import np_utils, generic_utils, to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
import numpy as np
import keras
import sys

batch_size = 32
nb_classes = 10
nb_epoch = 3

img_channels = 3
img_rows = 32
img_cols = 32

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)


test_datagen = ImageDataGenerator(rescale=1./255)
# model = load_model(sys.argv[1])
train_generator = train_datagen.flow_from_directory(
        'train',
        target_size=(256, 256),
        color_mode='rgb',
        batch_size=32,
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        'val',
        target_size=(256, 256),
        color_mode='rgb',
        batch_size=32,
        class_mode='categorical')
model = Sequential()

#Layer 1
model.add(Conv2D(filters=32, kernel_size=(3,3), padding='valid', input_shape=[32,32,3]))#Convo$
model.add(Activation('relu'))#Activation function
model.add(AveragePooling2D(pool_size=(2, 2)))
#14x14 output

#Layer 2
model.add(Conv2D(filters=32, kernel_size=(3,3), padding='valid'))#Convo$
model.add(Activation('relu'))#Activation function
model.add(AveragePooling2D(pool_size=(2, 2)))
#6x6
#keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)

#Layer 3
model.add(Conv2D(filters=32, kernel_size=(3,3), padding='valid'))#Convo$
model.add(Activation('relu'))#Activation function
model.add(AveragePooling2D(pool_size=(2, 2)))
#2x2

#Dense layer
model.add(Flatten())# shape equals to [batch_size, 32] 32 is the number of filters
model.add(Dense(10))#Fully connected layer
model.add(Activation('softmax'))

#opt = keras.optimizers.rmsprop(lr=0.001, decay=1e-6)
opt = keras.optimizers.SGD(lr=0.001, decay=1e-6)

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])


def train():
    model.fit_generator(
        train_generator,
        steps_per_epoch=2000,
        epochs=nb_epoch,
        validation_data=validation_generator,
        validation_steps=800)

train()

# serialize weights to HDF5
model.save(sys.argv[1])

print("Saved model to disk")
