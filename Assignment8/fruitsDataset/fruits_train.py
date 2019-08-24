
# Assignment 7
# UCID: JA573

from keras.models import Sequential
from keras.models import load_model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.utils import np_utils, generic_utils, to_categorical
# from keras import backend as K 
# K.set_image_dim_ordering('th')
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16 
from keras.models import Model
import numpy as np
import keras
import sys
import os

batch_size = 16
nb_classes = 101
nb_epoch = 1

img_channels = 3
img_rows = 100
img_cols = 100

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(
        sys.argv[1],
        target_size=(100, 100),
        color_mode='rgb',
        batch_size=16,
        class_mode='categorical')

b_model = VGG16(weights = 'imagenet', include_top = False, input_shape = (100,100,3))
op_layer = b_model.output
f_layer = Flatten()(op_layer)
d_layer = Dense(101)(f_layer)
a_layer = Activation('softmax')(d_layer)
model = Model(inputs = b_model.input, output = a_layer)

model = load_model(sys.argv[2])

for layer in model.layers[:18]:
	layer.trainable = False

model.compile(loss='categorical_crossentropy',
              optimizer='adamax',
              metrics=['accuracy'])

model.summary()

model.fit_generator(
        train_generator,
        steps_per_epoch=100,
        epochs=1)
model.save(sys.argv[2])

print("Saved model to disk")

