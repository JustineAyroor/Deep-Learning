# Assignmet 8
# Justine Ayroor UCID: ja573

from keras.models import load_model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.utils import np_utils, generic_utils, to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16 
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import keras
import sys

nb_classes = 5
model = load_model(sys.argv[2])

test_datagen = ImageDataGenerator(rescale=1./255,shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

validation_generator = test_datagen.flow_from_directory(
        sys.argv[1],
        target_size=(150, 150),
        batch_size=16,
	color_mode = 'rgb',
        class_mode='categorical')


model.compile(loss='categorical_crossentropy',
              optimizer='adamax',
              metrics=['accuracy'])

scores = model.evaluate_generator(validation_generator, steps = 1)

print("Accuracy of the Model:")
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
print("%s: %.2f%%" % ("Error: ", 100 - scores[1]*100))
