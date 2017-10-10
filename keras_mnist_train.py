""" Resources:
- https://github.com/tanmayb123/MNIST-CNN-in-Keras
- https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
"""
# pylint: disable=invalid-name

import os
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Flatten, Dense
from azureml.logging import get_azureml_logger

# initialize the logger
run_logger = get_azureml_logger()

# create the outputs folder where we will save the model file later
os.makedirs('./outputs', exist_ok=True)
os.listdir()
img_width, img_height = 28, 28

train_data_dir = '/tmp/data/mnist_png/training'
validation_data_dir = '/tmp/data/mnist_png/testing'

os.getcwd()
os.listdir(train_data_dir)
os.listdir(validation_data_dir)

num_training_samples = 60000
num_validation_samples = 10000

batch_size = 32
epochs_to_run = 4

# ** Model Begins **
model = Sequential()

model.add(Conv2D(16, (5, 5), input_shape=(img_width, img_height, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (5, 5)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())    # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(1000))
model.add(Activation('relu'))

model.add(Dense(10))
model.add(Activation('softmax'))

#model.summary()         # print model summary
# ** Model Ends **

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

history = model.fit_generator(
        train_generator,
        steps_per_epoch=num_training_samples // batch_size,
        epochs=epochs_to_run,
        validation_data=validation_generator,
        validation_steps=num_validation_samples // batch_size)

run_logger.log("Accuracy", history.history['acc'])
run_logger.log("Loss", history.history['loss'])

model.save_weights('./outputs/mnistneuralnet.h5')
