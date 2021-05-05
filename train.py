#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# Importing libraries
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

# Constants
WORKING_DIR = "./working_directory"
TRAINING_DIR = "dataset/training_set"
TESTING_DIR = "dataset/test_set"
PREDICTING_DIR = "dataset/prediction"
EPOCHS = 60


# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (150, 150, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))



# 2nd layer of convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (150, 150, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# 3rd layer of convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (150, 150, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 128, activation = 'relu'))

classifier.add(Dense(units = 12, activation = 'softmax'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)


training_set = train_datagen.flow_from_directory(TRAINING_DIR,
                                                 target_size = (150, 150),
                                                 batch_size = 16,
                                                 #classes=['white_pawn','black_pawn','black_knight'],
                                                 shuffle=True,
                                                 #seed=42,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory(TESTING_DIR,
                                            target_size = (150, 150),
                                            batch_size = 16,
                                            #classes=['white_pawn','black_pawn','black_knight'],
                                            #shuffle=False,
                                            #seed=42,
                                            class_mode = "categorical")
classifier.fit_generator(training_set,
                         steps_per_epoch = 40,
                         epochs = EPOCHS,
                         validation_data = test_set,
                         validation_steps = 14)


classifier.save('./model/')
































