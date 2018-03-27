"""
Class for managing our data.
"""
import csv
import numpy as np
import cv2
import os.path
import random
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical

class DataSet():
    def __init__(self, num_of_snip=5, opt_flow_len=10, image_shape=(224, 224), class_limit=None):
        """Constructor.
        opt_flow_len = (int) the number of optical flow frames to consider
        class_limit = (int) number of classes to limit the data to.
            None = no limit.
        """
        self.opt_flow_len = opt_flow_len
        self.num_of_snip = num_of_snip
        self.class_limit = class_limit
        self.image_shape = image_shape

        # Get the data.
        self.data_list = self.get_data_list()

        # Get the classes.
        self.classes = self.get_classes()

        # Now do some minor data cleaning
        self.data_list = self.clean_data_list()

    @staticmethod
    def get_data_list():
        """Load our data list from file."""
        with open(os.path.join('/data', 'data_list.csv'), 'r') as fin:
            reader = csv.reader(fin)
            data_list = list(reader)

        return data_list

    def clean_data_list(self):
        data_list_clean = []
        for item in self.data_list:
            if item[1] in self.classes:
                data_list_clean.append(item)

        return data_list_clean

    def get_classes(self):
        """Extract the classes from our data. If we want to limit them,
        only return the classes we need."""
        classes = []
        for item in self.data_list:
            if item[1] not in classes:
                classes.append(item[1])

        # Sort them.
        classes = sorted(classes)

        # Return.
        if self.class_limit is not None:
            return classes[:self.class_limit]
        else:
            return classes

    def get_class_one_hot(self, class_str):
        """Given a class as a string, return its number in the classes
        list. This lets us encode and one-hot it for training."""

        # Encode it first.
        label_encoded = self.classes.index(class_str)

        # Now one-hot it.
        label_hot = to_categorical(label_encoded, len(self.classes))

        return label_hot

def get_generators(data, image_shape=(224, 224), batch_size=32):
    train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            horizontal_flip=True,
            rotation_range=10.,
            width_shift_range=0.2,
            height_shift_range=0.2)

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
            os.path.join('/data', 'train'),
            target_size=image_shape,
            batch_size=batch_size,
            classes=data.classes,
            class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(
            os.path.join('/data', 'test'),
            target_size=image_shape,
            batch_size=batch_size,
            classes=data.classes,
            class_mode='categorical')

    return train_generator, validation_generator

