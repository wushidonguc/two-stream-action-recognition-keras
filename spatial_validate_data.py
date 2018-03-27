"""
Class for managing our data.
"""
import csv
import numpy as np
import os.path
import random
import threading
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
import cv2

class DataSet():
    def __init__(self, class_limit=None, image_shape=(224, 224), original_image_shape=(341, 256), n_snip=5, opt_flow_len=10, batch_size=16):
        """Constructor.
        opt_flow_len = (int) the number of optical flow frames to consider
        class_limit = (int) number of classes to limit the data to.
            None = no limit.
        """
        self.class_limit = class_limit
        self.image_shape = image_shape
        self.original_image_shape = original_image_shape
        self.n_snip = n_snip
        self.opt_flow_len = opt_flow_len
        self.batch_size = batch_size

        self.static_frame_path = os.path.join('/data', 'test')
        self.opt_flow_path = os.path.join('/data', 'opt_flow')

        # Get the data.
        self.data_list = self.get_data_list()

        # Get the classes.
        self.classes = self.get_classes()

        # Now do some minor data cleaning
        self.data_list = self.clean_data_list()

        # Get the right dataset for the generator.
        train, test = self.split_train_test()
        self.data_list = test
        
        # number of batches in 1 epoch
        self.n_batch = len(self.data_list) // self.batch_size

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
        """Extract the classes from our data, '\n'. If we want to limit them,
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

        assert label_hot.shape[0] == len(self.classes)

        return label_hot

    def split_train_test(self):
        """Split the data into train and test groups."""
        train = []
        test = []
        for item in self.data_list:
            if item[0] == 'train':
                train.append(item)
            else:
                test.append(item)
        return train, test

    def validation_generator(self):
        """Return a generator of optical frame stacks that we can use to test."""

        print("\nCreating validation generator with %d samples.\n" % len(self.data_list))

        idx = 0
        while 1:
            idx = idx % self.n_batch
            print("\nGenerating batch number {0}/{1} ...".format(idx + 1, self.n_batch))
            idx += 1
            
            X_batch = []
            y_batch = []

            # Get a list of batch-size samples.
            batch_list = self.data_list[idx * self.batch_size: (idx + 1) * self.batch_size]

            for row in batch_list:
                # Get the stacked optical flows from disk.
                X = self.get_static_frame(row)
                
                # Get the corresponding labels
                y = self.get_class_one_hot(row[1])
                y = np.array(y)
                y = np.squeeze(y)

                X_batch.append(X)
                y_batch.append(y)

            X_batch = np.array(X_batch)
            y_batch = np.array(y_batch)

            yield X_batch, y_batch

    def get_static_frame(self, row):
        static_frames = []

        static_frame_dir = os.path.join(self.static_frame_path, row[1], row[2])
        opt_flow_dir_x = os.path.join(self.opt_flow_path, 'u', row[2])
        opt_flow_dir_y = os.path.join(self.opt_flow_path, 'v', row[2])

        # temporal parameters
        total_frames = len(os.listdir(opt_flow_dir_x))
        if total_frames - self.opt_flow_len + 1 < self.n_snip:
            loop = True
            start_frame_window_len = 1
        else:
            loop = False
            start_frame_window_len = (total_frames - self.opt_flow_len + 1) // self.n_snip # starting frame selection window length

        # loop over snippets
        for i_snip in range(self.n_snip):
            if loop:
                start_frame = i_snip % (total_frames - self.opt_flow_len + 1) + 1
            else:
                start_frame = int(0.5 * start_frame_window_len + 0.5) + start_frame_window_len * i_snip

            # Get the static frame
            static_frame = cv2.imread(static_frame_dir + '-%04d' % start_frame + '.jpg')
            static_frame = static_frame * 1.0 / 255.0
            static_frame = cv2.resize(static_frame, self.image_shape)

            static_frames.append(static_frame)

        return np.array(static_frames)

