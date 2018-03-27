"""
Class for managing our data.
"""
import csv
import numpy as np
import os.path
import random
import threading
from keras.utils import to_categorical
import cv2
#from keras.preprocessing import image

class threadsafe_iterator:
    def __init__(self, iterator):
        self.iterator = iterator
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return next(self.iterator)

def threadsafe_generator(func):
    """Decorator"""
    def gen(*a, **kw):
        return threadsafe_iterator(func(*a, **kw))
    return gen

class DataSet():
    def __init__(self, num_of_snip=1, opt_flow_len=10, image_shape=(224, 224), original_image_shape=(341, 256), class_limit=None):
        """Constructor.
        opt_flow_len = (int) the number of optical flow frames to consider
        class_limit = (int) number of classes to limit the data to.
            None = no limit.
        """
        self.opt_flow_len = opt_flow_len
        self.num_of_snip = num_of_snip
        self.class_limit = class_limit
        self.image_shape = image_shape
        self.original_image_shape = original_image_shape
        self.opt_flow_path = os.path.join('/data', 'opt_flow')

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

    @threadsafe_generator
    def stack_generator(self, batch_size, train_test, name_str="N/D"):
        """Return a generator of optical frame stacks that we can use to train on. There are
        a couple different things we can return:
        """
        # Get the right dataset for the generator.
        train, test = self.split_train_test()
        data_list = train if train_test == 'train' else test

        idx = 0

        print("\nCreating %s generator with %d samples.\n" % (train_test,
            len(data_list)))

        while 1:
            idx += 1
            print("Generator yielding batch No.%d" % idx)
            if(train_test == 'test'):
                print("Validating for job: %s" % name_str)
            X, y = [], []

            # Generate batch_size samples.
            for _ in range(batch_size):
                # Reset to be safe.
                stack = []

                # Get a random sample.
                row = random.choice(data_list)

                # Get the stacked optical flows from disk.
                stack = self.get_stacked_opt_flows(row, train_test)

                X.append(stack)
                y.append(self.get_class_one_hot(row[1]))

            X = np.array(X)
            y = np.array(y)
            y = np.squeeze(y)

            yield X, y

    def get_stacked_opt_flows(self, row, train_test, crop='corner', val_aug='center'):
        # crop options for training: corner, random
        # augmentation options for testing: resize, center

        opt_flow_stack = []
        opt_flow_dir_x = os.path.join(self.opt_flow_path, 'u', row[2])
        opt_flow_dir_y = os.path.join(self.opt_flow_path, 'v', row[2])

        # spatial parameters
        if train_test == 'train':
            if crop == 'random':
                # crop at center and four corners randomly for training
                left, top = random.choice([[0, 0], [0, self.original_image_shape[1] - self.image_shape[1]], [self.original_image_shape[0] - self.image_shape[0], 0], [self.original_image_shape[0] - self.image_shape[0], self.original_image_shape[1] - self.image_shape[1]], [int((self.original_image_shape[0] - self.image_shape[0]) * 0.5), int((self.original_image_shape[1] - self.image_shape[1]) * 0.5)]])
            else:
                # random crop for training set
                left = int((self.original_image_shape[0] - self.image_shape[0]) * random.random())
                top = int((self.original_image_shape[1] - self.image_shape[1]) * random.random())
        else:
            # crop at center for validation
            left = int((self.original_image_shape[0] - self.image_shape[0]) * 0.5)
            top = int((self.original_image_shape[1] - self.image_shape[1]) * 0.5)
        right = left + self.image_shape[0]
        bottom = top + self.image_shape[1]

        # temporal parameters
        total_frames = len(os.listdir(opt_flow_dir_x))
        win_len = (total_frames - self.opt_flow_len) // self.num_of_snip # starting frame selection window length
        if train_test == 'train':
            start_frame = int(random.random() * win_len) + 1
        else:
            start_frame = int(0.5 * win_len) + 1
        frames = [] # selected optical flow frames
        for i in range(self.num_of_snip):
            frames += range(start_frame + self.opt_flow_len * i, start_frame + self.opt_flow_len * (i + 1))

        if train_test == 'train' and random.random() > 0.5:
            flip = True
        else:
            flip = False

        # loop over frames
        for i_frame in frames:

            # horizontal components
            img = None # reset to be safe
            img = cv2.imread(opt_flow_dir_x + '/frame' + "%06d"%(i_frame) + '.jpg', 0)
            print(opt_flow_dir_x + '/frame' + "%06d"%(i_frame) + '.jpg')
            img = np.array(img)
            # mean substraction 
            img = img - np.mean(img)
            if train_test == 'train' or val_aug == 'center':
                # crop
                img = img[left : right, top : bottom]
            else:
                #resize
                img = cv2.resize(img, self.image_shape)
            img = img / 255. # normalize pixels 
            if flip:
                img = -img
            opt_flow_stack.append(img)

            # vertical components
            img2 = None # reset to be safe
            img2 = cv2.imread(opt_flow_dir_y + '/frame' + "%06d"%(i_frame) + '.jpg', 0)
            # mean substraction 
            img2 = np.array(img2)
            img2 = np.swapaxes(img2, 0, 1)
            img2 = img2 - np.mean(img2)
            if train_test == 'train' or val_aug == 'center':
                # crop
                img2 = img2[left : right, top : bottom]
            else:
                #resize
                img2 = cv2.resize(img2, self.image_shape)
            img2 = img2 / 255. # normalize pixels 
            opt_flow_stack.append(img2)

        opt_flow_stack = np.array(opt_flow_stack)
        opt_flow_stack = np.swapaxes(opt_flow_stack, 0, 1)
        opt_flow_stack = np.swapaxes(opt_flow_stack, 1, 2)

        # random horizontal flip for training sets
        if flip:
            opt_flow_stack = np.flip(opt_flow_stack, 0)

        return opt_flow_stack


