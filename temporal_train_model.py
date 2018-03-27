import numpy as np
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam
from keras.layers.normalization import BatchNormalization

class ResearchModels():
    def __init__(self, nb_classes, num_of_snip, opt_flow_len, image_shape = (224, 224), saved_model=None):
        """
        `nb_classes` = the number of classes to predict
        `opt_flow_len` = the length of optical flow frames
        `image_shape` = shape of image frame
        `saved_model` = the path to a saved Keras model to load
        """
        self.num_of_snip = num_of_snip
        self.opt_flow_len = opt_flow_len
        self.load_model = load_model
        self.saved_model = saved_model
        self.nb_classes = nb_classes
        print("Number of classes:")
        print(self.nb_classes)

        # Set the metrics. Only use top k if there's a need.
        metrics = ['accuracy']
        if self.nb_classes >= 10:
            metrics.append('top_k_categorical_accuracy')

        if self.saved_model is not None:
            print("Loading model %s" % self.saved_model)
            self.model = load_model(self.saved_model)
        else:
            print("Loading CNN model for the temporal stream.")
            self.input_shape = (image_shape[0], image_shape[1], opt_flow_len * 2 * self.num_of_snip)
            self.model = self.cnn_temporal()

        optimizer = SGD(lr=1e-2, momentum=0.9, nesterov=True)

        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=metrics)

        print(self.model.summary())

    # CNN model for the temporal stream
    def cnn_temporal(self):
        print("Input shape:")
        print(self.input_shape)
        print("Numer of classes:")
        print(self.nb_classes)

        #model
        model = Sequential()

        #conv1
        model.add(Conv2D(96, (7, 7), strides=2, padding='same', input_shape=self.input_shape))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        #conv2
        model.add(Conv2D(256, (5, 5), strides=2, padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        #conv3
        model.add(Conv2D(512, (3, 3), strides=1, activation='relu', padding='same'))

        #conv4
        model.add(Conv2D(512, (3, 3), strides=1, activation='relu', padding='same'))

        #conv5
        model.add(Conv2D(512, (3, 3), strides=1, activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        #full6
        model.add(Flatten())
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.9))

        #full7
        model.add(Dense(2048, activation='relu'))
        model.add(Dropout(0.9))

        #softmax
        model.add(Dense(self.nb_classes, activation='softmax'))

        return model
