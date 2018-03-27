import numpy as np
from keras.models import Sequential, Model, load_model
from keras.layers import Input, average, TimeDistributed, GlobalAveragePooling1D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam
from keras.layers.normalization import BatchNormalization

class Research_Model():
    def __init__(self, nb_classes, n_snip, opt_flow_len, image_shape = (224, 224), saved_weights=None):
        """
        `nb_classes` = the number of classes to predict
        `opt_flow_len` = the length of optical flow frames
        `image_shape` = shape of image frame
        `saved_weights` = the path to a saved Keras weights to load
        """
        self.nb_classes = nb_classes
        self.n_snip = n_snip
        self.opt_flow_len = opt_flow_len
        self.image_shape = image_shape
        self.saved_weights = saved_weights

        self.input_shape = (image_shape[0], image_shape[1], opt_flow_len * 2)
        self.input_shape_multi = (self.n_snip, self.image_shape[0], self.image_shape[1], self.opt_flow_len * 2)

        # Set the metrics. Only use top k if there's a need.
        metrics = ['accuracy']
        if self.nb_classes >= 10:
            metrics.append('top_k_categorical_accuracy')

        self.model = self.cnn_temporal_multi()

        # Optimizer
        optimizer = SGD(lr=0.01, momentum=0.9, nesterov=True)

        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=metrics)

    # CNN model for the temporal stream with multiple inputs
    def cnn_temporal_multi(self):
        # shared cnn_temopral model
        cnn_temporal = self.cnn_temporal()
        cnn_temporal.load_weights(self.saved_weights)
        for layer in cnn_temporal.layers:
            layer.trainable = False

        # building inputs and output
        model = Sequential()
        model.add(TimeDistributed((cnn_temporal), input_shape=self.input_shape_multi))
        model.add(GlobalAveragePooling1D())

        return model

    # CNN model for the temporal stream
    def cnn_temporal(self):
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
        model.add(Dropout(0.5))

        #full7
        model.add(Dense(2048, activation='relu'))
        model.add(Dropout(0.5))

        #softmax
        model.add(Dense(self.nb_classes, activation='softmax'))

        return model
