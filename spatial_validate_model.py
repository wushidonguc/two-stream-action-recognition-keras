import numpy as np
from keras.applications.inception_v3 import InceptionV3
from keras.models import Sequential, load_model, Model
from keras.layers import Input, Average, GlobalAveragePooling2D
from keras.layers import TimeDistributed, GlobalAveragePooling1D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam
from keras.layers.normalization import BatchNormalization

class ResearchModels():
    def __init__(self, nb_classes, n_snip, opt_flow_len, image_shape = (224, 224), saved_weights=None):
        """
        `nb_classes` = the number of classes to predict
        `opt_flow_len` = the length of optical flow frames
        `image_shape` = shape of image frame
        `saved_model` = the path to a saved Keras model to load
        """
        self.nb_classes = nb_classes
        self.n_snip = n_snip
        self.opt_flow_len = opt_flow_len
        self.saved_weights = saved_weights

        self.input_shape = (image_shape[0], image_shape[1], 3)
        self.input_shape_multi = (self.n_snip, image_shape[0], image_shape[1], 3)

        # Set the metrics. Only use top k if there's a need.
        metrics = ['accuracy']
        if self.nb_classes >= 10:
            metrics.append('top_k_categorical_accuracy')

        self.model = self.cnn_spatial_multi()

        optimizer = Adam()

        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=metrics)

    # CNN model for the spatial stream with multiple inputs
    def cnn_spatial_multi(self):
        # shared cnn_spatial model
        cnn_spatial = self.cnn_spatial()
        cnn_spatial.load_weights(self.saved_weights)
        for layer in cnn_spatial.layers:
            layer.trainable = False

        # building inputs and output
        model = Sequential()
        model.add(TimeDistributed((cnn_spatial), input_shape=self.input_shape_multi))
        model.add(GlobalAveragePooling1D())

        return model

    # CNN model for the spatial stream
    def cnn_spatial(self, weights='imagenet'):
        # create the base pre-trained model
        base_model = InceptionV3(weights=weights, include_top=False)
    
        # add a global spatial average pooling layer
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        # let's add a fully-connected layer
        x = Dense(1024, activation='relu')(x)
        # and a logistic layer
        predictions = Dense(self.nb_classes, activation='softmax')(x)
    
        # this is the model we will train
        model = Model(inputs=base_model.input, outputs=predictions)

        return model

