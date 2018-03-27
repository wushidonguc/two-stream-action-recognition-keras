import numpy as np
from keras.applications.inception_v3 import InceptionV3
from keras.models import Sequential, load_model, Model
from keras.layers import Input, average, concatenate, GlobalAveragePooling2D
from keras.layers import TimeDistributed, GlobalAveragePooling1D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam
from keras.layers.normalization import BatchNormalization

class ResearchModels():
    def __init__(self, nb_classes, n_snip, opt_flow_len, image_shape = (224, 224), saved_model=None, saved_temporal_weights=None, saved_spatial_weights=None):
        """
        `nb_classes` = the number of classes to predict
        `opt_flow_len` = the length of optical flow frames
        `image_shape` = shape of image frame
        `saved_model` = the path to a saved Keras model to load
        """
        self.nb_classes = nb_classes
        self.n_snip = n_snip
        self.opt_flow_len = opt_flow_len
        self.load_model = load_model
        self.saved_model = saved_model
        self.saved_temporal_weights = saved_temporal_weights
        self.saved_spatial_weights = saved_spatial_weights

        self.input_shape_spatial = (image_shape[0], image_shape[1], 3)
        self.input_shape_temporal = (image_shape[0], image_shape[1], opt_flow_len * 2)
        self.input_shape_spatial_multi = (self.n_snip, image_shape[0], image_shape[1], 3)
        self.input_shape_temporal_multi = (self.n_snip, image_shape[0], image_shape[1], opt_flow_len * 2)

        # Set the metrics. Only use top k if there's a need.
        metrics = ['accuracy']
        if self.nb_classes >= 10:
            metrics.append('top_k_categorical_accuracy')

        # Load model
        # If saved fuse model exists, directly load
        if self.saved_model is not None: 
            print("\nLoading model %s" % self.saved_model)
            self.model = load_model(self.saved_model)
        # Otherwise build the model and load weights for both streams
        else: 
            print("\nLoading the two-stream model...")
            self.model = self.two_stream_fuse()

        optimizer = Adam()
#        optimizer = SGD(lr=0.01, momentum=0.9, nesterov=True)

        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=metrics)

    # Two-stream fused model
    def two_stream_fuse(self):
        # spatial stream (frozen)
        cnn_spatial_multi = self.cnn_spatial_multi()

        # temporal stream (frozen)
        cnn_temporal_multi = self.cnn_temporal_multi()

        # fused by taking average
        outputs = average([cnn_spatial_multi.output, cnn_temporal_multi.output])

        model = Model([cnn_spatial_multi.input, cnn_temporal_multi.input], outputs)

        return model

    # CNN model for the temporal stream with multiple inputs
    def cnn_spatial_multi(self):
        # spatial stream (frozen)
        cnn_spatial = self.cnn_spatial()
        if self.saved_spatial_weights is None:
            print("[ERROR] No saved_spatial_weights weights file!")
        else:
            cnn_spatial.load_weights(self.saved_spatial_weights)
        for layer in cnn_spatial.layers:
            layer.trainable = False

        # building inputs and output
        model = Sequential()
        model.add(TimeDistributed((cnn_spatial), input_shape=self.input_shape_spatial_multi))
        model.add(GlobalAveragePooling1D())

        return model

    # CNN model for the temporal stream with multiple inputs
    def cnn_temporal_multi(self):
        # spatial stream (frozen)
        cnn_temporal = self.cnn_temporal()
        if self.saved_temporal_weights is None:
            print("[ERROR] No saved_temporal_weights weights file!")
        else:
            cnn_temporal.load_weights(self.saved_temporal_weights)
        for layer in cnn_temporal.layers:
            layer.trainable = False

        # building inputs and output
        model = Sequential()
        model.add(TimeDistributed((cnn_temporal), input_shape=self.input_shape_temporal_multi))
        model.add(GlobalAveragePooling1D())

        return model

    # CNN model for the spatial stream
    def cnn_spatial(self):
        base_model = InceptionV3(weights='imagenet', include_top=False)
    
        # add a global spatial average pooling layer
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        # let's add a fully-connected layer
        x = Dense(1024, activation='relu')(x)
        # and a logistic layer
        predictions = Dense(self.nb_classes, activation='softmax')(x)
    
        model = Model(inputs=base_model.input, outputs=predictions)
        return model

    # CNN model for the temporal stream
    def cnn_temporal(self):
        #model
        model = Sequential()

        #conv1
        model.add(Conv2D(96, (7, 7), strides=2, padding='same', input_shape=self.input_shape_temporal))
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

