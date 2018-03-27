"""
Train our temporal-stream CNN on optical flow frames.
"""
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
from fuse_validate_model import ResearchModels
from fuse_validate_data import DataSet
import time
import os.path
from os import makedirs

def test_1epoch_fuse(
            class_limit=None, 
            n_snip=5,
            opt_flow_len=10,
            saved_model=None,
            saved_spatial_weights=None,
            saved_temporal_weights=None,
            image_shape=(224, 224),
            original_image_shape=(341, 256),
            batch_size=128,
            fuse_method='average'):

    print("class_limit = ", class_limit)

    # Get the data.
    data = DataSet(
            class_limit=class_limit,
            image_shape=image_shape,
            original_image_shape=original_image_shape,
            n_snip=n_snip,
            opt_flow_len=opt_flow_len,
            batch_size=batch_size
            )
    val_generator = data.validation_generator() # Get the validation generator
    steps = data.n_batch

    # Get the model.
    two_stream_fuse = ResearchModels(nb_classes=len(data.classes), n_snip=n_snip, opt_flow_len=opt_flow_len, image_shape=image_shape, saved_model=saved_model, saved_temporal_weights=saved_temporal_weights, saved_spatial_weights=saved_spatial_weights)

    # Evaluate!
    two_stream_fuse.model.fit_generator(generator=val_generator, steps_per_epoch=steps, max_queue_size=1)

def main():
    """These are the main training settings. Set each before running
    this file."""
    "=============================================================================="
    saved_spatial_weights = ''
    saved_temporal_weights = ''
    class_limit = None 
    n_snip = 19 # number of chunks used for each video
    opt_flow_len = 10 # number of optical flow frames used
    image_shape=(224, 224)
    original_image_shape=(341, 256)
    batch_size = 256
    fuse_method = 'average'
    "=============================================================================="

    test_1epoch_fuse(
            class_limit=class_limit, 
            n_snip=n_snip,
            opt_flow_len=opt_flow_len,
            saved_spatial_weights=saved_spatial_weights,
            saved_temporal_weights=saved_temporal_weights,
            image_shape=image_shape,
            original_image_shape=original_image_shape,
            batch_size=batch_size,
            fuse_method=fuse_method
            )

if __name__ == '__main__':
    main()
