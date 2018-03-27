"""
Train our temporal-stream CNN on optical flow frames.
"""
from spatial_validate_model import ResearchModels
from spatial_validate_data import DataSet, keras_validation_generator
import time
import os.path
from os import makedirs

def test_1epoch(class_limit=None, n_snip=5, opt_flow_len=10, image_shape=(224, 224), original_image_shape=(341, 256), batch_size=16, saved_weights=None):

    print("\nValidating for weights: %s\n" % saved_weights)

    # Get the data and process it.
    data = DataSet(class_limit, image_shape, original_image_shape, n_snip, opt_flow_len, batch_size)

    # Get the generator.
    val_generator = data.validation_generator()
    steps = data.n_batch

    # Get the model.
    spatial_cnn = ResearchModels(nb_classes=len(data.classes), n_snip=n_snip, opt_flow_len=opt_flow_len, image_shape=image_shape, saved_weights=saved_weights)

    # Evaluate the model!
#    results = temporal_cnn.model.evaluate_generator(generator=val_generator, steps=steps)
#    print(results)
#    print(temporal_cnn.model.metrics_names)
    
    spatial_cnn.model.fit_generator(generator=val_generator, steps_per_epoch=steps, max_queue_size=1)
    print('Finished validation of weights:', saved_weights)

def main():

    """These are the main training settings. Set each before running this file."""
    "=============================================================================="
    saved_weights = None # weights file
    class_limit = None  # int, can be 1-101 or None
    n_snip = 1 # number of chunks from each video used for testing
    opt_flow_len = 10 # number of optical flow frames used
    image_shape = (224, 224)
    batch_size = 1024
    "=============================================================================="

    test_1epoch(class_limit=class_limit, n_snip=n_snip, opt_flow_len=opt_flow_len, image_shape=image_shape, batch_size=batch_size, saved_weights=saved_weights)

if __name__ == '__main__':
    main()
