"""
Train our temporal-stream CNN on optical flow frames.
"""
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger, LearningRateScheduler
from temporal_train_model import ResearchModels
from temporal_train_data import DataSet
import time
import os.path
from os import makedirs

def fixed_schedule(epoch):
    initial_lr = 1.e-2
    lr = initial_lr

    if epoch == 1389:
        lr = 0.1 * lr
    if epoch == 1944:
        lr = 0.1 * lr

    return lr

def train(num_of_snip=5, opt_flow_len=10, saved_model=None,
        class_limit=None, image_shape=(224, 224),
        load_to_memory=False, batch_size=32, nb_epoch=100, name_str=None):

    # Get local time.
    time_str = time.strftime("%y%m%d%H%M", time.localtime())

    if name_str == None:
        name_str = time_str

    # Callbacks: Save the model.
    directory1 = os.path.join('out', 'checkpoints', name_str)
    if not os.path.exists(directory1):
            os.makedirs(directory1)
    checkpointer = ModelCheckpoint(
            filepath=os.path.join(directory1,
                    '{epoch:03d}-{val_loss:.3f}.hdf5'),
            verbose=1,
            save_best_only=True)

    # Callbacks: TensorBoard
    directory2 = os.path.join('out', 'TB', name_str)
    if not os.path.exists(directory2):
            os.makedirs(directory2)
    tb = TensorBoard(log_dir=os.path.join(directory2))

    # Callbacks: Early stopper.
    early_stopper = EarlyStopping(monitor='loss', patience=100)

    # Callbacks: Save results.
    directory3 = os.path.join('out', 'logs', name_str)
    if not os.path.exists(directory3):
            os.makedirs(directory3)
    timestamp = time.time()
    csv_logger = CSVLogger(os.path.join(directory3, 'training-' + \
            str(timestamp) + '.log'))

    # Learning rate schedule.
    lr_schedule = LearningRateScheduler(fixed_schedule, verbose=0)

    print("class_limit = ", class_limit)
    # Get the data and process it.
    if image_shape is None:
        data = DataSet(
                num_of_snip=num_of_snip,
                opt_flow_len=opt_flow_len,
                class_limit=class_limit
                )
    else:
        data = DataSet(
                num_of_snip=num_of_snip,
                opt_flow_len=opt_flow_len,
                image_shape=image_shape,
                class_limit=class_limit,
                )

    # Get samples per epoch.
    # Multiply by 0.7 to attempt to guess how much of data.data is the train set.
    steps_per_epoch = (len(data.data_list) * 0.7) // batch_size

    if load_to_memory:
        # Get data.
        X, y = data.get_all_stacks_in_memory('train')
        X_test, y_test = data.get_all_stacks_in_memory('test')
    else:
        # Get generators.
        generator = data.stack_generator(batch_size, 'train')

        val_generator = data.stack_generator(batch_size, 'test', name_str=name_str)

    # Get the model.
    temporal_cnn = ResearchModels(nb_classes=len(data.classes), num_of_snip=num_of_snip, opt_flow_len=opt_flow_len, image_shape=image_shape, saved_model=saved_model)

    # Fit!
    if load_to_memory:
        # Use standard fit.
        temporal_cnn.model.fit(
                X,
                y,
                batch_size=batch_size,
                validation_data=(X_test, y_test),
                verbose=1,
                callbacks=[tb, early_stopper, csv_logger],
                epochs=nb_epoch)
    else:
        # Use fit generator.
        temporal_cnn.model.fit_generator(
                generator=generator,
                steps_per_epoch=steps_per_epoch,
                epochs=nb_epoch,
                verbose=1,
                callbacks=[tb, early_stopper, csv_logger, checkpointer, lr_schedule],
                validation_data=val_generator,
                validation_steps=1,
                max_queue_size=20,
                workers=1,
                use_multiprocessing=False)

def main():
    """These are the main training settings. Set each before running
    this file."""
    "=============================================================================="
    saved_model = None
    class_limit = None  # int, can be 1-101 or None
    num_of_snip = 1 # number of chunks used for each video
    opt_flow_len = 10 # number of optical flow frames used
    image_shape=(224, 224)
    load_to_memory = False  # pre-load the sequences into memory
    batch_size = 64
    nb_epoch = 2222
    name_str = None
    "=============================================================================="

    train(num_of_snip=num_of_snip, opt_flow_len=opt_flow_len, saved_model=saved_model,
            class_limit=class_limit, image_shape=image_shape,
            load_to_memory=load_to_memory, batch_size=batch_size,
            nb_epoch=nb_epoch, name_str=name_str)

if __name__ == '__main__':
    main()
