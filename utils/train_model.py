import os
import sys
# model build
from keras import utils
from keras import optimizers
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.utils.vis_utils import plot_model
from contextlib import redirect_stdout
# dataframes
import pandas as pd

sys.path.append( '../' )
# load fire dataset
from utils.load_fire_dataset import load_cairfire_dataset
from utils.load_fire_dataset import load_firesense_dataset
from utils.load_fire_dataset import load_firenet_dataset
from utils.load_fire_dataset import load_firenet_test_dataset
from utils.load_fire_dataset import load_fismo_dataset
# utils
from utils.prepare_dataset import save_dataframe
# model
from keras_octconv.octave_resnet import OctaveResNet

# root paths
root_path = os.path.dirname(os.path.abspath(__file__))
datasets_root = os.path.join(root_path, '..', 'datasets')

datasets = {
    'cairfire': {
        'name': 'CairFire',
        'path': os.path.join(datasets_root, 'CairDataset'),
        'dataset': load_cairfire_dataset
    },
    'firesense': {
        'name': 'FireSense',
        'path': os.path.join(datasets_root, 'FireSenseDataset'),
        'dataset': load_firesense_dataset
    },
    'firenet': {
        'name': 'FireNet',
        'path': os.path.join(datasets_root, 'FireNetDataset'),
        'dataset': load_firenet_dataset
    },
    'firenet_test': {
        'name': 'FireNet-Test',
        'path': os.path.join(datasets_root, 'FireNetDataset'),
        'dataset': load_firenet_test_dataset
    },
    'fismo': {
        'name': 'FiSmo',
        'path': os.path.join(datasets_root, 'FiSmo-Images'),
        'dataset': load_fismo_dataset
    },
}

def OctFiResNet(include_top=True,
                   weights=None,
                   input_tensor=None,
                   input_shape=None,
                   pooling=None,
                   classes=10,
                   alpha=0.25,
                   expansion=4,
                   initial_filters=64,
                   initial_strides=False,
                   **kwargs):
    depth_v1 = [4, 2]
    depth_v2 = [4, 4]
    return OctaveResNet(depth_v1,
                        include_top,
                        weights,
                        input_tensor,
                        input_shape,
                        pooling,
                        classes,
                        alpha,
                        expansion,
                        initial_filters,
                        initial_strides,
                        **kwargs)
# end OctFiResNet

# model
def build_model(input_shape, num_classes=2, learning_rate=1e-4):
    #Load the base model
    model = OctFiResNet(input_shape=input_shape, classes=num_classes)

    # Compile the model
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.Nadam(lr=learning_rate),
                  metrics=['accuracy'])

    return model
# end build_model

def summarize_model(save_dir, model_name):
    model = OctFiResNet(classes=2)
    plot_model(model,
        to_file=os.path.join(save_dir, model_name + '_model.png'),
        show_shapes=False)

    # Show a summary of the model.
    model.summary()

    with open(os.path.join(save_dir, model_name + '_summary.txt'), 'w') as f:
        with redirect_stdout(f):
            model.summary()
# end summarize_mode

def prepare_callbacks(save_dir, suffix):
    # Prepare model model saving directory.
    model_name = 'model_%s.h5' % suffix
    history_name = 'history_%s.csv' % suffix

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    filepath = os.path.join(save_dir, model_name)
    historypath = os.path.join(save_dir, history_name)

    # Prepare callbacks for saving and for learning rate adjustment.
    checkpoint = ModelCheckpoint(filepath=filepath,
                                 monitor='val_acc',
                                 verbose=1,
                                 save_best_only=True)

    csv_logger = CSVLogger(filename=historypath,
                            separator=',',
                            append=False)

    return [csv_logger, checkpoint]
# end prepare_callbacks

def train_model(train_dataset, iterations, iter_start=0, test_dataset=None,
                save_dir='models', batch_size=32, epochs=100, input_shape=(96, 96, 3),
                debug=False):
    # dataset var
    train_id = train_dataset
    train_path = datasets[train_id]['path']
    train_load = datasets[train_id]['dataset']
    # img prep
    resize_w = input_shape[0]
    resize_h = input_shape[1]
    # train prep
    num_classes = 2
    iter_ends = iter_start + iterations

    if test_dataset is None:
        test_id = 'self'
        x_train, y_train, x_test, y_test = train_load(train_path,
                                                width_resize=resize_w,
                                                height_resize=resize_h,
                                                debug=debug)
    else:
        test_id = test_dataset
        test_path = datasets[test_id]['path']
        test_load = datasets[test_id]['dataset']

        x_train, y_train = train_load(train_path,
                                    val_split=False,
                                    width_resize=resize_w,
                                    height_resize=resize_h,
                                    debug=debug)

        x_test, y_test = test_load(test_path,
                                    val_split=False,
                                    width_resize=resize_w,
                                    height_resize=resize_h,
                                    debug=debug)
    # end if

    # Normalize data.
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    # summary
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(y_train[y_train==1].shape[0], 'fire')
    print(y_train[y_train==0].shape[0], 'no_fire')

    print('x_test shape:', x_test.shape)
    print(x_test.shape[0], 'test samples')
    print(y_test[y_test==1].shape[0], 'fire')
    print(y_test[y_test==0].shape[0], 'no_fire')

    # Convert class vectors to binary class matrices.
    y_train = utils.to_categorical(y_train, num_classes)
    y_test = utils.to_categorical(y_test, num_classes)

    # iterations history
    datasets_id = '%s_%s' % (train_id, test_id)
    history_file = 'validation_%s.csv' % datasets_id
    if os.path.exists(os.path.join(save_dir, history_file)):
        history = pd.read_csv(history_file)
    else:
        history = pd.DataFrame()
    # first run with same dataset
    for i in range(iter_start, iter_ends):
        idx = '{:03d}'.format(i)
        version_run = '%s_%s' % (datasets_id, idx)
        print('Running', version_run)

        #call the model
        model = build_model(input_shape)
        #set callbacks
        callbacks = prepare_callbacks(save_dir, version_run)

        #training with self dataset
        metrics = model.fit(x_train, y_train,
                            batch_size=batch_size, epochs=epochs,
                            validation_data=(x_test, y_test),
                            callbacks=callbacks,
                            shuffle=True)
        history[idx] = metrics.history['val_acc']

    save_dataframe(history, save_dir, filename=history_file)
# end train_model
