"""
Training complete in _m _s
Best val Acc: 0.93553
Test Acc: 0.91389
"""
import os
import numpy as np
# from keras import utils
from matplotlib import pyplot as plt
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.models import load_model

from datasets import FireNetDataset
from models.firenet_tf import firenet_tf
from sklearn.metrics import classification_report

must_train = False
must_test = True

### Training
if must_train:
    dt = FireNetDataset(size=(64,64))
    x_train, y_train, x_val, y_val = dt.load_train_val()

    # Normalize data.
    x_train = dt.preprocess(x_train)
    x_val = dt.preprocess(x_val)

    # summary
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(y_train[y_train==1].shape[0], 'fire')
    print(y_train[y_train==0].shape[0], 'no_fire')

    print('x_val shape:', x_val.shape)
    print(x_val.shape[0], 'test samples')
    print(y_val[y_val==1].shape[0], 'fire')
    print(y_val[y_val==0].shape[0], 'no_fire')

    num_classes = len(dt.classes)
    input_shape = x_train.shape[1:]
    print('num_classes', num_classes, 'input_shape', input_shape)

    # Convert class vectors to binary class matrices.
    # y_train = utils.to_categorical(y_train, num_classes)
    # y_val = utils.to_categorical(y_val, num_classes)

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

    model = firenet_tf(input_shape=input_shape)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.summary()
    # exit()
    #
    history = model.fit(x_train, y_train, batch_size=32, epochs=100,
                        validation_data=(x_val, y_val), callbacks=prepare_callbacks('models/saved', 'firenet_tf'))

    # model.save('models/saved/firenet_tf.h5')


    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

### Test
if must_test:
    dt = FireNetDataset(size=(64,64))
    x_test, y_test = dt.load_test()

    # Normalize data.
    x_test = dt.preprocess(x_test)

    # summary
    print('x_test shape:', x_test.shape)
    print(x_test.shape[0], 'test samples')
    print(y_test[y_test==1].shape[0], 'fire')
    print(y_test[y_test==0].shape[0], 'no_fire')

    num_classes = len(dt.classes)
    input_shape = x_test.shape[1:]
    print('num_classes', num_classes, 'input_shape', input_shape)

    model = firenet_tf(input_shape=input_shape)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.load_weights('models/saved/model_firenet_tf.h5')

    score = model.evaluate(x_test, y_test, verbose=2)
    print(score)

    #Confusion Matrix and Classification Report
    y_pred = model.predict(x_test, verbose=0)
    y_pred = np.argmax(y_pred, axis=1)
    # Y_test = np.argmax(y_test, axis=1)

    target_names = ['No Fire', 'Fire']
    class_report = classification_report(y_test, y_pred,
                            target_names=target_names)#, output_dict=True)

    print('Classification Report')
    print(class_report)
