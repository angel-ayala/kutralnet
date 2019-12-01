import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, AveragePooling2D

def firenet_tf(input_shape):
    model = Sequential()

    model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu', input_shape=input_shape))#X.shape[1:]))
    model.add(AveragePooling2D())
    model.add(Dropout(0.5))

    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
    model.add(AveragePooling2D())
    model.add(Dropout(0.5))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(AveragePooling2D())
    model.add(Dropout(0.5))

    model.add(Flatten())

    model.add(Dense(units=256, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(units=128, activation='relu'))

    model.add(Dense(units=2, activation = 'softmax'))

    return model
