import os.path
import tempfile
import tensorflow as tf
import keras.backend as K
from tensorflow.python.keras import Model, Sequential
from models.firenet_tf import firenet_tf
# import numpy as np

def get_flops(model):
    run_meta = tf.RunMetadata()
    opts = tf.profiler.ProfileOptionBuilder.float_operation()

    # We use the Keras session graph in the call to the profiler.
    flops = tf.profiler.profile(graph=K.get_session().graph,
                                run_meta=run_meta, cmd='op', options=opts)

    return flops.total_float_ops  # Prints the "flops" of the model.

model = firenet_tf(input_shape=(64, 64, 3))
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# .... Define your model here ....
print(get_flops(model))
