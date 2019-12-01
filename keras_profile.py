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

# input = np.random(1, 3, 64, 64)

def count_flops(model):
    """ Count flops of a keras model
    # Args.
        model: Model,
    # Returns
        int, FLOPs of a model
    # Raises
        TypeError, if a model is not an instance of Sequence or Model
    """

    if not isinstance(model, (Sequential, Model)):
        raise TypeError(
            'Model is expected to be an instance of Sequential or Model, '
            'but got %s' % type(model))

    output_op_names = [_out_tensor.op.name for _out_tensor in model.outputs]
    sess = tf.keras.backend.get_session()
    frozen_graph_def = tf.graph_util.convert_variables_to_constants(
        sess, sess.graph.as_graph_def(), output_op_names)

    with tempfile.TemporaryDirectory() as tmpdir:
        graph_file = os.path.join(os.path.join(tmpdir, 'graph.pb'))
        with tf.gfile.GFile(graph_file, "wb") as f:
            f.write(frozen_graph_def.SerializeToString())

        with tf.gfile.GFile(graph_file, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        with tf.Graph().as_default() as new_graph:
            tf.import_graph_def(graph_def, name='')
            tfprof_opts = tf.profiler.ProfileOptionBuilder.float_operation()
            flops = tf.profiler.profile(new_graph, options=tfprof_opts)
            writer = tf.summary.FileWriter('gg', graph=new_graph)
            writer.flush()

    return flops

model = firenet_tf(input_shape=(64, 64, 3))
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# .... Define your model here ....
print(count_flops(model))
