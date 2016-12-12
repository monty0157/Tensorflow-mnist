import tensorflow as tf

def manual_value(value, size_layer_j, size_layer_s):
    parameter_value = tf.convert_to_tensor([value]*(size_layer_j * size_layer_s), dtype=tf.float32)
    parameter_variable = tf.Variable(tf.reshape(parameter_value, [size_layer_j, size_layer_s]))

    return parameter_variable

def value(size_layer_j, size_layer_s):
    parameter_value = tf.Variable(tf.random_normal([size_layer_j, size_layer_s]))

    return parameter_value
