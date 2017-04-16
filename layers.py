import tensorflow as tf

def hidden_layer(data, layer):
    hidden = tf.matmul(data, layer['weights'])
    sigmoid_hidden = tf.sigmoid(hidden)

    return sigmoid_hidden
