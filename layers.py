import tensorflow as tf

def hidden_layer(data, layer):
    hidden = tf.add(tf.matmul(data, layer['weights']), layer['biases'])
    sigmoid_hidden = tf.sigmoid(hidden)

    return sigmoid_hidden
