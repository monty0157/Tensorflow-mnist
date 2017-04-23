import neural_net
import tensorflow as tf

with tf.name_scope('input'):
    x_input = tf.placeholder('float')

n_nodes_layer_input = 784
n_nodes_layer2 = 75
n_nodes_layer3 = 75
n_nodes_layer4 = 25
n_nodes_layer_output = 10

neural_net.train_neural_network(x_input, [n_nodes_layer_input, n_nodes_layer2, n_nodes_layer3,
                                n_nodes_layer_output], 2)
