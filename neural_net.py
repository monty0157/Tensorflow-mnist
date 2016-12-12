import tensorflow as tf
import parameters

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

batch_size = 100
learning_rate = 10
training_epochs = 10
n_nodes_layer_input = 784
n_nodes_layer2 = 100
n_nodes_layer3 = 100
n_nodes_layer4 = 100
n_nodes_layer_output = 10
parameter_value_manual = 2


with tf.name_scope('input'):
    x_input = tf.placeholder(tf.float32, shape=(None,784), name="x-input")
    y_output = tf.placeholder(tf.float32, shape=(None, 10), name="y-output")

with tf.name_scope('weights'):
    w_1 = parameters.value(n_nodes_layer_input, n_nodes_layer2)
    #w_1 = parameters.manual_value(parameter_value_manual, n_nodes_layer_input, n_nodes_layer2)
    w_2 = parameters.value(n_nodes_layer2, n_nodes_layer3)
    #w_2 = parameters.manual_value(parameter_value_manual, n_nodes_layer2, n_nodes_layer3)
    w_3 = parameters.value(n_nodes_layer3, n_nodes_layer4)
    #w_3 = parameters.manual_value(parameter_value_manual, n_nodes_layer3, n_nodes_layer_output)
    w_4 = parameters.value(n_nodes_layer4, n_nodes_layer_output)

    w_no_hl = parameters.value(n_nodes_layer_input, n_nodes_layer_output)

with tf.name_scope('biases'):
    b_1 = tf.Variable(tf.random_normal([1, n_nodes_layer2]))
    #b_1 = parameters.manual_value(parameter_value_manual, 1, n_nodes_layer2)
    b_2 = tf.Variable(tf.random_normal([1, n_nodes_layer3]))
    #b_2 = parameters.manual_value(parameter_value_manual, 1, n_nodes_layer3)
    b_3 = tf.Variable(tf.random_normal([1, n_nodes_layer4]))
    #b_3 = parameters.manual_value(parameter_value_manual, 1, n_nodes_layer_output)
    b_4 = parameters.value(1,n_nodes_layer_output)


def neural_network(data):
    layer_2 = {'weights': w_1, 'biases': b_1}

    layer_3 = {'weights': w_2, 'biases': b_2}

    layer_4 = {'weights': w_3, 'biases': b_3}

    output_layer = {'weights': w_4, 'biases': b_4}

    l2 = tf.add(tf.matmul(data, layer_2['weights']), layer_2['biases'])
    z_l2 = tf.sigmoid(l2)

    l3 = tf.add(tf.matmul(z_l2, layer_3['weights']), layer_3['biases'])
    z_l3 = tf.sigmoid(l3)

    l4 = tf.add(tf.matmul(z_l3, layer_4['weights']), layer_4['biases'])
    z_l4 = tf.sigmoid(l4)

    output = tf.add(tf.matmul(z_l4, output_layer['weights']), output_layer['biases'])
    output = tf.sigmoid(output)

    return output

def train_neural_network(x):
    prediction = neural_network(x)
    with tf.name_scope("cost"):
        #cost = tf.contrib.losses.mean_squared_error(prediction, y_output)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, y_output))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate, name='GradientDescent').minimize(cost)
        tf.summary.scalar("cost", cost)

    correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y_output, 1))


    with tf.name_scope('accuracy'):
       accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
       tf.summary.scalar("accuracy", accuracy)


    with tf.Session() as sess:
        summary_op = tf.merge_all_summaries()
        writer = tf.train.SummaryWriter("./logs/3hidden_layers_300n", graph=tf.get_default_graph())

        sess.run(tf.global_variables_initializer())

        for epoch in range(training_epochs):
            epoch_loss = 0
            for j in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                j, c, summary = sess.run([optimizer, cost, summary_op], feed_dict = {x: epoch_x, y_output: epoch_y })
                epoch_loss += c

                writer.add_summary(summary, j)

            print('Epoch', epoch, 'completed out of', training_epochs, 'loss:', epoch_loss,)


        print('Accuracy:', accuracy.eval({x:mnist.test.images, y_output:mnist.test.labels}))

train_neural_network(x_input)
