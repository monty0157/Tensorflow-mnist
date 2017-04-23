import tensorflow as tf
import layers
import parameters
import numpy as np
import image_import
from PIL import Image
from matplotlib import pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

n_nodes_layer_input = 784
n_nodes_layer_output = 10
batch_size = 100
learning_rate = 10
training_epochs = 12
parameter_value_manual = 6
parameter_list = []
activated_units = []

with tf.name_scope('output'):
    y_output = tf.placeholder(tf.float32, shape=(None, 10), name="y-output")


def neural_network(data, n_nodes_layers = None, n_hidden_layers = None):

    if n_nodes_layers is None:
        n_nodes_layers = [n_nodes_layer_input, n_nodes_layer_output]

    parameters_output = parameters.layer(n_nodes_layers[-2], n_nodes_layers[-1])

    if n_hidden_layers is None:
        output = layers.hidden_layer(data, parameters_output)
    else:

        if len(n_nodes_layers) != (n_hidden_layers + 2):
            raise ValueError('Number of hidden layers does not match layers in the list n_nodes_layers')


        for n in range(n_hidden_layers+1):
            parameters_n = parameters.layer(n_nodes_layers[n], n_nodes_layers[n+1])
            parameter_list.append(parameters_n)


        activated_units.append(data)
        for i in range(len(parameter_list)):
            z_hl = layers.hidden_layer(activated_units[i], parameter_list[i])
            activated_units.append(z_hl)

        output = layers.hidden_layer(activated_units[len(activated_units)-2], parameters_output)

    return output

def train_neural_network(x, *args):
    prediction = neural_network(x, *args)
    with tf.name_scope("cost"):
        #cost = tf.contrib.losses.mean_squared_error(prediction, y_output)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y_output))
        #cost = tf.reduce_mean(-y_output*tf.log(prediction)-(1-y_output)*tf.log(1-prediction))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate, name='GradientDescent').minimize(cost)
        tf.summary.scalar("cost", cost)

    correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y_output, 1))


    with tf.name_scope('accuracy'):
       accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
       tf.summary.scalar("accuracy", accuracy)

    '''tensorboard_image = mnist.train.images[1].reshape(28,28)
    plt.imsave('./test.jpeg', arr=tensorboard_image, format="jpeg")
    tensorboard_image = tf.image.encode_jpeg(mnist.train.images[1].reshape(28,28,1), format='grayscale')
    print(tensorboard_image)
    tensorboard_image = tf.image.decode_jpeg(tensorboard_image)
    print(tensorboard_image)
    with tf.name_scope('images'):
        tf.summary.image("images",tensorboard_image)'''


    saver = tf.train.Saver()

    with tf.Session() as sess:
        summary_op = tf.summary.merge_all()
        #writer = tf.summary.FileWriter("./logs/test", graph=tf.get_default_graph())

        sess.run(tf.global_variables_initializer())

        '''#FOR TRAINING THE NETWORK
        #for img in range (len(mnist.train.images)):
        #    mnist.train.images[img] -= np.mean(mnist.train.images, axis=0)
        #    print(img)

        for epoch in range(training_epochs):
            epoch_loss = 0
            start = 0
            end = int(batch_size)
            for j in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                j, c, summary = sess.run([optimizer, cost, summary_op], feed_dict = {x: epoch_x, y_output: epoch_y })
                epoch_loss += c
                #writer.add_summary(summary, j)
                start += int(batch_size)
                end += int(batch_size)
            print('Epoch', epoch, 'completed out of', training_epochs, 'loss:', epoch_loss)
            #save_path = saver.save(sess, "my-model")
        print('Accuracy:', accuracy.eval({x:mnist.test.images, y_output:mnist.test.labels}))

        print("TEST_NORMAL")
        total_accuracy = 0
        start = 0
        end = int(batch_size)
        for j in range(int(10000/batch_size)):
            acc_x, acc_y = mnist.test.images[start:end], mnist.test.labels[start:end]
            total_accuracy += accuracy.eval(feed_dict={x:acc_x, y_output:acc_y})
            start += int(batch_size)
            end += int(batch_size)

        mean_accuracy = total_accuracy/int(10000/batch_size)
        print(mean_accuracy)

        print("TEST_MANIPULATED_DATA")
        total_accuracy = 0
        start = 0
        end = int(batch_size)
        for img in range (len(mnist.test.images)):
            image_import.image_round(mnist.test.images[img])
        for j in range(int(10000/batch_size)):
            acc_x, acc_y = mnist.test.images[start:end], mnist.test.labels[start:end]
            total_accuracy += accuracy.eval(feed_dict={x:acc_x, y_output:acc_y})
            start += int(batch_size)
            end += int(batch_size)

        mean_accuracy = total_accuracy/int(10000/batch_size)
        print(mean_accuracy)'''


        #FOR TESTING THE NETWORK
        model_import = tf.train.import_meta_graph('my-model.meta')
        model_import.restore(sess, tf.train.latest_checkpoint('./'))


        image = image_import.image_processed('./live_digit_test_21.png', 28)
        print(image.shape)
        image = image.reshape((784))
        #image -= np.mean(mnist.train.images, axis=0)

        #mnist.train.images[10004] = image_import.image_round(mnist.train.images[10004])
        #mnist.train.images[10004] -= np.mean(mnist.train.images, axis=0)

        result_index = sess.run(tf.argmax(prediction,1), feed_dict={x: [image]})
        result_vector = sess.run(prediction, feed_dict={x: [image]})

        #Reshape for easy data manipulation
        result_vector = result_vector.reshape(10)

        #Convert scientific notation to floating point value
        np.set_printoptions(precision=10, suppress=True)
        print(result_vector)
        print (' '.join(map(str, result_index)))
        #print(mnist.train.images[10004])
        #print(image)

        plt.imshow(image.reshape(28,28))
        #plt.imshow(mnist.train.images[10004].reshape(28,28))
        plt.show()
