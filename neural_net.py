import tensorflow as tf
import scipy.io as sio
import layers
import parameters
import numpy as np
import image_import
from matplotlib import pyplot as plt

#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

#Loading SVHN dataset
SVHN = sio.loadmat('./train_32x32.mat')

#SVHN is a dict with 2 variables:
#X which is a 4-D matrix containing the images,
#and y which is a vector of class labels. To access the images,
#X(:,:,:,i) gives the i-th 32-by-32 RGB image, with class label y(i).

SVHN_images = SVHN['X']
n_SVHN_images = SVHN['X'].shape[3]
SVHN_labels = SVHN['y']

#Array to store labels converted to one-hot vectors
SVHN_labels_vector = []

n_nodes_layer_input = 784
n_nodes_layer_output = 10
batch_size = 100
learning_rate = 10
training_epochs = 10
parameter_value_manual = 6
parameter_list = []
activated_units = []


with tf.name_scope('output'):
    y_output = tf.placeholder('float')


def neural_network(data, n_nodes_layers = None, n_hidden_layers = None):

    if n_nodes_layers is None:
        n_nodes_layers = [n_nodes_layer_input, n_nodes_layer_output]

    parameters_output = parameters.layer(1024, 10)

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

        placeholder = tf.Variable(tf.truncated_normal([1,1024]), name="weight")
        output = layers.hidden_layer(placeholder, parameters_output)

    return output

def train_neural_network(x, *args):
    prediction = neural_network(x, *args)
    with tf.name_scope("cost"):
        #cost = tf.contrib.losses.mean_squared_error(prediction, y_output)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y_output))
        #cost = tf.reduce_mean(-y_output*tf.log(prediction)-(1-y_output)*tf.log(1-prediction))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate, name='GradientDescent').minimize(cost)
        tf.summary.scalar("cost", cost)

    correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y_output, 0))


    with tf.name_scope('accuracy'):
       accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
       tf.summary.scalar("accuracy", accuracy)



    saver = tf.train.Saver()

    with tf.Session() as sess:
        summary_op = tf.summary.merge_all()
        writer = tf.summary.FileWriter("./logs/unge_forskere_run2", graph=tf.get_default_graph())

        sess.run(tf.global_variables_initializer())



        #FOR TRAINING THE NETWORK
        for epoch in range(training_epochs):
            epoch_loss = 0
            start = 0
            end = int(batch_size)

            #labels iterator
            k = 0
            for j in range(int(3)):
                for i in range (end-start):
                    SVHN_labels_vector.append(tf.one_hot(SVHN_labels[k]-1, depth=10).eval())
                    k += 1
                print(SVHN_images[:,:,:,start:end].shape)
                epoch_x, epoch_y = SVHN_images[start:end,start:end,start:end,start:end], SVHN_labels_vector[start:end]
                j, c, summary = sess.run([optimizer, cost, summary_op], feed_dict = {x: epoch_x, y_output: epoch_y })
                epoch_loss += c
                print(epoch_loss)

                writer.add_summary(summary, j)

            print('Epoch', epoch, 'completed out of', training_epochs, 'loss:', epoch_loss)

            save_path = saver.save(sess, "my-model")

        print('Accuracy:', accuracy.eval({x:mnist.test.images, y_output:mnist.test.labels}))


        '''#FOR TESTING THE NETWORK
        model_import = tf.train.import_meta_graph('my-model_unge_forskere.meta')
        model_import.restore(sess, tf.train.latest_checkpoint('./'))

        shape = mnist.train.images[39775].reshape(28,28)

        image = image_import.downsample('./digit_test_2.png')
        image = image[1:-1, 1:-1].reshape(1,784)
#        print(image.shape)

#        print(image_import.downsample('./digit_test_2.png').shape)
        result = sess.run(tf.argmax(prediction,1), feed_dict={x: image})

        print (' '.join(map(str, result)))

        #print(mnist.train.labels[39775])
        #plt.imshow(255.0-shape)
        plt.imshow(image_import.downsample('./digit_test_2.png'))
        plt.show()'''
