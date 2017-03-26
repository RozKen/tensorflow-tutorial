'''
deep_mnist.py
@brief Deep MNIST for Experts (TensorFlow Tutorial)
@description Learning Tutorial on TensorFlow "Deep MNIST for Experts"
        https://www.tensorflow.org/get_started/mnist/pros
@author Kenichi Yorozu (original mnist_softmax.py is a work of TensorFlow Authors)
@email rozken@gmail.com
@notice This software includes the work that is distributed in the Apache Lincense 2.0
'''
import argparse
import sys
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

FLAGS = None

'''
@fn weight_variable
@brief initialize weight variables with truncated normal distribution(stddev = 0.1)
@param shape : shape of the Weight matrix
@return Initialized Weight Matrix
'''
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
'''
@fn bias_variable
@brief initialize bias variables with constant(0.1)
@param shape : shape of the bias vector
@return Initialized Bias Vector
'''
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
'''
@fn conv2d
@brief Convolute 2D image (4D tensor)
@description strides = [1, 1, 1, 1], padding: Zero-padding('SAME')
@param x : x[batch, in_height, in_width, in_channels]
@param W : Weights[filter_height, filter_width, in_channels, channel_multiplier]
@return Convoluted 2D image(4D tensor)
'''
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
'''
@fn max_pool_2x2
@brief Pooling 2x2 image (4D tensor) with Max Pooling Method 
@description strides = [1, 2, 2, 1], padding: Zero-padding('SAME')
@param x : x[batch, in_height, in_width, in_channels]
@return Pooled 2D image (4D tensor)
'''
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def main(_):
    # Import data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)   #FLAGS.data_dir
    
    '''
        Create the model
    '''
    
    #placeholders
    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])
    keep_prob = tf.placeholder(tf.float32)  # dropout to reduce overfitting set for train: 0.5, for test: 1.0
    
    #reshape x to a 4d tensor[-1, in_height, in_width, in_channels]
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    
    # 1st Convolutional Layer
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)     #image size: 28 x 28 -> 14 x 14
    
    #2nd Convolutional Layer
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)     #image size: 14 x 14 -> 7 x 7
    
    #1st Fully-connected Layer
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    
    #2nd Fully-connected Layer
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    
    # Define loss and optimizer
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    '''
        Train and Test
    '''
    
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    
    for i in range(1001):
        batch = mnist.train.next_batch(50)
        if i % 100 == 0:
            #Log Test Results
            train_accuracy = accuracy.eval(feed_dict={
                x:batch[0], y_:batch[1], keep_prob: 1.0})
            print("step %d, training accuracy %g"%(i, train_accuracy))
        train_step.run(feed_dict = {x: batch[0], y_:batch[1], keep_prob: 0.5})
    
    #Test Trained Model
    test_accuracy = accuracy.eval(feed_dict = {
        x: mnist.test.images[:5500], y_: mnist.test.labels[:5500], keep_prob: 1.0})
    #Program Closes over 6,000 tests Automatically with no Error in my machine (Memory Error?)
    print ("test accuracy %g"%test_accuracy)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)