from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range
import tensorflow.contrib.slim as slim

pickle_file = 'notMNIST.pickle'

with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  train_dataset = save['train_dataset']
  train_labels = save['train_labels']
  valid_dataset = save['valid_dataset']
  valid_labels = save['valid_labels']
  test_dataset = save['test_dataset']
  test_labels = save['test_labels']
  del save  # hint to help gc free up memory
  print('Training set', train_dataset.shape, train_labels.shape)
  print('Validation set', valid_dataset.shape, valid_labels.shape)
  print('Test set', test_dataset.shape, test_labels.shape)

MEAN = np.mean(train_dataset)
STD = np.std(train_dataset)

# Convenience method for reshaping images. The included MNIST dataset stores images# Conve 
# as Nx784 row vectors. This method reshapes the inputs into Nx28x28x1 images that are
# better suited for convolution operations and rescales the inputs so they have a
# mean of 0 and unit variance.
def resize_images(images):
    reshaped = (images - MEAN)/STD
    reshaped = np.reshape(reshaped, [-1, 28, 28, 1])
    
    assert(reshaped.shape[1] == 28)
    assert(reshaped.shape[2] == 28)
    assert(reshaped.shape[3] == 1)
    
    return reshaped

def nielsen_net(inputs, is_training, scope='NielsenNet'):
    with tf.variable_scope(scope, 'NielsenNet'):
        # First Group: Convolution + Pooling 28x28x1 => 28x28x20 => 14x14x20
        net = slim.conv2d(inputs, 20, [5, 5], padding='SAME', scope='layer1-conv')
        net = slim.max_pool2d(net, 2, stride=2, scope='layer2-max-pool')

        # Second Group: Convolution + Pooling 14x14x20 => 10x10x40 => 5x5x40
        net = slim.conv2d(net, 40, [5, 5], padding='VALID', scope='layer3-conv')
        net = slim.max_pool2d(net, 2, stride=2, scope='layer4-max-pool')

        # Reshape: 5x5x40 => 1000x1
        net = tf.reshape(net, [-1, 5*5*40])

        # Fully Connected Layer: 1000x1 => 1000x1
        net = slim.fully_connected(net, 1000, scope='layer5')
        net = slim.dropout(net, is_training=is_training, scope='layer5-dropout')

        # Second Fully Connected: 1000x1 => 1000x1
        net = slim.fully_connected(net, 1000, scope='layer6')
        net = slim.dropout(net, is_training=is_training, scope='layer6-dropout')

        # Output Layer: 1000x1 => 10x1
        net = slim.fully_connected(net, 10, scope='output')
        net = slim.dropout(net, is_training=is_training, scope='output-dropout')

        return net

sess = tf.InteractiveSession()

# Create the placeholder tensors for the input images (x), the training labels (y_actual)
# and whether or not dropout is active (is_training)
x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name='Inputs')
y_actual = tf.placeholder(tf.float32, shape=[None, 10], name='Labels')
is_training = tf.placeholder(tf.bool, name='IsTraining')

# Pass the inputs into nielsen_net, outputting the logits
logits = nielsen_net(x, is_training, scope='NielsenNetTrain')

# Use the logits to create four additional operations:
#
# 1: The cross entropy of the predictions vs. the actual labels
# 2: The number of correct predictions
# 3: The accuracy given the number of correct predictions
# 4: The update step, using the MomentumOptimizer
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, y_actual))
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_actual, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
train_step = tf.train.MomentumOptimizer(0.01, 0.5).minimize(cross_entropy)


# To monitor our progress using tensorboard, create two summary operations# To mo 
# to track the loss and the accuracy
loss_summary = tf.summary.scalar('loss', cross_entropy)
accuracy_summary = tf.summary.scalar('accuracy', accuracy)

sess.run(tf.global_variables_initializer())
train_writer = tf.summary.FileWriter('/tmp/nielsen-net', sess.graph)


eval_data  = {
    x: resize_images(valid_dataset),
    y_actual: valid_labels,
    is_training: False
}

for i in xrange(100000):
    images, labels = mnist.train.next_batch(100)
    summary, _ = sess.run([loss_summary, train_step], feed_dict={x: resize_images(images), y_actual: labels, is_training: True})
    train_writer.add_summary(summary, i)
    
    if i % 1000 == 0:
        summary, acc = sess.run([accuracy_summary, accuracy], feed_dict=eval_data)
        train_writer.add_summary(summary, i)
        print("Step: %5d, Validation Accuracy = %5.2f%%" % (i, acc * 100))

test_data = {
    x: resize_images(test_dataset),
    y_actual: test_labels,
    is_training: False
}

acc = sess.run(accuracy, feed_dict=test_data)

print("Test Accuracy = %5.2f%%" % (100 * acc))