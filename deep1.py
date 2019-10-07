import numpy as np
from PIL import Image
import tensorflow as tf


"import the MNIST dataset and store the image data in the variable mnist"
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)  # y labels are hot-encoded

"dataset has been split into 55,000 images for training, 5000 for validation, and 10,000 for testing"
n_train = mnist.train.num_examples  # 55,000
n_validation = mnist.validation.num_examples  # 5000
n_test = mnist.test.num_examples  # 10,000

"Neural Network Input Units per Layer"
n_input = 784  # input layer (28x28 pixels)
n_hidden1 = 512  # 1st hidden layer
n_hidden2 = 256  # 2nd hidden layer
n_hidden3 = 128  # 3rd hidden layer
n_hidden4 = 64  # 4rd hidden layer
n_output = 10  # output layer (0-9 digits)

"Hyperparameters"
learning_rate = 1e-3
n_iterations = 2000
batch_size = 256
dropout = 1

"Tensor declaration as placeholders"
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_output])
keep_prob = tf.placeholder(tf.float32)

"Weights and biases for neural network prediction, ease of access when set to dictionary data type"
weights = {
    'w1': tf.Variable(tf.truncated_normal([n_input, n_hidden1], stddev=0.1)),
    'w2': tf.Variable(tf.truncated_normal([n_hidden1, n_hidden2], stddev=0.1)),
    'w3': tf.Variable(tf.truncated_normal([n_hidden2, n_hidden3], stddev=0.1)),
    'w4': tf.Variable(tf.truncated_normal([n_hidden3, n_hidden4], stddev=0.1)),
    'out': tf.Variable(tf.truncated_normal([n_hidden4, n_output], stddev=0.1)),
}

biases = {
    'b1': tf.Variable(tf.constant(0.1, shape=[n_hidden1])),
    'b2': tf.Variable(tf.constant(0.1, shape=[n_hidden2])),
    'b3': tf.Variable(tf.constant(0.1, shape=[n_hidden3])),
    'b4': tf.Variable(tf.constant(0.1, shape=[n_hidden4])),
    'out': tf.Variable(tf.constant(0.1, shape=[n_output]))
}

"Set up layers which handle the Tensors"
layer_1 = tf.add(tf.matmul(X, weights['w1']), biases['b1'])
layer_2 = tf.add(tf.matmul(layer_1, weights['w2']), biases['b2'])
layer_3 = tf.add(tf.matmul(layer_2, weights['w3']), biases['b3'])
layer_4 = tf.add(tf.matmul(layer_3, weights['w4']), biases['b4'])
layer_drop = tf.nn.dropout(layer_4, keep_prob)
output_layer = tf.matmul(layer_4, weights['out']) + biases['out']


"cross_entropy(log-loss) error reduction function, using gradient descent optimization to converge on minimal loss"
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(
        labels=Y, logits=output_layer
        ))
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

"Evaluate if prediction is correct with boolean statement and create accuracy %"
correct_pred = tf.equal(tf.argmax(output_layer, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

"Training initialization"
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
"Training per batch for each iteration"
for i in range(n_iterations):
    batch_x, batch_y = mnist.train.next_batch(batch_size)
    sess.run(train_step, feed_dict={
        X: batch_x, Y: batch_y, keep_prob: dropout
        })

    # print loss and accuracy (per minibatch)
    if i % 100 == 0:
        minibatch_loss, minibatch_accuracy = sess.run(
            [cross_entropy, accuracy],
            feed_dict={X: batch_x, Y: batch_y, keep_prob: 1.0}
            )
        print("Training"
            "Iteration",
           str(i),
           "\t| Loss =",
            str(minibatch_loss),
            "\t| Accuracy =",
            str(minibatch_accuracy)
            )

test_accuracy = sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1.0})
print("\nAccuracy on test set:", test_accuracy)
"Test images against the database to verify learning"
#Using UI1 which takes active input from user
def prediction(img):
    return sess.run(tf.argmax(output_layer, 1), feed_dict={X: [img]})


"""Console input code from image created 28x28 through Paint
img = np.invert(Image.open(r"C:\\Users\\erick\\Desktop\\img_test.png").convert('L')).ravel()
print ("Prediction for test image:", np.squeeze(prediction))"""
