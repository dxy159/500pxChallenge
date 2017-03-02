##########################################################
####### CODE FROM MNIST TUTORIAL 
##########################################################

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

import tensorflow as tf
sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

sess.run(tf.initialize_all_variables())

y = tf.matmul(x,W) + b

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

for i in range(1000):
  batch = mnist.train.next_batch(100)
  train_step.run(feed_dict={x: batch[0], y_: batch[1]})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

###############################################################
######### NOW WE CREATE ADVERSARIAL IMAGES
###############################################################

# Import the necessary modules
from sklearn.preprocessing import LabelBinarizer
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mplimg

# We need to transform onehot encoding back to multiclass labels

lb = LabelBinarizer()
lb.fit([0,1,2,3,4,5,6,7,8,9])
labels_reg = lb.inverse_transform(mnist.test.labels)

# Find all the images and make label for 2
images_2 = mnist.test.images[labels_reg == 2]
label_2 = np.array([[0,0,1,0,0,0,0,0,0,0]])
lb.inverse_transform(label_2)

# Ravel the label
label_2 = label_2.ravel()

# Get the labels into an array and stack them
a = [label_2 for x in range(images_2.shape[0])]
labels_stack_2 = np.stack(a, axis=0)

# We need the weights and biases in the form of an array

W_arr = W.eval()
b_arr = b.eval()

# Get the weight properties for 6
W_6 = W_arr[:, 6]

# Get the transformed matrix by multiplying with the 6 weight vector
transformed = images_2 + 1.5 * W_6

# Make the stacked labels for 6
label_6 = np.array([[0,0,0,0,0,0,1,0,0,0]])
label_6 = label_6.ravel()
b = [label_6 for x in range(images_2.shape[0])]
labels_stack_6 = np.stack(b, axis=0)

def plot(array=None):
  img_plot = plt.imshow(array)

# Create original image
img = images_2[0, :].reshape((28,28))
plot(array=img)

# Plot transformed
img_transformed = transformed[0, :].reshape((28,28))
plot(array=img_transformed)

# Plot delta
W_delta = 1.5 * W_6
img_delta = W_delta.reshape((28,28))
plot(array=img_delta)

# Function that plots image given the matrix of image, type of the matrix to be either original
# or transformed, and the order of the images in the matrix
def plot_im(array=None, ind=0):
    img_reshaped = array[ind, :].reshape((28, 28))
    imgplot = plt.imshow(img_reshaped)

plot_im(array=images_2, ind=1)
plot_im(array=transformed, ind=1)



# Output as a grid of 10 rows and 3 cols with first column being original, second being
# delta and third column being the transformed image
nrow = 10
ncol = 3
n = 0

from matplotlib import gridspec
fig = plt.figure(figsize=(8, 20)) 
gs = gridspec.GridSpec(nrow, ncol, width_ratios=[1, 1, 1], 
                       wspace=0.0, hspace=0.0, 
                       top=0.95, bottom=0.05, left=0.17, right=0.845) 

for row in range(nrow):
    for col in range(ncol):
        ax = plt.subplot(gs[row, col])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        if col == 0:
            plot_im(array=images_2, ind=row)
        elif col == 1:
            plt.imshow(img_delta)
        else:
            plot_im(array=transformed, ind=row)
        n += 1

plt.savefig('result.jpg')











