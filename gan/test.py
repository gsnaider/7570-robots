from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from math import *

CHECKPOINT_DIR = "checkpoints-test"

LATENT_SPACE_SHAPE = 100

GEN_VARIABLE_SCOPE = "generator"


def display_data(images):
    example_width = int(round(sqrt(images.shape[1])))

    # Compute rows, cols
    print(images.shape)
    (m, n) = images.shape
    example_height = int(n / example_width)

    # Compute number of items to display
    display_rows = 10
    display_cols = 4

    # Between images padding
    pad = 1

    # Setup blank display
    display_array = - np.ones((pad + display_rows * (example_height + pad),
                               pad + display_cols * (example_width + pad)))
    # Copy each example into a patch on the display array
    curr_ex = 0
    for j in range(display_rows):
        for i in range(display_cols):
            if curr_ex >= m:
                break
            # Copy the patch

            # Get the max value of the pach
            max_val = np.max(np.abs(images[curr_ex, :]))

            height_idx = pad + j * (example_height + pad)
            width_idx = pad + i * (example_width + pad)
            display_array[height_idx: height_idx + example_height, width_idx: width_idx + example_width] = np.reshape(
                images[curr_ex, :],
                (example_height,
                 example_width)) / max_val
            curr_ex = curr_ex + 1
        if curr_ex >= m:
            break
    plt.imshow(display_array.T, cmap='gray')
    plt.show()

def generator(latent_space, label, training=False):
    """
    Defines the generator network using the latent_space as input.
    Args:
        latent_space: input for the generator network
        label: 10 dimensioanl one-hot tensor
    Returns:
        Generated images
    """
    with tf.variable_scope(GEN_VARIABLE_SCOPE):
        net = tf.concat([latent_space, label], axis=1)
        net = tf.layers.dense(net, 7 * 7 * 64, activation=None, use_bias=False)
        net = tf.layers.batch_normalization(net, training=training)

        # 7 x 7
        net = tf.reshape(net, [-1, 7, 7, 64])

        # 14 x 14
        net = tf.layers.conv2d_transpose(net, 64, kernel_size=(5, 5),
                                         strides=(1, 1),
                                         activation=None,
                                         padding='same',
                                         use_bias=False)
        net = tf.layers.batch_normalization(net, training=training)
        net = tf.nn.leaky_relu(net)

        net = tf.layers.conv2d_transpose(net, 32, kernel_size=(5, 5),
                                         strides=(2, 2),
                                         activation=None,
                                         padding='same',
                                         use_bias=False)
        net = tf.layers.batch_normalization(net, training=training)
        net = tf.nn.leaky_relu(net)

        # 28 x 28
        images = tf.layers.conv2d_transpose(net, 1, kernel_size=(5, 5),
                                            strides=(2, 2),
                                            activation=tf.nn.sigmoid,
                                            padding='same',
                                            use_bias=False)
        return images


step = tf.train.get_or_create_global_step()
latent_space = tf.placeholder(tf.float32, shape=[None, LATENT_SPACE_SHAPE])
G_label = tf.placeholder(tf.int32, shape=[None])
G_label_one_hot = tf.one_hot(G_label, 10)
G_images = generator(latent_space, G_label_one_hot)

with tf.train.MonitoredTrainingSession(checkpoint_dir=CHECKPOINT_DIR) as sess:
    # Test
    images = []
    for label in range(10):
        for _ in range(4):
            latent_space_np = np.random.randn(1, LATENT_SPACE_SHAPE)
            image = sess.run([G_images],
                             feed_dict={latent_space: latent_space_np,
                                        G_label: np.array([label])})
            image = np.squeeze(image).flatten()
            images.append(image)
    display_data(np.array(images))