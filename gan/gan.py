from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import cv2

TRAINING_DATA_DIR = os.path.join("data", "mnist_png", "training")

# Number of inputs counting both mnist data and generated data for the discriminator, and number of random inputs for
# the generator
BATCH_SIZE = 60

GEN_LEARNING_RATE = 0.1
DISC_LEARNING_RATE = 0.1

GEN_HIDDEN_LAYERS = [20, 20, 20, 20, 20]
DISC_HIDDEN_LAYERS = [20, 20, 20, 20, 20]

LATENT_SPACE_SHAPE = 100

GEN_VARIABLE_SCOPE = "generator"
DISC_VARIABLE_SCOPE = "discriminator"


def generator(latent_space):
    """
    Defines the generator network using the latent_space as input.
    Args:
        latent_space: input for the generator network
    Returns:
        Generated images
    """
    with tf.variable_scope(GEN_VARIABLE_SCOPE):
        net = latent_space
        for layer in GEN_HIDDEN_LAYERS:
            net = tf.layers.dense(net, layer, activation=tf.nn.relu)

        output = tf.layers.dense(net, 28 * 28, activation=tf.nn.sigmoid)
        images = tf.reshape(output, [-1, 28, 28])
        return images


def discriminator(images):
    """Defines the discriminator network
    Args:
        images: input images as 28x28 tensors
    Returns:
        Logits and prediction for each image
    """
    with tf.variable_scope(DISC_VARIABLE_SCOPE, reuse=tf.AUTO_REUSE):
        net = tf.reshape(images, [-1, 28 * 28])
        for layer in DISC_HIDDEN_LAYERS:
            net = tf.layers.dense(net, layer, activation=tf.nn.relu)

        logits = tf.layers.dense(net, 1)
        prediction = tf.nn.sigmoid(logits)
        return logits, prediction


def _parse_function(filename, label):
    """
    Reads an image from a file, decodes it into a dense tensor, and resizes it to a fixed shape.
    """
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_png(image_string)
    image_resized = tf.reshape(image_decoded, [28, 28])
    return tf.cast(image_resized, tf.float32) / 255, label


def _mnist_filenames_and_labels():
    """
    Returns:
        A tuple of lists, where the first list contains the mnist png file paths, and the second list contains the
        label for each image.
    """
    images_paths = []
    images_labels = []
    for label in range(10):
        images_dir = os.path.join(TRAINING_DATA_DIR, str(label))
        current_images_paths = os.listdir(images_dir)
        images_paths += list(
            map(lambda image_path: os.path.join(images_dir, image_path),
                current_images_paths))
        images_labels += [label] * len(current_images_paths)
    return images_paths, images_labels


def shuffle(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


filenames, labels = _mnist_filenames_and_labels()
filenames, labels = shuffle(np.array(filenames), np.array(labels))

step = tf.train.get_or_create_global_step()
increment_step = tf.assign(step, step + 1)

dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
dataset = dataset.map(_parse_function)
dataset = dataset.shuffle(buffer_size=1000)

# The other half of the batch will come from the generator.
dataset = dataset.batch(batch_size=BATCH_SIZE // 2)

dataset = dataset.repeat()

iterator = dataset.make_one_shot_iterator()

# Iterator for tuples of images and labels
next = iterator.get_next()

latent_space = tf.placeholder(tf.float32, shape=[None, LATENT_SPACE_SHAPE])
G_images = generator(latent_space)

D_fake_logits, D_fake_pred = discriminator(G_images)
D_real_logits, D_real_pred = discriminator(next[0])

G_expected = tf.zeros_like(D_fake_logits)
G_loss = tf.losses.sigmoid_cross_entropy(G_expected, D_fake_logits)

D_real_expected = tf.zeros_like(D_real_logits)
D_fake_expected = tf.ones_like(D_fake_logits)

D_real_loss = tf.losses.sigmoid_cross_entropy(D_real_expected, D_real_logits)
D_fake_loss = tf.losses.sigmoid_cross_entropy(D_fake_expected, D_fake_logits)
D_loss = D_real_loss + D_fake_loss

G_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                GEN_VARIABLE_SCOPE)

print("G_variables ", G_variables)

G_optimizer = tf.train.GradientDescentOptimizer(
    learning_rate=GEN_LEARNING_RATE).minimize(G_loss, var_list=G_variables)

D_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                DISC_VARIABLE_SCOPE)

print("D_variables ", D_variables)

D_optimizer = tf.train.GradientDescentOptimizer(
    learning_rate=DISC_LEARNING_RATE).minimize(D_loss, var_list=D_variables)

