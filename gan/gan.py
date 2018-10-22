from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import cv2

TRAINING_DATA_DIR = os.path.join("data", "mnist_png", "training")
CHECKPOINT_DIR = "checkpoints"

# Number of inputs counting both mnist data and generated data for the discriminator, and number of random inputs for
# the generator
BATCH_SIZE = 60

GEN_LEARNING_RATE = 1e-4
DISC_LEARNING_RATE = 1e-4

LATENT_SPACE_SHAPE = 100

GEN_VARIABLE_SCOPE = "generator"
DISC_VARIABLE_SCOPE = "discriminator"

MAX_STEPS = 1000000
EPOCHS = 150


def generator(latent_space, label, training=True):
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


def discriminator(images, label, training=True):
    """Defines the discriminator network
    Args:
        images: input images as 28x28 tensors
        label: 10 dimensioanl one-hot tensor
    Returns:
        Logits and prediction for each image
    """
    with tf.variable_scope(DISC_VARIABLE_SCOPE, reuse=tf.AUTO_REUSE):
        net = images
        net = tf.layers.conv2d(net, 64, kernel_size=(5, 5), strides=(2, 2),
                               activation=tf.nn.leaky_relu, padding='same')
        net = tf.layers.dropout(net, training=training)

        net = tf.layers.conv2d(net, 128, kernel_size=(5, 5), strides=(2, 2),
                               activation=tf.nn.leaky_relu, padding='same')
        net = tf.layers.dropout(net, training=training)

        net_shape = net.shape
        net_reshaped = tf.reshape(net, [-1,
                                        net_shape[1] * net_shape[2] * net_shape[
                                            3]])
        net_with_label = tf.concat([net_reshaped, label], axis=1)
        logits = tf.layers.dense(net_with_label, 1,
                             activation=None)

        return logits


def _parse_function(filename, label):
    """
    Reads an image from a file, decodes it into a dense tensor, and resizes it to a fixed shape.
    """
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_png(image_string)
    image_resized = tf.reshape(image_decoded, [28, 28, 1])
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
G_label = tf.placeholder(tf.int32, shape=[None])
G_label_one_hot = tf.one_hot(G_label, 10)
G_images = generator(latent_space, G_label_one_hot)

D_fake_logits = discriminator(G_images, G_label_one_hot)

real_image = next[0]
real_label = tf.one_hot(next[1], 10)
D_real_logits = discriminator(real_image, real_label)

G_expected = tf.ones_like(D_fake_logits)
G_loss = tf.losses.sigmoid_cross_entropy(G_expected, D_fake_logits)

D_real_expected = tf.ones_like(D_real_logits)
D_fake_expected = tf.zeros_like(D_fake_logits)

D_real_loss = tf.losses.sigmoid_cross_entropy(D_real_expected, D_real_logits)
D_fake_loss = tf.losses.sigmoid_cross_entropy(D_fake_expected, D_fake_logits)
D_loss = D_real_loss + D_fake_loss

G_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                GEN_VARIABLE_SCOPE)

G_optimizer = tf.train.AdamOptimizer(
    learning_rate=GEN_LEARNING_RATE).minimize(G_loss, var_list=G_variables)

D_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                DISC_VARIABLE_SCOPE)

D_optimizer = tf.train.AdamOptimizer(
    learning_rate=DISC_LEARNING_RATE).minimize(D_loss, var_list=D_variables)

tf.summary.scalar("Gen loss", G_loss, family="Generator")
tf.summary.scalar("Disc loss", D_loss, family="Discriminator")
tf.summary.image("Gen images", G_images, max_outputs=1)


def _generator_step(sess):
    latent_space_np = np.random.randn(BATCH_SIZE, LATENT_SPACE_SHAPE)
    label = np.random.randint(10, size=BATCH_SIZE)
    _, G_loss_np = sess.run([G_optimizer, G_loss],
                            feed_dict={latent_space: latent_space_np,
                                       G_label: label})
    if sess.run(step) % 97 == 0:
        print()
        print("Step: ", sess.run(step))
        print("G_loss: ", G_loss_np)
    sess.run(increment_step)


def _discriminator_step(sess):
    latent_space_np = np.random.randn(BATCH_SIZE // 2, LATENT_SPACE_SHAPE)
    label = np.random.randint(10, size=BATCH_SIZE // 2)
    _, D_loss_np = sess.run([D_optimizer, D_loss],
                            feed_dict={latent_space: latent_space_np,
                                       G_label: label})
    if sess.run(step) % 97 == 0:
        print()
        print("Step: ", sess.run(step))
        print("D_loss: ", D_loss_np)
    sess.run(increment_step)


hooks = [tf.train.StopAtStepHook(num_steps=MAX_STEPS)]

with tf.train.MonitoredTrainingSession(checkpoint_dir=CHECKPOINT_DIR,
                                       hooks=hooks) as sess:
    while not sess.should_stop():
        _generator_step(sess)
        _discriminator_step(sess)
