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

def _parse_function(filename, label):
    """
    Reads an image from a file, decodes it into a dense tensor, and resizes it to a fixed shape.
    """
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_png(image_string)
    image_resized = tf.reshape(image_decoded, [28, 28])
    return image_resized, label


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
        images_paths += list(map(lambda image_path : os.path.join(images_dir, image_path), current_images_paths))
        images_labels += [label] * len(current_images_paths)
    return images_paths, images_labels

def shuffle(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


filenames, labels = _mnist_filenames_and_labels()
filenames, labels = shuffle(np.array(filenames), np.array(labels))

dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
dataset = dataset.map(_parse_function)
dataset = dataset.shuffle(buffer_size=1000)

# The other half of the batch will come from the generator.
dataset = dataset.batch(batch_size= BATCH_SIZE // 2)

dataset = dataset.repeat()

iterator = dataset.make_one_shot_iterator()
next = iterator.get_next()

with tf.Session() as sess:
    images, labels = sess.run(next)
    for i in range(10):
        image = images[i]
        print(labels[i])
        plt.imshow(image, cmap='gray')
        plt.show()