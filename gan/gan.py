from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import cv2

TRAINING_DATA_DIR = os.path.join("data", "mnist_png", "training")



def _parse_function(filename, label):
    """
    Reads an image from a file, decodes it into a dense tensor, and resizes it to a fixed shape.
    """
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_png(image_string)
    image_resized = tf.image.resize_images(image_decoded, [28, 28])
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

filenames, labels = _mnist_filenames_and_labels()

dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
dataset = dataset.map(_parse_function)

iterator = dataset.make_one_shot_iterator()
next_image, next_label = iterator.get_next()

with tf.Session() as sess:
    print(sess.run(next_label))
    image = sess.run(next_image)
    print(image.shape)
    plt.imshow(np.squeeze(image), cmap='gray')
    plt.show()