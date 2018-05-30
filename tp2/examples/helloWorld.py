import tensorflow as tf
import os

# Don't show warning messages
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

a = tf.constant([2])
b = tf.constant([3])

c = tf.add(a, b)


with tf.Session() as session:
    result = session.run(c)
    print(result)
