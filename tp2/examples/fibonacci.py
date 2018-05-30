# -*- coding: utf-8 -*-

import tensorflow as tf
import os

# Don't show warning messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Traditional form
a = tf.Variable(0)
b = tf.Variable(1)
temp = tf.Variable(0)
c = a + b

update1 = tf.assign(temp, c)
update2 = tf.assign(a, b)
update3 = tf.assign(b, temp)

init_op = tf.initialize_all_variables()
with tf.Session() as s:
    s.run(init_op)
    for _ in range(15):
        print(s.run(a))
        s.run(update1)
        s.run(update2)
        s.run(update3)


# With tensors
f = [tf.constant(1), tf.constant(1)]
for i in range(2, 10):
    temp = f[i - 1] + f[i - 2]
    f.append(temp)

with tf.Session() as sess:
    result = sess.run(f)
    print result
