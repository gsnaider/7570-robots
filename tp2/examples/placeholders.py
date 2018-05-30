# -*- coding: utf-8 -*-

import tensorflow as tf
import os

# Don't show warning messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# creamos un placeholder
a = tf.placeholder(tf.float32)

# definimos una multiplicacion
b = a * 2

# Se llama a la función run de la sesión como siempre, pero se le pasa un
# argumento extra: ​ feed_dict​ , en el cual se le pasará un ​ diccionario​ ,
# en donde cada placeholder es seguido de sus datos:
with tf.Session() as sess:
    result = sess.run(b, feed_dict={a: 3.5})
    print(result)

dictionary = {a: [[[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], [
    [13, 14, 15], [16, 17, 18], [19, 20, 21], [22, 23, 24]]]}
with tf.Session() as sess:
    result = sess.run(b, feed_dict=dictionary)
    print(result)
