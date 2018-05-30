#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Las líneas anteriores nos permiten trabajar y poner comentarios en UTF-8
import numpy as np
import tensorflow as tf
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

# generamos aleatoriamente 100 números entre 0 y 1, este vector tendrá los números en X
x_data = np.random.rand(100).astype(np.float32)

# definimos un nuevo vector que son los números en Y (dados por la ecuación y=x*3+2)
y_data = x_data * 3 + 2

# Esto agrega ruido a los números en X
y_data = np.vectorize(lambda y: y + np.random.normal(loc=0.0, scale=0.1))(y_data)

# si quieren graficar los puntos, ejecuten las siguientes dos líneas:
plt.scatter(x_data, y_data)
plt.show()

# definimos dos variables a y b de TensorFlow y las inicializamos con un valor
# cualquiera
a = tf.Variable(1.0)
b = tf.Variable(0.2)

# definimos y como la ecuación entre estas variables y todos nuestros X
y = a * x_data + b

# definimos ahora el error (loss) como el error cuadrático medio de la diferencia.
# ​ tf.reduce_mean()​ básicamente calcula el promedio de todo el tensor y devuelve un valor
#  escalar, un tensor de dimensión 1 que es el promedio de todos los valores
loss = tf.reduce_mean(tf.square(y - y_data))

# ahora viene la magia, tenemos que definir un método de optimización, un método que
# optimice el error (la variable loss), usamos descenso por gradiente con una tasa de
# aprendizaje de 0.5
optimizer = tf.train.GradientDescentOptimizer(0.5)
# le decimos al método que minimice loss
train = optimizer.minimize(loss)

#inicializamos variables
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

#corremos el algoritmos de entrenamiento
train_data = []                         #definimos un array vacío
for step in range(100):                 # entrenamos por 100 ciclos
    evals = sess.run([train,a,b])[1:]   #ejecutamos train y guardamos la salida
    if step % 5 == 0:                   #cada 5 ejecuciones....
        print(step, evals)              #imprimimos el resultado
    train_data.append(evals)

#graficamos las rectas intermedias (que guardamos en ​ train_data​ )y la recta final
#hallada junto con los puntos
converter = plt.colors
cr, cg, cb = (0.1, 1.0, 0.0)
for f in train_data:
    cb += 1.0 / len(train_data)
    cg -= 1.0 / len(train_data)
    if cb > 1.0: cb = 1.0
    if cg < 0.0: cg = 0.0
    [a, b] = f
    f_y = np.vectorize(lambda x: a*x + b)(x_data)
    line = plt.plot(x_data, f_y)
    plt.setp(line, color=(cr,cg,cb))

plt.plot(x_data, y_data, 'ro')
green_line = mpatches.Patch(color='red', label='Data Points')
plt.legend(handles=[green_line])
plt.show()