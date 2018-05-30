#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Las líneas anteriores nos permiten trabajar y poner comentarios en UTF-8

# Importamos las librerías que vamos a usar
import tensorflow as tf
# TensorFlow
import pandas as pd  # Pandas, para manipulación de datos
# numpy, biblioteca para manejo de números y operaciones matemáticas en python
import numpy as np

from sklearn.datasets import load_iris  # Esta función permitirá importar directamente
# el CSV en un par de arrays llamados “data” y “target”
from sklearn.model_selection import train_test_split  # función que permite partir un
# array en dos, para armar conjuntos de entrenamiento y prueba
import matplotlib.pyplot as plt  # libreria para graficar funciones en python

# Primero que nada cargamos los datos en el CSV, lo almacenamos en una variable iris
iris = load_iris()

# Las variables linguisticas:“Iris-setosa”, “Iris-versicolor”, “Iris-virginica”
# las importa como números: 0,1,2
iris_X, iris_y = iris.data[:-1, :], iris.target[:-1]

# convertimos los datos de salida, en vectores, así 0 será [1,0,0],
# 1 será [0,1,0] y 2 será [0,0,1]
iris_y = pd.get_dummies(iris_y).values

# Finalmente partimos el conjunto en 0.66 para entrenamiento y 0.33 para testing.
# Además añadimos aleatoriedad a la muestra
trainX, testX, trainY, testY = train_test_split(iris_X, iris_y, test_size=0.33,
                                                random_state=42)

# cantidad de valores de entrada, que son 4
numFeatures = trainX.shape[1]

# cantidad de valores de salida, en este caso 3. ya que convertimos las etiquetas
# linguisticas en vectores de 3 dimensiones
numLabels = trainY.shape[1]

# Creamos los Placeholders
###########################

# 'None' significa que tensorFlow no esperará un número fijo en esa dimensión.
# X representará a los valores de entrada
X = tf.placeholder(tf.float32, [None, numFeatures])

# yGold es la respuesta esperada.
yGold = tf.placeholder(tf.float32, [None, numLabels])

# Pesos y umbrales
##################

# Creamos la matriz de pesos, de 4x3 ya que 4 son los valores de entrada
# y 3 los de salida. Es una variable ya que es la que se irá ajustando en cada ciclo.
weights = tf.Variable(tf.random_normal([numFeatures, numLabels],
                                       mean=0,
                                       stddev=0.01,
                                       name="weights"))

# Creamos la variable para los umbrales (bias). Esta de 3 dimensiones
bias = tf.Variable(tf.random_normal([1, numLabels],
                                    mean=0,
                                    stddev=0.01,
                                    name="bias"))

# Grafo de tensorFlow
####################

# Ahora sí creamos el grafo de Tensor flow que hará lo siguiente:
# y = sigma(WX + b) en donde W es la matriz de pesos (4x3), X son los valores de
# entrada (4) y b los umbrales (3)

# multiplicación de matrices
apply_weights_OP = tf.matmul(X, weights, name="apply_weights")

# suma de umbrales
add_bias_OP = tf.add(apply_weights_OP, bias, name="add_bias")

# función SIGMA
activation_OP = tf.nn.sigmoid(add_bias_OP, name="activation")

##Definimos parámetros de entrenamiento
######################################

# ciclos de entrenamiento
numEpochs = 700

# definimos el decaimiento de la tasa de aprendizaje, será una exponencial
learningRate = tf.train.exponential_decay(learning_rate=0.0008,
                                          global_step=1,
                                          decay_steps=trainX.shape[0],
                                          decay_rate=0.95,
                                          staircase=True)

# El error o costo será la salida de "activation_OP"
# menos lo que definimos como yGold
cost_OP = tf.nn.l2_loss(activation_OP - yGold, name="squared_error_cost")

# Definimos método de entrenamiento: Descenso por gradiente
training_OP = tf.train.GradientDescentOptimizer(learningRate).minimize(cost_OP)

# Sesion de TensorFlow
######################

# Creamos session
sess = tf.Session()

# Inicializar variables
init_OP = tf.global_variables_initializer()
sess.run(init_OP)

# argmax(activation_OP, 1) devuelve la etiqueta con más probabilidad
# argmax(yGold, 1) devuelve la etiqueta correcta siempre
correct_predictions_OP = tf.equal(tf.argmax(activation_OP, 1), tf.argmax(yGold, 1))

# calculamos la precisión de nuestro estimador como el promedio
# entre las precisiones correctas. 1 es verdadero y 0 falso según lo anterior,
# ya que es 1 si la precisión fue igual al valor esperado
accuracy_OP = tf.reduce_mean(tf.cast(correct_predictions_OP, "float"))

################## CICLO ENTRENAMIENTO ##################
#########################################################

# Inicialización de las variables que vamos a utilizar para ir armando un reporte
cost = 0
diff = 1
epoch_values = []
accuracy_values = []
cost_values = []

# Entrenamos el grafo de TensorFlow
for i in range(numEpochs):
    if i > 1 and diff < .0001:
        print("El cambio en el costo es %g; => hay convergencia." % diff)
        # se hace un break ya que la diferencia entre los últimos
        # errores calculados es muy chica, es decir converge
        break
    else:
        # Corremos el grafo que es: "training_OP" y lo alimentamos
        # con los valores de entrenamiento del CSV
        step = sess.run(training_OP, feed_dict={X: trainX, yGold: trainY})
        # cada 10 ciclos reportamos algo
        if i % 10 == 0:
            # epoc actual
            epoch_values.append(i)
            # obtenemos precisión y costo
            train_accuracy, newCost = sess.run([accuracy_OP, cost_OP], feed_dict={X:
                                                                                      trainX, yGold: trainY})
            # añadimos la precisión a la lista para graficar luego
            accuracy_values.append(train_accuracy)
            # añadimos el costo a la lista de costos para graficar luego
            cost_values.append(newCost)
            # asignamos el último costo y la diferencia con el anterior
            diff = abs(newCost - cost)
            cost = newCost
            # Imprimimos el reporte
            print("paso %d, precisión de entrenamiento %g, costo %g, cambio en el costo % g" % (
                i, train_accuracy, newCost, diff))

# ahora, lo probamos con el conjunto de prueba
print("prueba final de precisión en el conjunto de prueba: %s"
      % str(sess.run(accuracy_OP, feed_dict={X: testX, yGold: testY})))

# Graficamos como fue cayendo el error
plt.plot([np.mean(cost_values[i - 50:i]) for i in range(len(cost_values))])
plt.show()
