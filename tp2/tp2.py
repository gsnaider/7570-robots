#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Las líneas anteriores nos permiten trabajar y poner comentarios en UTF-8

from __future__ import print_function

# Don't show warning messages
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split

# Parámetros ​ usados​ para entrenar la red
learning_rate = 0.1  # tasa de aprendizaje
num_steps = 1000  # cantidad de pasos de entrenamiento
batch_size = 30  # cantidad de ejemplos por paso

# Parámetros para la construcción de la red
n_hidden_1 = 20  # número de neuronas en la capa oculta 1
n_hidden_2 = 20  # número de neuronas en la capa oculta 2
num_classes = 2  # clases: sobrevivio (0-1)


# Definimos la red neuronal
def neural_net(x_dict):
    # x_dic es un diccionario con los valores de entrada
    x = x_dict['data']  # en particular vendrán en el campo “data”
    # Conectamos x (la entrada) con la capa oculta 1: Conexión full
    layer_1 = tf.layers.dense(x, n_hidden_1)
    # Conectamos la capa oculta 1, con la capa oculta 2: Conexión full
    layer_2 = tf.layers.dense(layer_1, n_hidden_2)
    # Conectamos la capa oculta 2 con la capa de salida
    out_layer = tf.layers.dense(layer_2, num_classes)
    return out_layer


# Usamos la clase “TF Estimator Template”, para definir cómo será el entrenamiento
def model_fn(features, labels, mode):
    # Llamamos a la función anterior para construir la red
    logits = neural_net(features)

    # Predicciones
    pred_classes = tf.argmax(logits, axis=1)
    pred_probas = tf.nn.softmax(logits)

    # Si es de predicción devolvemos directamente un EstimatorSpec
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=pred_classes)

    # Definimos nuestro error
    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=tf.cast(labels, dtype=tf.int32)))
    # ​ sparse_softmax_cross_entropy_with_logits​ : Mide el error de probabilidad
    # en tareas de clasificación discretas en las que las clases son mutuamente
    # excluyentes (cada entrada está en exactamente una clase)
    # ​ reduce_mean​ : Calcula la media de los elementos a través de las dimensiones
    # de un tensor

    # Definimos un optimizador, que trabaja por el metodo de descenso por gradiente
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op, global_step=tf.train.get_global_step())

    # Definimos cómo se evaluará la precisión del modelo
    acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)
    # Finalmente devolvemos un objeto: “EstimatorSpec”, indicando todo lo que
    # calculamos para el entrenamiento: modo, predicción, error (loss), método de
    # entrenamiento y métricas
    estim_specs = tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=pred_classes,
        loss=loss_op,
        train_op=train_op,
        eval_metric_ops={'accuracy': acc_op})

    return estim_specs


def get_test_data(path, mean, std):
    df = pd.read_csv(path)
    data = df[['Age', 'SibSp', 'Parch', 'Fare']]

    # Normalizar datos
    data = (data - mean) / std

    # Mapeamos Sex y Pclass a vectores binarios
    data = data.join(pd.get_dummies(df['Sex']))
    data = data.join(pd.get_dummies(df['Pclass']))

    # Seteamos el promedio del campo en los datos faltantes
    data = data.fillna(data.mean())

    return data.values


def get_data(path):
    df = pd.read_csv(path)
    data = df[['Age', 'SibSp', 'Parch', 'Fare']]

    mean = data.mean()
    std = data.std()

    # Normalizar datos
    data = (data - mean) / std

    # Mapeamos Sex y Pclass a vectores binarios
    data = data.join(pd.get_dummies(df['Sex']))
    data = data.join(pd.get_dummies(df['Pclass']))

    # Seteamos el promedio en los datos faltantes
    data = data.fillna(data.mean())

    data_X = data.values
    data_y = df['Survived'].values

    # Devolvemos X e y, junto con la media y la desviación estandard
    # para poder luego ajustar el set de test con esos mismos parámetros.
    return (data_X, data_y, mean, std)


(data_X, data_y, mean, std) = get_data("Titanic/train.csv")

trainX, testX, trainY, testY = train_test_split(data_X, data_y, test_size=0.33, random_state=42)

# Construimos un estimador, le decimos que use la función antes definida
model = tf.estimator.Estimator(model_fn)

# pasamos ahora todos los parámetros que necesita la función definida
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'data': trainX}, y=trainY,
    batch_size=batch_size, num_epochs=None, shuffle=True)

# Entrenamos el modelo
model.train(input_fn, steps=num_steps)

# Definimos la entrada para evaluar
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'data': testX}, y=testY,
    batch_size=batch_size, shuffle=False)

# Usamos el método 'evaluate'del modelo
e = model.evaluate(input_fn)

print("Precisión en el conjunto de prueba:", e['accuracy'])
print("Costo final:", e['loss'])

# Descomentar para evaluar en set de datos de test e imprimir predicciones
'''
data = get_test_data("Titanic/test.csv", mean, std)

input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'data': data},
    batch_size=batch_size, shuffle=False)

for pred in model.predict(input_fn):
    print(pred)
'''intelli