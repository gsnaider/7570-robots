import tensorflow as tf
import os

# Don't show warning messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

Matrix_one = tf.constant([[1,2,3],[2,3,4],[3,4,5]])
Matrix_two = tf.constant([[2,2,2],[2,2,2],[2,2,2]])

#SUMA DE MATRICES
first_operation = tf.add(Matrix_one, Matrix_two)

#SUMA DE MATRICES USANDO EL OPERADOR +
second_operation = Matrix_one + Matrix_two

#MULTIPLICACION DE MATRICES
third_operation = tf.matmul(Matrix_one, Matrix_two)

with tf.Session() as session:
    result = session.run(first_operation)
    print "Defined using tensorflow function :"
    print(result)
    result = session.run(second_operation)
    print "Defined using normal expressions:"
    print(result)
    result = session.run(third_operation)
    print "Defined using tensorflow function :"
    print(result)

