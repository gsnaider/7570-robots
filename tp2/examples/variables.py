import tensorflow as tf
import os

# Don't show warning messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

state = tf.Variable(0)
one = tf.constant(1)

new_value = tf.add(state, one)

update = tf.assign(state, new_value)    

# inicializamos las variables (es tambien una operacion que
# debe ser ejecutada en la sesion)
init_op = tf.global_variables_initializer()


with tf.Session() as session:
    session.run(init_op)
    print("originalmente state es:" + str(session.run(state)))
    for _ in range(3):
        session.run(update)         # se ejecuta UPDATE 3 veces
        print(session.run(state))   # se imprime el valor de state 3 veces
