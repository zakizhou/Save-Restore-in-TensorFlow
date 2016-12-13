import tensorflow as tf
import numpy as np

matrix = tf.convert_to_tensor(np.random.uniform(0, 1, (4, 3)), dtype=tf.float32, name="matrix")
with tf.variable_scope("predict"):
    variable = tf.get_variable(name="variable",
                               shape=[3, 2],
                               initializer=tf.truncated_normal_initializer(stddev=0.05),
                               dtype=tf.float32)
    product = tf.matmul(matrix, variable, name="product")
sess = tf.Session()
sess.run(tf.initialize_all_variables())
graph_def = sess.graph_def
print(sess.run(product))
tf.train.write_graph(graph_def=graph_def, logdir="test", name="test.pb", as_text=False)
