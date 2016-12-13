import tensorflow as tf
import numpy as np

matrix_o = tf.convert_to_tensor(np.random.uniform(0, 1, (4, 3)), dtype=tf.float32, name="matrix_o")
f = tf.gfile.FastGFile("test/test.pb", 'r')
graph_def = tf.GraphDef()
graph_def.ParseFromString(f.read())
product = tf.import_graph_def(graph_def=graph_def,
                              input_map={
                                  "matrix:0": matrix_o
                              },
                              return_elements=['predict/product:0'])
sess = tf.Session()
tf.get_variable_scope().reuse_variables()
sess.run(tf.initialize_all_variables())
tf.train.import_meta_graph()
tf.import_graph_def()