from __future__ import absolute_import
from __future__ import print_function
from __future__ import division


from tensorflow.examples.tutorials.mnist import input_data
# from model import inference, loss, train_op, validate, DIM, NUM_CLASSES
import tensorflow as tf
from train import BATCH_SIZE


# sess = tf.Session()
# saver = tf.train.import_meta_graph("protobuf/model.meta")
# saver.restore(sess, "protobuf/model")
# graph = sess.graph

test = input_data.read_data_sets("/home/windows98/TensorFlow/mnist_data").test
imgs, lbs = test.next_batch(BATCH_SIZE)
imgs = tf.convert_to_tensor(imgs, dtype=tf.float32)
lbs = tf.convert_to_tensor(lbs, dtype=tf.int32)
f = tf.gfile.FastGFile("protobuf/train.pb", 'r')
graph_def = tf.GraphDef()
graph_def.ParseFromString(f.read())
train_op, validation, loss = tf.import_graph_def(graph_def=graph_def,
                                                 return_elements=["train_op", "validation:0", "loss:0"],
                                                 input_map={
                                                    "inputs:0": imgs,
                                                    "labels:0": lbs
                                                })
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

