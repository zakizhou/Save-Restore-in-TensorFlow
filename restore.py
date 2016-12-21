from __future__ import absolute_import
from __future__ import print_function
from __future__ import division


from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from train import BATCH_SIZE
IMPORT_SCOPE = 'restore'

test = input_data.read_data_sets("/home/windows98/TensorFlow/mnist_data").test
for _ in range(24):
    imgs, lbs = test.next_batch(BATCH_SIZE)
imgs = tf.convert_to_tensor(imgs, dtype=tf.float32, name="imgs")
lbs = tf.convert_to_tensor(lbs, dtype=tf.int32, name="lbs")

sess = tf.Session()
# the core point is that we can change the inputs of the original model, here I change
# the inputs and labels from `placeholder` into two constant tensors, so next time I run
# train or validate op, there's no need to add feed_dict to arguments. In real world datasets,
# we often train with tfrecords which is the dequeue outputs of input queues, afher training
# when do validate, we can also change the inputs into placeholder
saver = tf.train.import_meta_graph("protobuf/model.ckpt.meta",
                                   import_scope=IMPORT_SCOPE,
                                   input_map={
                                       "inputs:0": imgs,
                                       "labels:0": lbs
                                   })
saver.restore(sess, "protobuf/model.ckpt")
graph = sess.graph

validation = graph.get_tensor_by_name("restore/validation:0")

ops = graph.get_operations()
print("\n".join([op.name for op in ops]))

# sess.run(init)
accuracy = sess.run(validation)
print(accuracy)

