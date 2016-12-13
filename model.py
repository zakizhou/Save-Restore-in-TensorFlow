from __future__ import absolute_import
from __future__ import print_function
from __future__ import division


import tensorflow as tf


DIM = 784
NUM_CLASSES = 10


def inference(inputs):
    with tf.variable_scope("inference"):
        softmax_w = tf.get_variable(name="softmax_w",
                                    shape=[DIM, NUM_CLASSES],
                                    initializer=tf.truncated_normal_initializer(stddev=0.05),
                                    dtype=tf.float32)
        softmax_b = tf.get_variable(name="softmax_b",
                                    shape=[NUM_CLASSES],
                                    initializer=tf.constant_initializer(value=0.),
                                    dtype=tf.float32)
        logits = tf.nn.xw_plus_b(inputs, softmax_w, softmax_b)
    return logits


def loss(logits, labels):
    loss_per_example = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels)
    loss_average = tf.reduce_mean(loss_per_example, name="loss")
    return loss_average


def train_op(loss):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=5e-3)
    return optimizer.minimize(loss, name="train_op")


def validate(logits, labels):
    predict = tf.argmax(logits, 1)
    equal = tf.equal(tf.cast(predict, tf.int32), labels)
    accuracy = tf.reduce_mean(tf.cast(equal, tf.float32), name="validation")
    return accuracy