from __future__ import absolute_import
from __future__ import print_function
from __future__ import division


from tensorflow.examples.tutorials.mnist import input_data
from model import inference, loss, train_op, validate, DIM, NUM_CLASSES
import tensorflow as tf


BATCH_SIZE = 64
NUM_FEATURES = 784
NUM_STEPS = 300


def main():
    mnist_data = input_data.read_data_sets("/home/windows98/TensorFlow/mnist_data").train
    valid_input = input_data.read_data_sets("/home/windows98/TensorFlow/mnist_data").validation
    inputs = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, NUM_FEATURES], name="inputs")
    labels = tf.placeholder(dtype=tf.int32, shape=[BATCH_SIZE], name="labels")
    logits = inference(inputs)
    validation_accuracy = validate(logits, labels)
    cross_entropy = loss(logits, labels)
    train = train_op(cross_entropy)
    # print(inputs.name)
    # print(labels.name)
    # print(logits.name)
    # print(cross_entropy.name)
    # print(train.name)
    init = tf.initialize_all_variables()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        try:
            for i in range(NUM_STEPS):
                feed_inputs, feed_labels = mnist_data.next_batch(BATCH_SIZE)
                _, loss_value = sess.run([train, cross_entropy], feed_dict={
                    inputs: feed_inputs,
                    labels: feed_labels
                })
                print("step: %d loss is :%f" % (i + 1, loss_value))
                if (i + 1) % 5 == 0:
                    feed_valid_inputs, feed_valid_labels = valid_input.next_batch(BATCH_SIZE)
                    valid_accuracy = sess.run(validation_accuracy, feed_dict={
                        inputs: feed_valid_inputs,
                        labels: feed_valid_labels
                    })
                    print("validation accuracy: %f" % valid_accuracy)
                    if valid_accuracy > 0.75:
                        tf.train.write_graph(sess.graph_def, "protobuf", "train.pb", False)
                        saver.save(sess=sess, save_path="protobuf/model.ckpt")
                        saver.export_meta_graph("protobuf.meta")
                        break
        except KeyboardInterrupt:
            print("stopping!")
            del sess


if __name__ == "__main__":
    main()