# Save-Restore-in-TensorFlow

note on save and restore pipeline in tensorflow

##Save:
```
saver.save(sess=sess, save_path="", write_meta_graph=True)
```
The above function internally calls `tf.train.export_meta_graph` while this
function internally calls `tf.write_graph` (only write graph no weights saved)

##Restore:
a `meta_graph_def` can be read from file with `read_meta_graph_file` function in the
`tensorflow.python.training.saver` module.i.e
```
from tensorflow.python.training.saver import read_meta_graph_file
meta_graph_def = read_meta_graph_file("path/to/file.meta")
```
When obtained a `meta_graph_def`, we can import a graph with `tf.train.import_meta_graph` which does all
the dirty work for you(including restore weights and graph structure)
this function internally calls `tf.import_graph_def` function which only restores graph and `saver.restore` 
which only restores weights of this graph

**this function can only be used in tensorflow version 0.12 or later**

##TODO:
new api `tf.saved_model` is avaiable in tensorflow 1.0, add example for its usage


