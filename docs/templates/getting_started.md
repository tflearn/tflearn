# Getting started with TFLearn

Here is a basic guide that introduces TFLearn and its functionalities. First highlighting TFLearn high-level API, for fast neural network building and training, and then showing how TFLearn layers, built-in ops and helpers can directly benefit any model implementation with Tensorflow.

# High-Level API usage

TFLearn introduces a High-Level API that makes neural network building and training fast and easy. This API is intuitive and fully compatible with Tensorflow.

### Layers

Layers are a core feature of TFLearn. While completely defining a model using Tensorflow ops can be time consuming and repetitive, TFLearn brings "layers" that represent an abstract set of operations to make building neural networks more convenient. For example, a convolutional layer will:

- Create and initialize weights and biases variables
- Apply convolution over incoming tensor
- Add an activation function after the convolution
- etc...

In Tensorflow, write those kind of operation set can be quite fastidious:

```python
with tf.name_scope('conv1'):
    W = tf.Variable(tf.random_normal([5, 5, 1, 32]), dtype=tf.float32, name='Weights')
    b = tf.Variable(tf.random_normal([32]), dtype=tf.float32, name='biases')
    x = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    x = tf.add_bias(W, b)
    x = tf.nn.relu(x)
```

While in TFLearn, it only takes a line:
```python
tflearn.conv_2d(x, 32, 5, activation='relu', name='conv1')
```

Here is a list of all currently available layers:

File | Layers
-----|-------
[core](http://tflearn.org/layers/core/) | input_data, fully_connected, dropout, custom_layer, reshape, flatten, activation, single_unit
[conv](http://tflearn.org/layers/conv/) | conv_2d, conv_2d_transpose, max_pool_2d, avg_pool_2d, conv_1d, max_pool_1d, avg_pool_1d, shallow_residual_block, deep_residual_block
[recurrent](http://tflearn.org/layers/recurrent/) | simple_rnn, lstm, gru, bidirectionnal_rnn, dynamic_rnn
[embedding](http://tflearn.org/layers/embedding/) | embedding
[normalization](http://tflearn.org/layers/normalization/) | batch_normalization, local_response_normalization
[merge](http://tflearn.org/layers/merge/) | merge, merge_outputs
[estimator](http://tflearn.org/layers/estimator/) | regression

### Built-in Operations

Besides layers concept, TFLearn also provides many different ops to be used while building a neural network. These ops are firstly mean to be used as part of the above 'layers' arguments, but they can also be used independently in any other Tensorflow graph for convenience.

File | Ops
-----|----
[activations](http://tflearn.org/activations) | linear, tanh, sigmoid, softmax, softplus, softsign, relu, relu6, leaky_relu, prelu, elu
[objectives](http://tflearn.org/objectives) | softmax_categorical_crossentropy, categorical_crossentropy, binary_crossentropy, mean_square, hinge_loss
[optimizers](http://tflearn.org/optimizers) | SGD, RMSProp, Adam, Momentum, AdaGrad, Ftrl
[metrics](http://tflearn.org/metrics) | Accuracy, Top_k, R2
[initializations](http://tflearn.org/initializations) | zeros, uniform, uniform_scaling, normal, truncated_normal
[losses](http://tflearn.org/losses) | l1, l2

In practice, the arguments (such as 'activation' or 'regularizer' of conv_2d) just require the op name. Below are some quick examples:

```python
# Activation and Regularization inside a layer:
fc2 = tflearn.dense(fc1, 32, activation='tanh', regularizer='L2')
# Equivalent to:
fc2 = tflearn.dense(fc1, 32)
tflearn.add_weights_regularization(fc2, loss='L2')
fc2 = tflearn.relu(fc2)

# Optimizer, Objective and Metric:
reg = tflearn.regression(fc4, optimizer='rmsprop', metric='accuracy', loss='categorical_crossentropy')
# Ops can also be defined outside, for deeper customization:
momentum = tflearn.optimizers.Momentum(learning_rate=0.1, weight_decay=0.96, decay_step=200)
top5 = tflearn.metrics.Top_k(k=5)
reg = tflearn.regression(fc4, optimizer=momentum, metric=top5, loss='categorical_crossentropy')
```

### Training, Evaluating & Predicting

Training functions are another core feature of TFLearn. In Tensorflow, there are no pre-built API to train a network, so TFLearn integrates a set of functions to easily handle any neural network training, whatever the number of inputs, outputs and optimizers.

If you are using TFlearn layers, many parameters are already self managed, so it is very easy to train a model, using `DNN` class:

```python
network = ... (some layers) ...
network = regression(network, optimizer='sgd', loss='categorical_crossentropy')

model = DNN(network)
model.fit(X, Y)
```

It can also directly be called for prediction, or evaluation:

```python
network = ...

model = DNN(network)
model.load('model.tflearn')
model.predict(X)
```

- To learn more about those wrappers, see: [dnn](http://tflearn.org/models/dnn) and [estimator](http://tflearn.org/layers/estimator).

### Visualization

While writing a Tensorflow model and adding tensorboard summaries isn't very practical, TFLearn has the ability to self managed a lot of useful logs. Currently, TFLearn training classes are supporting a verbose level to automatically manage summaries:

- 0: Loss & Metric (Best speed).
- 1: Loss, Metric & Gradients.
- 2: Loss, Metric, Gradients & Weights.
- 3: Loss, Metric, Gradients, Weights, Activations & Sparsity (Best Visualization).

Using `DNN` class, it is very simple, only specify the verbose level argument is required:
```python
model = DNN(network, tensorboard_verbose=3)
```

Then, you can run Tensorboard and visualize your network and its performance:

```
$ tensorboard --logdir='/tmp/tflearn_logs'
```

**Graph**

![Graph Visualization](./img/graph.png)

**Loss & Accuracy (multiple runs)**

![Layer Visualization](./img/loss_acc.png)

**Layers**

![Multiple Loss Visualization](./img/layer_visualization.png)

### Weights persistence

To save or restore a model, you can simply invoke 'save' or 'load' method of `DNN` class.

```python
# Save a model
model.save('my_model.tflearn')
# Load a model
model.load('my_model.tflearn')
```

### Fine-tuning

Fine-tune a pre-trained model on a new task might be useful in many cases. So, when defining a model in TFLearn, you can specify which layer's weights you want to be restored or not (when loading pre-trained model). This can be handle with the 'restore' argument of layer functions (only available for layers with weights).

```python
# Weights will be restored by default.
dense_layer = Dense(input_layer, 32)
# Weights will not be restored, if specified so.
dense_layer = Dense(input_layer, 32, restore='False')
```

All weights that doesn't need to be restored will be added to tf.GraphKeys.EXCL_RESTORE_VARS collection, and when loading a pre-trained model, these variables restoration will simply be ignored.
The following example shows how to fine-tune a network on a new task by restoring all weights except the last dense layer, and then train the new model on a new dataset:

- Fine-tuning example: [finetuning.py](https://github.com/tflearn/blob/master/tflearn/examples/basics/finetuning.py).

### Data management

TFLearn supports numpy array data. Additionally, it also supports HDF5 for handling large datasets. HDF5 is a data model, library, and file format for storing and managing data. It supports an unlimited variety of datatypes, and is designed for flexible and efficient I/O and for high volume and complex data ([more info](https://www.hdfgroup.org/HDF5/)). TFLearn can directly use HDF5 formatted data:

```python
# Load hdf5 dataset
h5f = h5py.File('data.h5', 'r')
X, Y = h5f['MyLargeData']

... define network ...

# Use HDF5 data model to train model
model = DNN(network)
model.fit(X, Y)
```

For an example, see: [hdf5.py](https://github.com/tflearn/blob/master/tflearn/examples/basics/hdf5.py).

### Graph Initialization

It might be useful to limit resources, or assigns more or less GPU RAM memory while training. To do so, a graph initializer can be used to configure a graph before run:

```python
tflearn.init_graph(set_seed=8888, num_cores=16, gpu_memory_fraction=0.5)
```

- See: [config](http://tflearn.org/config).

# Extending Tensorflow

TFLearn is a very flexible library designed to let you use any of its component independently. A model can be succinctly built using any combination of Tensorflow operations and TFLearn built-in layers and operations. The following instructions will show you the basics for extending Tensorflow with TFLearn.

### Layers

Any layer can be used with any other Tensor from Tensorflow, this means that you can directly use TFLearn wrappers into your own Tensorflow graph.

```python
# Some operations using Tensorflow.
X = tf.placeholder(shape=(None, 784), dtype=tf.float32)
net = tf.reshape(X, [-1, 28, 28, 1])

# Using TFLearn convolution layer.
net = tflearn.conv_2d(net, 32, 3, activation='relu')

# Using Tensorflow's max pooling op.
net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

...
```

- For an example, see: [layers.py](https://github.com/tflearn/blob/master/tflearn/examples/extending_tensorflow/layers.py).

### Built-in Operations

TFLearn built-in ops makes Tensorflow graphs writing faster and more readable. So, similar to layers, built-in ops are fully compatible with any TensorFlow expression. The following code example shows how to use them along with pure Tensorflow API.

- See: [builtin_ops.py](https://github.com/tflearn/blob/master/tflearn/examples/extending_tensorflow/builtin_ops.py).

Here is a list of available ops, click on the file for more details:

File | Ops
-----|----
[activations](http://tflearn.org/activations) | linear, tanh, sigmoid, softmax, softplus, softsign, relu, relu6, leaky_relu, prelu, elu
[objectives](http://tflearn.org/objectives) | softmax_categorical_crossentropy, categorical_crossentropy, binary_crossentropy, mean_square, hinge_loss
[optimizers](http://tflearn.org/optimizers) | SGD, RMSProp, Adam, Momentum, AdaGrad, Ftrl
[metrics](http://tflearn.org/metrics#accuracy_op) | accuracy_op, top_k_op, r2_op
[initializations](http://tflearn.org/initializations) | zeros, uniform, uniform_scaling, normal, truncated_normal
[losses](http://tflearn.org/losses) | l1, l2

Note:
- Optimizers are designed as class and not function, for usage outside of TFlearn models, check: [optimizers](http://tflearn.org/optimizers).

### Trainer / Evaluator / Predictor

If you are using you own Tensorflow model, TFLearn also provides some 'helpers' functions that can train any Tensorflow graph. It is suitable to make training more convenient, by introducing realtime monitoring, batch sampling, moving averages, tensorboard logs, data feeding, etc... It supports any number of inputs, outputs and optimization ops.

TFLearn implements a `TrainOp` class to represent an optimization process (i.e. backprop). It is defined as follow:

```python
trainop = TrainOp(net=my_network, loss=loss, metric=accuracy)
```

Then, all TrainOp can be feeded into a `Trainer` class, that will handle the whole training process, considering all TrainOp together as a whole model.

```python
model = Trainer(trainops=trainop, tensorboard_dir='/tmp/tflearn')
model.fit(feed_dict={input_placeholder: X, target_placeholder: Y})
```

While most models will only have a single optimization process, it can be useful for more complex models to handle multiple ones.

```python
model = Trainer(trainops=[trainop1, trainop2])
model.fit(feed_dict=[{in1: X1, label1: Y1}, {in2: X2, in3: X3, label2: Y2}])
```

- To learn more about TrainOp and Trainer, see: [trainer](http://tflearn.org/helpers/trainer).

- For an example, see: [trainer.py](https://github.com/tflearn/blob/master/tflearn/examples/extending_tensorflow/trainer.py).

For prediction, TFLearn implements a `Predictor` class that is working in a similar way as `Trainer`. It takes any network as parameter and return the predicted value.
```python
model = Predictor(network)
model.predict(feed_dict={input_placeholder: X})
```

- To learn more about Predictor class: [predictor](http://tflearn.org/helpers/predictor).

To handle network that behave differently at training and testing time (such as dropout and batch normalization), `Trainer` class uses a boolean variable ('training'), that specifies if the network is used for training or testing/predicting. This variable is stored under tf.GraphKeys.IS_TRAINING collection, as its first element.
So, when defining such ops, you need to add a condition to your op:

```python
# Example for Dropout:
x = ...

def apply_dropout(): # Function to apply when training mode ON.
  return tf.nn.dropout(x, keep_prob)

is_training = tflearn.get_training_mode() # Retrieve is_training variable.
tf.cond(is_training, apply_dropout, lambda: x) # Only apply dropout at training time.
```

To make it easy, TFLearn implements functions to retrieve that variable or change its value:

- See: [training config](http://tflearn.org/config#is_training).

### Variables

TFLearn defines a set of functions for users to quickly define variables.

While in Tensorflow, variable creation requires predifinied value or initializer, as well as an explicit device placement, TFLearn simplify variable definition:

```python
import tflearn.variables as vs
my_var = vs.variable('W',
                     shape=[784, 128],
                     initializer='truncated_normal',
                     regularizer='L2',
                     device='/gpu:0')
```

- For an example, see: [variables.py](https://github.com/tflearn/blob/master/tflearn/examples/extending_tensorflow/variables.py).

### Summaries

When using `Trainer` class, it is also very easy to manage summaries. It just additionally required that the activations to monitor are stored into `tf.GraphKeys.ACTIVATIONS` collection.

Then, simply specify verbose level to control visualization depth:
```python
model = Trainer(network, loss=loss, metric=acc, tensorboard_verbose=3)
```

Beside `Trainer` self-managed summaries option, you can also directly use TFLearn ops to quickly add summaries to your current Tensorflow graph.

```python
import tflearn.helpers.summarizer as s
s.summarize_variables(train_vars=[...]) # Summarize all given variables' weights (All trainable variables if None).
s.summarize_activations(activations=[...]) # Summarize all given activations
s.summarize_gradients(grads=[...]) # Summarize all given variables' gradient (All trainable variables if None).
s.summarize(value, type) # Summarize anything.
```

Every function above accepts a collection as parameter, and will return a merged summary over that collection (Default name: 'tflearn_summ'). So you just need to run the last summarizer to get the whole summary ops collection, already merged.

```python
s.summarize_variables(collection='my_summaries')
s.Summarize_gradients(collection='my_summaries')
summary_op = s.summarize_activations(collection='my_summaries')
# summary_op is a the merged op of previously define weights, gradients and activations summary ops.
```

- For an example, see: [summaries.py](https://github.com/tflearn/blob/master/tflearn/examples/extending_tensorflow/summaries.py).

### Regularizers

Add regularization to a model can be completed using TFLearn [regularizer](http://tflearn.org/helpers/regularizers) helpers. It currently supports weights and activation regularization. Available regularization losses can be found in [here](http://tflearn.org/helpers/regularizers). All regularization losses are stored into tf.GraphKeys.REGULARIZATION_LOSSES collection.

```python
# Add L2 regularization to a variable
W = tf.Variable(tf.random_normal([784, 256]), name="W")
tflearn.add_weight_regularizer(W, 'L2', weight_decay=0.001)
```


### Preprocessing

Besides tensor operations, it might be useful to perform some preprocessing on input data. Thus, TFLearn has a set of preprocessing functions to make data manipulation more convenient (such as sequence padding, categorical labels, shuffling at unison, image processing, etc...).

- For more details, see: [data_utils](http://tflearn.org/data_utils).


# Getting Further

There are a lot of examples along with numerous neural network
implementations available for you to practice TFLearn more in depth:

- See: [Examples](http://tflearn.org/examples).
