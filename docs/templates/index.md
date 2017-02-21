# TFLearn: Deep learning library featuring a higher-level API for TensorFlow.

TFlearn is a modular and transparent deep learning library built on top of Tensorflow.  It was designed to provide a higher-level API to TensorFlow in order to facilitate and speed-up experimentations, while remaining fully transparent and compatible with it.

TFLearn features include:

- Easy-to-use and understand high-level API for implementing deep neural networks, with tutorial and examples.
- Fast prototyping through highly modular built-in neural network layers, regularizers, optimizers, metrics...
- Full transparency over Tensorflow. All functions are built over tensors and can be used independently of TFLearn.
- Powerful helper functions to train any TensorFlow graph, with support of multiple inputs, outputs and optimizers.
- Easy and beautiful graph visualization, with details about weights, gradients, activations and more...
- Effortless device placement for using multiple CPU/GPU.

The high-level API currently supports most of recent deep learning models, such as Convolutions, LSTM, BiRNN, BatchNorm, PReLU, Residual networks, Generative networks... In the future, TFLearn is also intended to stay up-to-date with latest deep learning techniques.

Note: Latest TFLearn (v0.3) is only compatible with TensorFlow v1.0 and over.

# Quick overview

**Code Example**
```python
# Classification
tflearn.init_graph(num_cores=8, gpu_memory_fraction=0.5)

net = tflearn.input_data(shape=[None, 784])
net = tflearn.fully_connected(net, 64)
net = tflearn.dropout(net, 0.5)
net = tflearn.fully_connected(net, 10, activation='softmax')
net = tflearn.regression(net, optimizer='adam', loss='categorical_crossentropy')

model = tflearn.DNN(net)
model.fit(X, Y)
```

```python
# Sequence Generation
net = tflearn.input_data(shape=[None, 100, 5000])
net = tflearn.lstm(net, 64)
net = tflearn.dropout(net, 0.5)
net = tflearn.fully_connected(net, 5000, activation='softmax')
net = tflearn.regression(net, optimizer='adam', loss='categorical_crossentropy')

model = tflearn.SequenceGenerator(net, dictionary=idx, seq_maxlen=100)
model.fit(X, Y)
model.generate(50, temperature=1.0)
```

There are many more examples available [here](examples).

# Where to Start?

To install TFLearn, see: [Installation Guide](installation).

If your version of Tensorflow is under 1.0: [Upgrade Tensorflow](installation/#upgrade-tensorflow).

To get started: [Getting Started with TFLearn](getting_started) and [TFLearn Tutorials](tutorials).

For more examples: [Examples List](examples).

To browse the API, check the [API Documentation](doc_index#API).


# Model Visualization

**Graph**

![Graph Visualization](./img/graph.png)

**Loss & Accuracy (multiple runs)**

![Loss Visualization](./img/loss_acc.png)

**Layers**

![Layers Visualization](./img/layer_visualization.png)

# Sources

GitHub: [https://github.com/tflearn/tflearn](https://github.com/tflearn/tflearn).

# Contributions

This is the first release of TFLearn, if you find any bug, please report it in the GitHub issues section.

Improvements and requests for new features are more than welcome! Do not hesitate to twist and tweak TF Learn, and send pull-requests.

For more info: *[Contribute to TFLearn](contributions)*.

# License

MIT License
