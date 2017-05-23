# TFLearn Examples

## Basics
- [Linear Regression](https://github.com/tflearn/tflearn/blob/master/examples/basics/linear_regression.py). Implement a linear regression using TFLearn.
- [Logical Operators](https://github.com/tflearn/tflearn/blob/master/examples/basics/logical.py). Implement logical operators with TFLearn (also includes a usage of 'merge').
- [Weights Persistence](https://github.com/tflearn/tflearn/blob/master/examples/basics/weights_persistence.py). Save and Restore a model.
- [Fine-Tuning](https://github.com/tflearn/tflearn/blob/master/examples/basics/finetuning.py). Fine-Tune a pre-trained model on a new task.
- [Using HDF5](https://github.com/tflearn/tflearn/blob/master/examples/basics/use_hdf5.py). Use HDF5 to handle large datasets.
- [Using DASK](https://github.com/tflearn/tflearn/blob/master/examples/basics/use_dask.py). Use DASK to handle large datasets.

## Extending TensorFlow
- [Layers](https://github.com/tflearn/tflearn/blob/master/examples/extending_tensorflow/layers.py). Use TFLearn layers along with TensorFlow.
- [Trainer](https://github.com/tflearn/tflearn/blob/master/examples/extending_tensorflow/trainer.py). Use TFLearn trainer class to train any TensorFlow graph.
- [Built-in Ops](https://github.com/tflearn/tflearn/blob/master/examples/extending_tensorflow/builtin_ops.py). Use TFLearn built-in operations along with TensorFlow.
- [Summaries](https://github.com/tflearn/tflearn/blob/master/examples/extending_tensorflow/summaries.py). Use TFLearn summarizers along with TensorFlow.
- [Variables](https://github.com/tflearn/tflearn/blob/master/examples/extending_tensorflow/variables.py). Use TFLearn variables along with TensorFlow.

## Computer Vision
### Supervised
- [Multi-layer perceptron](https://github.com/tflearn/tflearn/blob/master/examples/images/dnn.py). A multi-layer perceptron implementation for MNIST classification task.
- [Convolutional Network (MNIST)](https://github.com/tflearn/tflearn/blob/master/examples/images/convnet_mnist.py). A Convolutional neural network implementation for classifying MNIST dataset.
- [Convolutional Network (CIFAR-10)](https://github.com/tflearn/tflearn/blob/master/examples/images/convnet_cifar10.py). A Convolutional neural network implementation for classifying CIFAR-10 dataset.
- [Network in Network](https://github.com/tflearn/tflearn/blob/master/examples/images/network_in_network.py). 'Network in Network' implementation for classifying CIFAR-10 dataset.
- [Alexnet](https://github.com/tflearn/tflearn/blob/master/examples/images/alexnet.py). Apply Alexnet to Oxford Flowers 17 classification task.
- [VGGNet](https://github.com/tflearn/tflearn/blob/master/examples/images/vgg_network.py). Apply VGG Network to Oxford Flowers 17 classification task.
- [VGGNet Finetuning (Fast Training)](https://github.com/tflearn/tflearn/blob/master/examples/images/vgg_network_finetuning.py). Use a pre-trained VGG Network and retrain it on your own data, for fast training.
- [RNN Pixels](https://github.com/tflearn/tflearn/blob/master/examples/images/rnn_pixels.py). Use RNN (over sequence of pixels) to classify images.
- [Highway Network](https://github.com/tflearn/tflearn/blob/master/examples/images/highway_dnn.py). Highway Network implementation for classifying MNIST dataset.
- [Highway Convolutional Network](https://github.com/tflearn/tflearn/blob/master/examples/images/convnet_highway_mnist.py). Highway Convolutional Network implementation for classifying MNIST dataset.
- [Residual Network (MNIST)](https://github.com/tflearn/tflearn/blob/master/examples/images/residual_network_mnist.py). A bottleneck residual network applied to MNIST classification task.
- [Residual Network (CIFAR-10)](https://github.com/tflearn/tflearn/blob/master/examples/images/residual_network_cifar10.py). A residual network applied to CIFAR-10 classification task.
- [ResNeXt (CIFAR-10)](https://github.com/tflearn/tflearn/blob/master/examples/images/resnext_cifar10.py). Aggregated residual transformations network (ResNeXt) applied to CIFAR-10 classification task.
- [Google Inception (v3)](https://github.com/tflearn/tflearn/blob/master/examples/images/googlenet.py). Google's Inception v3 network applied to Oxford Flowers 17 classification task.
### Unsupervised
- [Auto Encoder](https://github.com/tflearn/tflearn/blob/master/examples/images/autoencoder.py). An auto encoder applied to MNIST handwritten digits.
- [Variational Auto Encoder](https://github.com/tflearn/tflearn/blob/master/examples/images/variational_autoencoder.py). A Variational Auto Encoder (VAE) trained to generate digit images.
- [GAN (Generative Adversarial Networks)](https://github.com/tflearn/tflearn/blob/master/examples/images/gan.py). Use generative adversarial networks (GAN) to generate digit images from a noise distribution.
- [DCGAN (Deep Convolutional Generative Adversarial Networks)](https://github.com/tflearn/tflearn/blob/master/examples/images/dcgan.py). Use deep convolutional generative adversarial networks (DCGAN) to generate digit images from a noise distribution.

## Natural Language Processing
- [Recurrent Neural Network (LSTM)](https://github.com/tflearn/tflearn/blob/master/examples/nlp/lstm.py). Apply an LSTM to IMDB sentiment dataset classification task.
- [Bi-Directional RNN (LSTM)](https://github.com/tflearn/tflearn/blob/master/examples/nlp/bidirectional_lstm.py). Apply a bi-directional LSTM to IMDB sentiment dataset classification task.
- [Dynamic RNN (LSTM)](https://github.com/tflearn/tflearn/blob/master/examples/nlp/dynamic_lstm.py). Apply a dynamic LSTM to classify variable length text from IMDB dataset.
- [City Name Generation](https://github.com/tflearn/tflearn/blob/master/examples/nlp/lstm_generator_cityname.py). Generates new US-cities name, using LSTM network.
- [Shakespeare Scripts Generation](https://github.com/tflearn/tflearn/blob/master/examples/nlp/lstm_generator_shakespeare.py). Generates new Shakespeare scripts, using LSTM network.
- [Seq2seq](https://github.com/tflearn/tflearn/blob/master/examples/nlp/seq2seq_example.py). Pedagogical example of seq2seq reccurent network. See [this repo](https://github.com/ichuang/tflearn_seq2seq) for full instructions.
- [CNN Seq](https://github.com/tflearn/tflearn/blob/master/examples/nlp/cnn_sentence_classification.py). Apply a 1-D convolutional network to classify sequence of words from IMDB sentiment dataset.

## Reinforcement Learning
- [Atari Pacman 1-step Q-Learning](https://github.com/tflearn/tflearn/blob/master/examples/reinforcement_learning/atari_1step_qlearning.py). Teach a machine to play Atari games (Pacman by default) using 1-step Q-learning.

## Others
- [Recommender - Wide & Deep Network](https://github.com/tflearn/tflearn/blob/master/examples/others/recommender_wide_and_deep.py). Pedagogical example of wide & deep networks for recommender systems.

## Notebooks
- [Spiral Classification Problem](https://github.com/tflearn/tflearn/blob/master/examples/notebooks/spiral.ipynb). TFLearn implementation of spiral classification problem from Stanford CS231n.
