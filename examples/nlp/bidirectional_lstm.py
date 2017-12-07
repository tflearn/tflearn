# -*- coding: utf-8 -*-
"""
Simple example using LSTM recurrent neural network to classify IMDB
sentiment dataset.

References:
    - Long Short Term Memory, Sepp Hochreiter & Jurgen Schmidhuber, Neural
    Computation 9(8): 1735-1780, 1997.
    - Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng,
    and Christopher Potts. (2011). Learning Word Vectors for Sentiment
    Analysis. The 49th Annual Meeting of the Association for Computational
    Linguistics (ACL 2011).

Links:
    - http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf
    - http://ai.stanford.edu/~amaas/data/sentiment/

"""

from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.data_utils import to_categorical, pad_sequences
from tflearn.datasets import imdb
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.embedding_ops import embedding
from tflearn.layers.recurrent import bidirectional_rnn, BasicLSTMCell
from tflearn.layers.estimator import regression

# IMDB Dataset loading
train, test, _ = imdb.load_data(path='imdb.pkl', n_words=10000,
                                valid_portion=0.1)
trainX, trainY = train
testX, testY = test

# Data preprocessing
# Sequence padding
trainX = pad_sequences(trainX, maxlen=200, value=0.)
testX = pad_sequences(testX, maxlen=200, value=0.)
# Converting labels to binary vectors
trainY = to_categorical(trainY)
testY = to_categorical(testY)

# Network building
net = input_data(shape=[None, 200])
net = embedding(net, input_dim=20000, output_dim=128)
net = bidirectional_rnn(net, BasicLSTMCell(128), BasicLSTMCell(128))
net = dropout(net, 0.5)
net = fully_connected(net, 2, activation='softmax')
net = regression(net, optimizer='adam', loss='categorical_crossentropy')

# Training
model = tflearn.DNN(net, clip_gradients=0., tensorboard_verbose=2)
model.fit(trainX, trainY, validation_set=0.1, show_metric=True, batch_size=64)
