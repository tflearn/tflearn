from __future__ import absolute_import, division, print_function

import os, sys, argparse
import urllib

import tflearn
from tflearn.data_utils import *

parser = argparse.ArgumentParser(description=
    'Pass a text file to generate LSTM output')

parser.add_argument('filename')
parser.add_argument('-t','--temp', help=
    'Defaults to displaying multiple temperature outputs which is suggested.' +
    ' If temp is specified, a value of 0.0 to 2.0 is recommended.' +
    ' Temperature is the novelty or' +
    ' riskiness of the generated output.  A value closer to 0 will result' +
    ' in output closer to the input, so higher is riskier.', 
    required=False, nargs=1, type=float)
parser.add_argument('-l','--length', help=
    'Optional length of text sequences to analyze.  Defaults to 25.',
    required=False, default=25, nargs=1, type=int)

args = vars(parser.parse_args())

path = args['filename']
if args['temp'] and args['temp'][0] is not None:
    temp = args['temp'][0]
    print("Temperature set to", temp)
    if temp > 2 or temp < 0:
        print("Temperature out of suggested range.  Suggested temp range is 0.0-2.0") 
    else:
        print("Will display multiple temperature outputs")

if args['length'] is not 25: 
    maxlen = args['length'][0] # default 25 is set in .add_argument above if not set by user
    print("Sequence max length set to ", maxlen)
else:
    maxlen = args['length']
model_name=path.split('.')[0]  # create model name from textfile input

if not os.path.isfile(path):
    print("Couldn't find the text file. Are you sure the you passed is correct?")

X, Y, char_idx = \
    textfile_to_semi_redundant_sequences(path, seq_maxlen=maxlen, redun_step=3)

g = tflearn.input_data([None, maxlen, len(char_idx)])
g = tflearn.lstm(g, 512, return_seq=True)
g = tflearn.dropout(g, 0.5)
g = tflearn.lstm(g, 512, return_seq=True)
g = tflearn.dropout(g, 0.5)
g = tflearn.lstm(g, 512)
g = tflearn.dropout(g, 0.5)
g = tflearn.fully_connected(g, len(char_idx), activation='softmax')
g = tflearn.regression(g, optimizer='adam', loss='categorical_crossentropy',
                       learning_rate=0.001)

m = tflearn.SequenceGenerator(g, dictionary=char_idx,
                              seq_maxlen=maxlen,
                              clip_gradients=5.0,
                              checkpoint_path='model_'+ model_name)

for i in range(50):
    seed = random_sequence_from_textfile(path, maxlen)
    m.fit(X, Y, validation_set=0.1, batch_size=128,
          n_epoch=1, run_id=model_name)
    print("-- TESTING...")
    if args['temp'] is not None:
        temp = args['temp'][0]
        print("-- Test with temperature of %s --" % temp)
        print(m.generate(600, temperature=temp, seq_seed=seed))
    else:
        print("-- Test with temperature of 1.0 --")
        print(m.generate(600, temperature=1.0, seq_seed=seed))
        print("-- Test with temperature of 0.5 --")
        print(m.generate(600, temperature=0.5, seq_seed=seed))
