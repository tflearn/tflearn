'''
Pedagogical example realization of seq2seq recurrent neural networks, using TensorFlow and TFLearn.
More info at https://github.com/ichuang/tflearn_seq2seq
'''

from __future__ import division, print_function

import os
import sys
import tflearn
import argparse
import json

import numpy as np
import tensorflow as tf

from tensorflow.python.ops import seq2seq
from tensorflow.python.ops import rnn_cell

#-----------------------------------------------------------------------------

class SequencePattern(object):

    INPUT_SEQUENCE_LENGTH = 10
    OUTPUT_SEQUENCE_LENGTH = 10
    INPUT_MAX_INT = 9
    OUTPUT_MAX_INT = 9
    PATTERN_NAME = "sorted"

    def __init__(self, name=None, in_seq_len=None, out_seq_len=None):
        if name is not None:
            assert hasattr(self, "%s_sequence" % name)
            self.PATTERN_NAME = name
        if in_seq_len:
            self.INPUT_SEQUENCE_LENGTH = in_seq_len
        if out_seq_len:
            self.OUTPUT_SEQUENCE_LENGTH = out_seq_len

    def generate_output_sequence(self, x):
        '''
        For a given input sequence, generate the output sequence.  x is a 1D numpy array 
        of integers, with length INPUT_SEQUENCE_LENGTH.
        
        Returns a 1D numpy array of length OUTPUT_SEQUENCE_LENGTH
        
        This procedure defines the pattern which the seq2seq RNN will be trained to find.
        '''
        return getattr(self, "%s_sequence" % self.PATTERN_NAME)(x)

    def maxmin_dup_sequence(self, x):
        '''
        Generate sequence with [max, min, rest of original entries]
        '''
        x = np.array(x)
        y = [ x.max(), x.min()] +  list(x[2:])
        return np.array(y)[:self.OUTPUT_SEQUENCE_LENGTH]	# truncate at out seq len

    def sorted_sequence(self, x):
        '''
        Generate sorted version of original sequence
        '''
        return np.array( sorted(x) )[:self.OUTPUT_SEQUENCE_LENGTH]

    def reversed_sequence(self, x):
        '''
        Generate reversed version of original sequence
        '''
        return np.array( x[::-1] )[:self.OUTPUT_SEQUENCE_LENGTH]

#-----------------------------------------------------------------------------

class TFLearnSeq2Seq(object):
    '''
    seq2seq recurrent neural network, implemented using TFLearn.
    '''
    AVAILABLE_MODELS = ["embedding_rnn", "embedding_attention"]
    def __init__(self, sequence_pattern, seq2seq_model=None, verbose=None, name=None, data_dir=None):
        '''
        sequence_pattern_class = a SequencePattern class instance, which defines pattern parameters 
                                 (input, output lengths, name, generating function)
        seq2seq_model = string specifying which seq2seq model to use, e.g. "embedding_rnn"
        '''
        self.sequence_pattern = sequence_pattern
        self.seq2seq_model = seq2seq_model or "embedding_rnn"
        assert self.seq2seq_model in self.AVAILABLE_MODELS
        self.in_seq_len = self.sequence_pattern.INPUT_SEQUENCE_LENGTH
        self.out_seq_len = self.sequence_pattern.OUTPUT_SEQUENCE_LENGTH
        self.in_max_int = self.sequence_pattern.INPUT_MAX_INT
        self.out_max_int = self.sequence_pattern.OUTPUT_MAX_INT
        self.verbose = verbose or 0
        self.n_input_symbols = self.in_max_int + 1
        self.n_output_symbols = self.out_max_int + 2		# extra one for GO symbol
        self.model_instance = None
        self.name = name
        self.data_dir = data_dir

    def generate_trainig_data(self, num_points):
        '''
        Generate training dataset.  Produce random (integer) sequences X, and corresponding
        expected output sequences Y = generate_output_sequence(X).

        Return xy_data, y_data (both of type uint32)

        xy_data = numpy array of shape [num_points, in_seq_len + out_seq_len], with each point being X + Y
        y_data  = numpy array of shape [num_points, out_seq_len]
        '''
        x_data = np.random.randint(0, self.in_max_int, size=(num_points, self.in_seq_len))		# shape [num_points, in_seq_len]
        x_data = x_data.astype(np.uint32)						# ensure integer type

        y_data = [ self.sequence_pattern.generate_output_sequence(x) for x in x_data ]
        y_data = np.array(y_data)

        xy_data = np.append(x_data, y_data, axis=1)		# shape [num_points, 2*seq_len]
        return xy_data, y_data

    def sequence_loss(self, y_pred, y_true):
        '''
        Loss function for the seq2seq RNN.  Reshape predicted and true (label) tensors, generate dummy weights,
        then use seq2seq.sequence_loss to actually compute the loss function.
        '''
        if self.verbose > 2: print ("my_sequence_loss y_pred=%s, y_true=%s" % (y_pred, y_true))
        logits = tf.unstack(y_pred, axis=1)		# list of [-1, num_decoder_synbols] elements
        targets = tf.unstack(y_true, axis=1)		# y_true has shape [-1, self.out_seq_len]; unpack to list of self.out_seq_len [-1] elements
        if self.verbose > 2:
            print ("my_sequence_loss logits=%s" % (logits,))
            print ("my_sequence_loss targets=%s" % (targets,))
        weights = [tf.ones_like(yp, dtype=tf.float32) for yp in targets]
        if self.verbose > 4: print ("my_sequence_loss weights=%s" % (weights,))
        sl = seq2seq.sequence_loss(logits, targets, weights)
        if self.verbose > 2: print ("my_sequence_loss return = %s" % sl)
        return sl

    def accuracy(self, y_pred, y_true, x_in):		# y_pred is [-1, self.out_seq_len, num_decoder_symbols]; y_true is [-1, self.out_seq_len]
        '''
        Compute accuracy of the prediction, based on the true labels.  Use the average number of equal
        values.
        '''
        pred_idx = tf.to_int32(tf.argmax(y_pred, 2))		# [-1, self.out_seq_len]
        if self.verbose > 2: print ("my_accuracy pred_idx = %s" % pred_idx)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(pred_idx, y_true), tf.float32), name='acc')
        return accuracy
    
    def model(self, mode="train", num_layers=1, cell_size=32, cell_type="BasicLSTMCell", embedding_size=20, learning_rate=0.0001,
              tensorboard_verbose=0, checkpoint_path=None):
        '''
        Build tensor specifying graph of operations for the seq2seq neural network model.

        mode = string, either "train" or "predict"
        cell_type = attribute of rnn_cell specifying which RNN cell type to use
        cell_size = size for the hidden layer in the RNN cell
        num_layers = number of RNN cell layers to use

        Return TFLearn model instance.  Use DNN model for this.
        '''
        assert mode in ["train", "predict"]

        checkpoint_path = checkpoint_path or ("%s%ss2s_checkpoint.tfl" % (self.data_dir or "", "/" if self.data_dir else ""))
        GO_VALUE = self.out_max_int + 1		# unique integer value used to trigger decoder outputs in the seq2seq RNN

        network = tflearn.input_data(shape=[None, self.in_seq_len + self.out_seq_len], dtype=tf.int32, name="XY")
        encoder_inputs = tf.slice(network, [0, 0], [-1, self.in_seq_len], name="enc_in")	# get encoder inputs
        encoder_inputs = tf.unstack(encoder_inputs, axis=1)					# transform into list of self.in_seq_len elements, each [-1]

        decoder_inputs = tf.slice(network, [0, self.in_seq_len], [-1, self.out_seq_len], name="dec_in")	# get decoder inputs
        decoder_inputs = tf.unstack(decoder_inputs, axis=1)					# transform into list of self.out_seq_len elements, each [-1]

        go_input = tf.multiply( tf.ones_like(decoder_inputs[0], dtype=tf.int32), GO_VALUE ) # insert "GO" symbol as the first decoder input; drop the last decoder input
        decoder_inputs = [go_input] + decoder_inputs[: self.out_seq_len-1]				# insert GO as first; drop last decoder input

        feed_previous = not (mode=="train")

        if self.verbose > 3:
            print ("feed_previous = %s" % str(feed_previous))
            print ("encoder inputs: %s" % str(encoder_inputs))
            print ("decoder inputs: %s" % str(decoder_inputs))
            print ("len decoder inputs: %s" % len(decoder_inputs))

        self.n_input_symbols = self.in_max_int + 1		# default is integers from 0 to 9 
        self.n_output_symbols = self.out_max_int + 2		# extra "GO" symbol for decoder inputs

        single_cell = getattr(rnn_cell, cell_type)(cell_size, state_is_tuple=True)
        if num_layers==1:
            cell = single_cell
        else:
            cell = rnn_cell.MultiRNNCell([single_cell] * num_layers)

        if self.seq2seq_model=="embedding_rnn":
            model_outputs, states = seq2seq.embedding_rnn_seq2seq(encoder_inputs,	# encoder_inputs: A list of 2D Tensors [batch_size, input_size].
                                                                  decoder_inputs,
                                                                  cell,
                                                                  num_encoder_symbols=self.n_input_symbols,
                                                                  num_decoder_symbols=self.n_output_symbols,
                                                                  embedding_size=embedding_size,
                                                                  feed_previous=feed_previous)
        elif self.seq2seq_model=="embedding_attention":
            model_outputs, states = seq2seq.embedding_attention_seq2seq(encoder_inputs,	# encoder_inputs: A list of 2D Tensors [batch_size, input_size].
                                                                        decoder_inputs,
                                                                        cell,
                                                                        num_encoder_symbols=self.n_input_symbols,
                                                                        num_decoder_symbols=self.n_output_symbols,
                                                                        embedding_size=embedding_size,
                                                                        num_heads=1,
                                                                        initial_state_attention=False,
                                                                        feed_previous=feed_previous)
        else:
            raise Exception('[TFLearnSeq2Seq] Unknown seq2seq model %s' % self.seq2seq_model)
            
        tf.add_to_collection(tf.GraphKeys.LAYER_VARIABLES + '/' + "seq2seq_model", model_outputs)	# for TFLearn to know what to save and restore

        # model_outputs: list of the same length as decoder_inputs of 2D Tensors with shape [batch_size x output_size] containing the generated outputs.
        if self.verbose > 2: print ("model outputs: %s" % model_outputs)
        network = tf.stack(model_outputs, axis=1)		# shape [-1, n_decoder_inputs (= self.out_seq_len), num_decoder_symbols]
        if self.verbose > 2: print ("packed model outputs: %s" % network)
        
        if self.verbose > 3:
            all_vars = tf.get_collection(tf.GraphKeys.VARIABLES)
            print ("all_vars = %s" % all_vars)

        with tf.name_scope("TargetsData"):			# placeholder for target variable (i.e. trainY input)
            targetY = tf.placeholder(shape=[None, self.out_seq_len], dtype=tf.int32, name="Y")

        network = tflearn.regression(network, 
                                     placeholder=targetY,
                                     optimizer='adam',
                                     learning_rate=learning_rate,
                                     loss=self.sequence_loss, 
                                     metric=self.accuracy,
                                     name="Y")

        model = tflearn.DNN(network, tensorboard_verbose=tensorboard_verbose, checkpoint_path=checkpoint_path)
        return model

    def train(self, num_epochs=20, num_points=100000, model=None, model_params=None, weights_input_fn=None, 
              validation_set=0.1, snapshot_step=5000, batch_size=128, weights_output_fn=None):
        '''
        Train model, with specified number of epochs, and dataset size.

        Use specified model, or create one if not provided.  Load initial weights from file weights_input_fn, 
        if provided. validation_set specifies what to use for the validation.

        Returns logits for prediction, as an numpy array of shape [out_seq_len, n_output_symbols].
        '''
        trainXY, trainY = self.generate_trainig_data(num_points)
        print ("[TFLearnSeq2Seq] Training on %d point dataset (pattern '%s'), with %d epochs" % (num_points, 
                                                                                               self.sequence_pattern.PATTERN_NAME,
                                                                                               num_epochs))
        if self.verbose > 1:
            print ("  model parameters: %s" % json.dumps(model_params, indent=4))
        model_params = model_params or {}
        model = model or self.setup_model("train", model_params, weights_input_fn)
        
        model.fit(trainXY, trainY, 
                  n_epoch=num_epochs, 
                  validation_set=validation_set, 
                  batch_size=batch_size,
                  shuffle=True,
                  show_metric=True,
                  snapshot_step=snapshot_step,
                  snapshot_epoch=False, 
                  run_id="TFLearnSeq2Seq"
             )
        print ("Done!")
        if weights_output_fn is not None:
            weights_output_fn = self.canonical_weights_fn(weights_output_fn)
            model.save(weights_output_fn)
            print ("Saved %s" % weights_output_fn)
            self.weights_output_fn = weights_output_fn
        return model

    def canonical_weights_fn(self, iteration_num=0):
        '''
        Construct canonical weights filename, based on model and pattern names.
        '''
        if not type(iteration_num)==int:
            try:
                iteration_num = int(iteration_num)
            except Exception as err:
                return iteration_num
        model_name = self.name or "basic"
        wfn = "ts2s__%s__%s_%s.tfl" % (model_name, self.sequence_pattern.PATTERN_NAME, iteration_num)
        if self.data_dir:
            wfn = "%s/%s" % (self.data_dir, wfn)
        self.weights_filename = wfn
        return wfn

    def setup_model(self, mode, model_params=None, weights_input_fn=None):
        '''
        Setup a model instance, using the specified mode and model parameters.
        Load the weights from the specified file, if it exists.
        If weights_input_fn is an integer, use that the model name, and
        the pattern name, to construct a canonical filename.
        '''
        model_params = model_params or {}
        model = self.model_instance or self.model(mode=mode, **model_params)
        self.model_instance = model
        if weights_input_fn:
            if type(weights_input_fn)==int:
                weights_input_fn = self.canonical_weights_fn(weights_input_fn)
            if os.path.exists(weights_input_fn):
                model.load(weights_input_fn)
                print ("[TFLearnSeq2Seq] model weights loaded from %s" % weights_input_fn)
            else:
                print ("[TFLearnSeq2Seq] MISSING model weights file %s" % weights_input_fn)
        return model

    def predict(self, Xin, model=None, model_params=None, weights_input_fn=None):
        '''
        Make a prediction, using the seq2seq model, for the given input sequence Xin.
        If model is not provided, create one (or use last created instance).

        Return prediction, y

        prediction = array of integers, giving output prediction.  Length = out_seq_len
        y = array of shape [out_seq_len, out_max_int], giving logits for output prediction
        '''
        if not model:
            model = self.model_instance or self.setup_model("predict", model_params, weights_input_fn)

        if self.verbose: print ("Xin = %s" % str(Xin))

        X = np.array(Xin).astype(np.uint32)
        assert len(X)==self.in_seq_len
        if self.verbose:
            print ("X Input shape=%s, data=%s" % (X.shape, X))
            print ("Expected output = %s" % str(self.sequence_pattern.generate_output_sequence(X)))

        Yin = [0]*self.out_seq_len

        XY = np.append(X, np.array(Yin).astype(np.float32))
        XY = XY.reshape([-1, self.in_seq_len + self.out_seq_len])		# batch size 1
        if self.verbose > 1: print ("XY Input shape=%s, data=%s" % (XY.shape, XY))

        res = model.predict(XY)
        res = np.array(res)
        if self.verbose > 1: print ("prediction shape = %s" % str(res.shape))
        y = res.reshape(self.out_seq_len, self.n_output_symbols)
        prediction = np.argmax(y, axis=1)
        if self.verbose:
            print ("Predicted output sequence: %s" % str(prediction))
        return prediction, y

#-----------------------------------------------------------------------------

class VAction(argparse.Action):
    def __call__(self, parser, args, values, option_string=None):
        curval = getattr(args, self.dest, 0) or 0
        values=values.count('v')+1
        setattr(args, self.dest, values + curval)
    
#-----------------------------------------------------------------------------

def CommandLine(args=None, arglist=None):
    '''
    Main command line.  Accepts args, to allow for simple unit testing.
    '''
    help_text = """
Commands:

train - give size of training set to use, as argument
predict - give input sequence as argument (or specify inputs via --from-file <filename>)

"""
    parser = argparse.ArgumentParser(description=help_text, formatter_class=argparse.RawTextHelpFormatter)
    
    parser.add_argument("cmd", help="command")
    parser.add_argument("cmd_input", nargs='*', help="input to command")
    parser.add_argument('-v', "--verbose", nargs=0, help="increase output verbosity (add more -v to increase versbosity)", action=VAction, dest='verbose')
    parser.add_argument("-m", "--model", help="seq2seq model name: either embedding_rnn (default) or embedding_attention", default=None)
    parser.add_argument("-r", "--learning-rate", type=float, help="learning rate (default 0.0001)", default=0.0001)
    parser.add_argument("-e", "--epochs", type=int, help="number of trainig epochs", default=10)
    parser.add_argument("-i", "--input-weights", type=str, help="tflearn file with network weights to load", default=None)
    parser.add_argument("-o", "--output-weights", type=str, help="new tflearn file where network weights are to be saved", default=None)
    parser.add_argument("-p", "--pattern-name", type=str, help="name of pattern to use for sequence", default=None)
    parser.add_argument("-n", "--name", type=str, help="name of model, used when generating default weights filenames", default=None)
    parser.add_argument("--in-len", type=int, help="input sequence length (default 10)", default=None)
    parser.add_argument("--out-len", type=int, help="output sequence length (default 10)", default=None)
    parser.add_argument("--from-file", type=str, help="name of file to take input data sequences from (json format)", default=None)
    parser.add_argument("--iter-num", type=int, help="training iteration number; specify instead of input- or output-weights to use generated filenames", default=None)
    parser.add_argument("--data-dir", help="directory to use for storing checkpoints (also used when generating default weights filenames)", default=None)
    # model parameters
    parser.add_argument("-L", "--num-layers", type=int, help="number of RNN layers to use in the model (default 1)", default=1)
    parser.add_argument("--cell-size", type=int, help="size of RNN cell to use (default 32)", default=32)
    parser.add_argument("--cell-type", type=str, help="type of RNN cell to use (default BasicLSTMCell)", default="BasicLSTMCell")
    parser.add_argument("--embedding-size", type=int, help="size of embedding to use (default 20)", default=20)
    parser.add_argument("--tensorboard-verbose", type=int, help="tensorboard verbosity level (default 0)", default=0)

    if not args:
        args = parser.parse_args(arglist)
    
    if args.iter_num is not None:
        args.input_weights = args.iter_num
        args.output_weights = args.iter_num + 1

    model_params = dict(num_layers=args.num_layers,
                        cell_size=args.cell_size,
                        cell_type=args.cell_type,
                        embedding_size=args.embedding_size,
                        learning_rate=args.learning_rate,
                        tensorboard_verbose=args.tensorboard_verbose,
                    )

    if args.cmd=="train":
        try:
            num_points = int(args.cmd_input[0])
        except:
            raise Exception("Please specify the number of datapoints to use for training, as the first argument")
        sp = SequencePattern(args.pattern_name, in_seq_len=args.in_len, out_seq_len=args.out_len)
        ts2s = TFLearnSeq2Seq(sp, seq2seq_model=args.model, data_dir=args.data_dir, name=args.name, verbose=args.verbose)
        ts2s.train(num_epochs=args.epochs, num_points=num_points, weights_output_fn=args.output_weights, 
                   weights_input_fn=args.input_weights, model_params=model_params)
        return ts2s
        
    elif args.cmd=="predict":
        if args.from_file:
            inputs = json.loads(args.from_file)
        try:
            input_x = map(int, args.cmd_input)
            inputs = [input_x]
        except:
            raise Exception("Please provide a space-delimited input sequence as the argument")

        sp = SequencePattern(args.pattern_name, in_seq_len=args.in_len, out_seq_len=args.out_len)
        ts2s = TFLearnSeq2Seq(sp, seq2seq_model=args.model, data_dir=args.data_dir, name=args.name, verbose=args.verbose)
        results = []
        for x in inputs:
            prediction, y = ts2s.predict(x, weights_input_fn=args.input_weights, model_params=model_params)
            print("==> For input %s, prediction=%s (expected=%s)" % (x, prediction, sp.generate_output_sequence(x)))
            results.append([prediction, y])
        ts2s.prediction_results = results
        return ts2s

    else:
        print("Unknown command %s" % args.cmd)

#-----------------------------------------------------------------------------
# unit tests

def test_sp1():
    '''
    Test two different SequencePattern instances
    '''
    sp = SequencePattern("maxmin_dup")
    y = sp.generate_output_sequence(range(10))
    assert all(y==np.array([9, 0, 2, 3, 4, 5, 6, 7, 8, 9]))    
    sp = SequencePattern("sorted")
    y = sp.generate_output_sequence([5,6,1,2,9])
    assert all(y==np.array([1, 2, 5, 6, 9]))
    sp = SequencePattern("reversed")
    y = sp.generate_output_sequence(range(10))
    assert all(y==np.array([9, 8, 7, 6, 5, 4, 3, 2, 1, 0]))

def test_sp2():
    '''
    Test two SequencePattern instance with lengths different from default
    '''
    sp = SequencePattern("sorted", in_seq_len=20, out_seq_len=5)
    x = np.random.randint(0, 9, 20)
    y = sp.generate_output_sequence(x)
    assert len(y)==5
    y_exp = sorted(x)[:5]
    assert all(y==y_exp)

def test_train1():
    '''
    Test simple training of an embedding_rnn seq2seq model
    '''
    sp = SequencePattern()
    ts2s = TFLearnSeq2Seq(sp)
    ofn = "test_%s" % ts2s.canonical_weights_fn(0)
    print ("using weights filename %s" % ofn)
    if os.path.exists(ofn):
        os.unlink(ofn)
    tf.reset_default_graph()
    ts2s.train(num_epochs=1, num_points=10000, weights_output_fn=ofn)
    assert os.path.exists(ofn)

def test_predict1():
    '''
    Test simple preductions using weights just produced (in test_train1)
    '''
    sp = SequencePattern()
    ts2s = TFLearnSeq2Seq(sp, verbose=1)
    wfn = "test_%s" % ts2s.canonical_weights_fn(0)
    print ("using weights filename %s" % wfn)
    tf.reset_default_graph()
    prediction, y = ts2s.predict(Xin=range(10), weights_input_fn=wfn)
    assert len(prediction==10)

def test_train_predict2():
    '''
    Test that the embedding_attention model works, with saving and loading of weights
    '''
    import tempfile
    sp = SequencePattern()
    tempdir = tempfile.mkdtemp()
    ts2s = TFLearnSeq2Seq(sp, seq2seq_model="embedding_attention", data_dir=tempdir, name="attention")
    tf.reset_default_graph()
    ts2s.train(num_epochs=1, num_points=1000, weights_output_fn=1, weights_input_fn=0)
    assert os.path.exists(ts2s.weights_output_fn)

    tf.reset_default_graph()
    ts2s = TFLearnSeq2Seq(sp, seq2seq_model="embedding_attention", data_dir="DATA", name="attention", verbose=1)
    prediction, y = ts2s.predict(Xin=range(10), weights_input_fn=1)
    assert len(prediction==10)

    os.system("rm -rf %s" % tempdir)

def test_train_predict3():
    '''
    Test that a model trained on sequencees of one length can be used for predictions on other sequence lengths
    '''
    import tempfile
    sp = SequencePattern("sorted", in_seq_len=10, out_seq_len=10)
    tempdir = tempfile.mkdtemp()
    ts2s = TFLearnSeq2Seq(sp, seq2seq_model="embedding_attention", data_dir=tempdir, name="attention")
    tf.reset_default_graph()
    ts2s.train(num_epochs=1, num_points=1000, weights_output_fn=1, weights_input_fn=0)
    assert os.path.exists(ts2s.weights_output_fn)

    tf.reset_default_graph()
    sp = SequencePattern("sorted", in_seq_len=20, out_seq_len=8)
    tf.reset_default_graph()
    ts2s = TFLearnSeq2Seq(sp, seq2seq_model="embedding_attention", data_dir="DATA", name="attention", verbose=1)
    x = np.random.randint(0, 9, 20)
    prediction, y = ts2s.predict(x, weights_input_fn=1)
    assert len(prediction==8)

    os.system("rm -rf %s" % tempdir)

def test_main1():
    '''
    Integration test - training
    '''
    import tempfile
    tempdir = tempfile.mkdtemp()
    arglist = "--data-dir %s -e 2 --iter-num=1 -v -v --tensorboard-verbose=1 train 5000" % tempdir
    arglist = arglist.split(' ')
    tf.reset_default_graph()
    ts2s = CommandLine(arglist=arglist)
    assert os.path.exists(ts2s.weights_output_fn)
    os.system("rm -rf %s" % tempdir)

def test_main2():
    '''
    Integration test - training then prediction
    '''
    import tempfile
    tempdir = tempfile.mkdtemp()
    arglist = "--data-dir %s -e 2 --iter-num=1 -v -v --tensorboard-verbose=1 train 5000" % tempdir
    arglist = arglist.split(' ')
    tf.reset_default_graph()
    ts2s = CommandLine(arglist=arglist)
    wfn = ts2s.weights_output_fn
    assert os.path.exists(wfn)

    arglist = "-i %s predict 1 2 3 4 5 6 7 8 9 0" % wfn
    arglist = arglist.split(' ')
    tf.reset_default_graph()
    ts2s = CommandLine(arglist=arglist)
    assert len(ts2s.prediction_results[0][0])==10

    os.system("rm -rf %s" % tempdir)

def test_main3():
    '''
    Integration test - training then prediction: attention model
    '''
    import tempfile
    wfn = "tmp_weights.tfl"
    if os.path.exists(wfn):
        os.unlink(wfn)
    arglist = "-e 2 -o tmp_weights.tfl -v -v -v -v -m embedding_attention train 5000"
    arglist = arglist.split(' ')
    tf.reset_default_graph()
    ts2s = CommandLine(arglist=arglist)
    assert os.path.exists(wfn)

    arglist = "-i tmp_weights.tfl -v -v -v -v -m embedding_attention predict 1 2 3 4 5 6 7 8 9 0" 
    arglist = arglist.split(' ')
    tf.reset_default_graph()
    ts2s = CommandLine(arglist=arglist)
    assert len(ts2s.prediction_results[0][0])==10

#-----------------------------------------------------------------------------

if __name__=="__main__":
    CommandLine()
