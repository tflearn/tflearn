################################################
# The Street View House Numbers (SVHN) Dataset #
# http://ufldl.stanford.edu/housenumbers       #
# Format 2: Cropped Digits                     #
# Train set: 73257 32x32 RGB Digits            #
# Test set:  26032 32x32 RGB Digits            #
# Extra set: 531131 32x32 RGB Digits           #
################################################
from __future__ import print_function
import numpy as np
import scipy.io
from six.moves import urllib
import os

URL_BASE = 'http://ufldl.stanford.edu/housenumbers/'
TRAIN_FILE = 'train_32x32.mat'
TEST_FILE = 'test_32x32.mat'
EXTRA_FILE = 'extra_32x32.mat'
TRAIN_INSTANCES = 73257
TEST_INSTANCES = 26032
EXTRA_INSTANCES = 531131

def load_data(data_dir="svhn/", one_hot=True):
	train_filepath = maybe_download(TRAIN_FILE,data_dir)
	test_filepath = maybe_download(TEST_FILE,data_dir)
	trainX, trainY = read_data_from_file(train_filepath,TRAIN_INSTANCES) 
	testX, testY = read_data_from_file(test_filepath,TEST_INSTANCES)
	return trainX, trainY, testX, testY

def load_extra_data(data_dir="svhn/", one_hot=True):
	extra_filepath = maybe_download(EXTRA_FILE,data_dir)
	extraX, extraY = read_data_from_file(extra_filepath,EXTRA_INSTANCES) 
	return extraX, extraY

def read_data_from_file(filepath,instances):
	print('Reading SVHN Dataset...')
	mat = scipy.io.loadmat(filepath)
	Y = mat['y'] ##Y.shape = (instances,1) 
	X = mat['X'] #X.shape = (32, 32, 3, instances) -> 32x32 RGB 
	nX = np.zeros(instances*3*32*32).reshape(instances,32,32,3)
	for n in range (instances):
		for rgb in range(3):
			for i in range(32):
				for j in range(32):
					nX[n,i,j,rgb]=X[i,j,rgb,n] #output shape: (Nx32x32x3)
	nY = np.zeros(instances*10).reshape(instances,10)
	for n in range(instances):
		nY[n] = label_to_one_hot_y(Y[n,0],10)
	print('   ...dataset read!')
	return nX, nY

def label_to_one_hot_y(y,classes):
	#original .mat files has the 'y' classes labeled from 1 up to 10
	Y = np.zeros(classes)
	Y[y-1] = 1 #classes labeled from 0 up to 9: one_hot vector y
	return Y

def maybe_download(filename, work_directory):
    """Download the data from Stanford's website, unless it's already here."""
    if not os.path.exists(work_directory):
        os.mkdir(work_directory)
    filepath = os.path.join(work_directory, filename)
    if not os.path.exists(filepath):
        print('Downloading SVHN Dataset...')
        filepath, _ = urllib.request.urlretrieve(URL_BASE + filename,filepath)
        statinfo = os.stat(filepath)
        print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
    return filepath
