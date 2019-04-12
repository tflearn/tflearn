
""" 5 convolutional .

    
    - [Fruit Dataset](http://www.site.uottawa.ca/~shervin/food/)
"""
import numpy as np         # dealing with arrays
import os                  # dealing with directories
from random import shuffle # mixing up or currently ordered data that might lead our network astray in training.
import glob
import cv2

path = 'path of folder contains fruits folders'
IMG_SIZE = 300 #image size for training
LR = 1e-3 #learning rate

MODEL_NAME = 'Fruits_dectector-{}-{}.model'.format(LR, '5conv-basic')
no_of_fruits=7 # no. of folders contain in path file of fruits
split=0.3 #spliting criteria
no_of_images=100 # no. of image of each fruits

def create_train_data(path):
    training_data = []
    folders=os.listdir(path)[0:no_of_fruits]
    for i in range(len(folders)):
        label = [0 for i in range(no_of_fruits)]
        label[i] = 1
        print(folders[i])
        k=0
        for j in glob.glob(path+"\\"+folders[i]+"\\*.jpg"):            
            if(k==no_of_images):
                break
            k=k+1
            img = cv2.imread(j)
            img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
            training_data.append([np.array(img),np.array(label)])
    np.save('training_{}_{}_{}.npz'.format(no_of_fruits,no_of_images,IMG_SIZE),training_data)
    shuffle(training_data)
    return training_data,folders


import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression


#model building

import tensorflow as tf
tf.reset_default_graph()
convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')

convnet = conv_2d(convnet, 32, 5, activation='relu')

convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')

convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 128, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)


convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, no_of_fruits, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log')
    

#data loading
training_data,labels=create_train_data(path)
a=int(len(training_data)*split)
train = training_data[:-a]
test=training_data[-a:]

X = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,3)
Y = [i[1] for i in train]
Y =np.array(Y)
test_x = np.array([i[0] for i in test]).reshape(-1,IMG_SIZE,IMG_SIZE,3)
test_y = [i[1] for i in test]

model.fit({'input': X}, {'targets': Y}, n_epoch=10, validation_set=({'input': test_x}, {'targets': test_y}), 
    snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

model.save(MODEL_NAME)
