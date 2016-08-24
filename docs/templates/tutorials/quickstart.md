# TFLearn - Quick Start

In this tutorial, you will learn to use TFLearn and TensorFlow to estimate Titanic passengers chance of surviving the sinking, using their personal information (such as gender, age, etc...). To tackle this classic machine learning task, we are going to build a deep neural network classifier.

## Prerequisite
Make sure that you have tensorflow and tflearn installed. If you don't, please follow these [instructions](http://tflearn.org/installation).

# Overview
![Titanic](http://www.maritime-reproductions.com/images/1000_titanic_sinking_12x8.jpg)
## Introduction
On April 15, 1912, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, children, and the upper-class. In this tutorial, we carry an analysis to find out who these people are.

## Dataset
Let's take a look at the dataset (TFLearn will automatically download it for you). For each passenger, the following information are provided:
```
VARIABLE DESCRIPTIONS:
survived        Survived
                (0 = No; 1 = Yes)
pclass          Passenger Class
                (1 = 1st; 2 = 2nd; 3 = 3rd)
name            Name
sex             Sex
age             Age
sibsp           Number of Siblings/Spouses Aboard
parch           Number of Parents/Children Aboard
ticket          Ticket Number
fare            Passenger Fare
```

Here are some samples extracted from the dataset:

| survived | pclass | name | sex | age | sibsp | parch | ticket | fare |
| -------- | ------ | ---- | --- | --- | ----- | ----- | ------ | ---- |
|1|1|Aubart, Mme. Leontine Pauline|female|24|0|0|PC 17477|69.3000|
|0|2|Bowenur, Mr. Solomon|male|42|0|0|211535|13.0000|
|1|3|Baclini, Miss. Marie Catherine|female|5|2|1|2666|19.2583|
|0|3|Youseff, Mr. Gerious|male|45.5|0|0|2628|7.2250|

There are 2 classes in our task 'not survived' (class 0) and 'survived' (class 1), and the passengers data have 8 features.

# Build the Classifier
## Loading Data
The Dataset is stored in a csv file, so we can use TFLearn `load_csv()` function to load the data from file into a python `list`. We specify 'target_column' argument to indicate that our labels (survived or not) are located in the first column (id: 0). The function will return a tuple: (data, labels).
```python
import numpy as np
import tflearn

# Download the Titanic dataset
from tflearn.datasets import titanic
titanic.download_dataset('titanic_dataset.csv')

# Load CSV file, indicate that the first column represents labels
from tflearn.data_utils import load_csv
data, labels = load_csv('titanic_dataset.csv', target_column=0,
                        categorical_labels=True, n_classes=2)
```

## Preprocessing Data
Data are given 'as it' and need some preprocessing to be ready to be used in our deep neural network classifier.

First, we will discard the fields that are not likely to help in our analysis. For example, we make the assumption that 'name' field will not be very useful in our task, because we estimate that a passenger name and his chance of surviving are not correlated. With such thinking, we discard 'name' and 'ticket' fields.

Then, we need to convert all our data to numerical values, because a neural network model can only perform operations over numbers. However, our dataset contains some non numerical values, such as 'name' or 'sex'. Because 'name' is discarded, we just need to handle 'sex' field. In this simple case, we will just assign '0' to males and '1' to females.

Here is the preprocessing function:
```python
# Preprocessing function
def preprocess(data, columns_to_ignore):
    # Sort by descending id and delete columns
    for id in sorted(columns_to_ignore, reverse=True):
        [r.pop(id) for r in data]
    for i in range(len(data)):
      # Converting 'sex' field to float (id is 1 after removing labels column)
      data[i][1] = 1. if data[i][1] == 'female' else 0.
    return np.array(data, dtype=np.float32)

# Ignore 'name' and 'ticket' columns (id 1 & 6 of data array)
to_ignore=[1, 6]

# Preprocess data
data = preprocess(data, to_ignore)
```

## Build a Deep Neural Network
We are building a 3-layers neural network using TFLearn. We need to specify the shape of our input data. In our case, each sample has a total of 6 features and we will process samples per batch to save memory, so our data input shape is [None, 6] ('None' stands for an unknown dimension, so we can change the total number of samples that are processed in a batch).
```python
# Build neural network
net = tflearn.input_data(shape=[None, 6])
net = tflearn.fully_connected(net, 32)
net = tflearn.fully_connected(net, 32)
net = tflearn.fully_connected(net, 2, activation='softmax')
net = tflearn.regression(net)
```

## Training
TFLearn provides a model wrapper 'DNN' that can automatically performs a neural network classifier tasks, such as training, prediction, save/restore, etc...
We will run it for 10 epochs (the network will see all data 10 times) with a batch size of 16.
```python
# Define model
model = tflearn.DNN(net)
# Start training (apply gradient descent algorithm)
model.fit(data, labels, n_epoch=10, batch_size=16, show_metric=True)
```

Output:
```
---------------------------------
Run id: MG9PV8
Log directory: /tmp/tflearn_logs/
---------------------------------
Training samples: 1309
Validation samples: 0
--
Training Step: 82  | total loss: 0.64003
| Adam | epoch: 001 | loss: 0.64003 - acc: 0.6620 -- iter: 1309/1309
--
Training Step: 164  | total loss: 0.61915
| Adam | epoch: 002 | loss: 0.61915 - acc: 0.6614 -- iter: 1309/1309
--
Training Step: 246  | total loss: 0.56067
| Adam | epoch: 003 | loss: 0.56067 - acc: 0.7171 -- iter: 1309/1309
--
Training Step: 328  | total loss: 0.51807
| Adam | epoch: 004 | loss: 0.51807 - acc: 0.7799 -- iter: 1309/1309
--
Training Step: 410  | total loss: 0.47475
| Adam | epoch: 005 | loss: 0.47475 - acc: 0.7962 -- iter: 1309/1309
--
Training Step: 492  | total loss: 0.51677
| Adam | epoch: 006 | loss: 0.51677 - acc: 0.7701 -- iter: 1309/1309
--
Training Step: 574  | total loss: 0.48988
| Adam | epoch: 007 | loss: 0.48988 - acc: 0.7891 -- iter: 1309/1309
--
Training Step: 656  | total loss: 0.55073
| Adam | epoch: 008 | loss: 0.55073 - acc: 0.7427 -- iter: 1309/1309
--
Training Step: 738  | total loss: 0.50242
| Adam | epoch: 009 | loss: 0.50242 - acc: 0.7854 -- iter: 1309/1309
--
Training Step: 820  | total loss: 0.41557
| Adam | epoch: 010 | loss: 0.41557 - acc: 0.8110 -- iter: 1309/1309
--
```

Our model finish to train with an overall accuracy around 81%, which means that it can predict the correct outcome (survived or not) for 81% of the total passengers.

## Try the Model
It is time to try out our model. For fun, let's take Titanic movie protagonists (DiCaprio and Winslet) and calculate their chance of surviving (class 1).
```python
# Let's create some data for DiCaprio and Winslet
dicaprio = [3, 'Jack Dawson', 'male', 19, 0, 0, 'N/A', 5.0000]
winslet = [1, 'Rose DeWitt Bukater', 'female', 17, 1, 2, 'N/A', 100.0000]
# Preprocess data
dicaprio, winslet = preprocess([dicaprio, winslet], to_ignore)
# Predict surviving chances (class 1 results)
pred = model.predict([dicaprio, winslet])
print("DiCaprio Surviving Rate:", pred[0][1])
print("Winslet Surviving Rate:", pred[1][1])
```

Output:
```
DiCaprio Surviving Rate: 0.13849584758281708
Winslet Surviving Rate: 0.92201167345047
```

Impressive! Our model accurately predicted the outcome of the movie. Odds were against DiCaprio, but Winslet had a high chance of surviving.

More generally, it can be seen through this study that women and children passengers from first class have the highest chance of surviving, while third class male passengers have the lowest.

# Source Code
```python
from __future__ import print_function

import numpy as np
import tflearn

# Download the Titanic dataset
from tflearn.datasets import titanic
titanic.download_dataset('titanic_dataset.csv')

# Load CSV file, indicate that the first column represents labels
from tflearn.data_utils import load_csv
data, labels = load_csv('titanic_dataset.csv', target_column=0,
                        categorical_labels=True, n_classes=2)


# Preprocessing function
def preprocess(data, columns_to_ignore):
    # Sort by descending id and delete columns
    for id in sorted(columns_to_ignore, reverse=True):
        [r.pop(id) for r in data]
    for i in range(len(data)):
      # Converting 'sex' field to float (id is 1 after removing labels column)
      data[i][1] = 1. if data[i][1] == 'female' else 0.
    return np.array(data, dtype=np.float32)

# Ignore 'name' and 'ticket' columns (id 1 & 6 of data array)
to_ignore=[1, 6]

# Preprocess data
data = preprocess(data, to_ignore)

# Build neural network
net = tflearn.input_data(shape=[None, 6])
net = tflearn.fully_connected(net, 32)
net = tflearn.fully_connected(net, 32)
net = tflearn.fully_connected(net, 2, activation='softmax')
net = tflearn.regression(net)

# Define model
model = tflearn.DNN(net)
# Start training (apply gradient descent algorithm)
model.fit(data, labels, n_epoch=10, batch_size=16, show_metric=True)

# Let's create some data for DiCaprio and Winslet
dicaprio = [3, 'Jack Dawson', 'male', 19, 0, 0, 'N/A', 5.0000]
winslet = [1, 'Rose DeWitt Bukater', 'female', 17, 1, 2, 'N/A', 100.0000]
# Preprocess data
dicaprio, winslet = preprocess([dicaprio, winslet], to_ignore)
# Predict surviving chances (class 1 results)
pred = model.predict([dicaprio, winslet])
print("DiCaprio Surviving Rate:", pred[0][1])
print("Winslet Surviving Rate:", pred[1][1])

```
