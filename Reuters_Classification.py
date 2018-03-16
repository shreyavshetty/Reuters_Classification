'''
Implementation of a Simple Neural Network Model for Document Classification
Dataset Used: Reuters Neswire Classification Dataset
Keras library provides a set of Data sets to play around. Reuters Neswire Classification Dataset is one among those.
Dataset of 11,228 newswires from Reuters, labeled over 46 topics
In this dataset, Documents have been preprocessed, and each review is encoded as a sequence of word indexes (integers). Words are indexed by overall frequency in the dataset, so that for instance the integer "3" encodes the 3rd most frequent word in the data. This allows for quick filtering operations such as: "only consider the top 10,000 most common words, but eliminate the top 20 most common words". 
'''
#import all the required libraries
import numpy as np
import keras
#import keras's Reuters Neswire Classification Dataset
from keras.datasets import reuters
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.preprocessing.text import Tokenizer
print()
'''
********************************************************************************************
Load the dataset. This has to be divided to train set and test set.
x_train, x_test: list of sequences, which are lists of indexes (integers) based on frequency
y_train, y_test: list of integer labels, here it's classes,i.e,46 topics
max_words =2000 : We are considering only 1000 most frequent words. Any less frequent word will appear as oov_char value in the sequence data.
test_split =0.3 : Fraction of the dataset to be used as test data. Here, 30% of the dataset will be test dataset.
seed: Seed for reproducible data shuffling. link: https://www.tutorialspoint.com/python/number_seed.htm
********************************************************************************************
'''
print()
max_words = 2000
print('**************')
print('Loading data')
print('**************')
print()
(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=max_words,seed=45,                                                test_split=0.3)
'''
********************************************************************************************
Here is a look at the dataset:
The first 10 values in x_train,y_train,x_test,y_test
The length of train and test dataset
********************************************************************************************
'''
print('x_train values',x_train[0:10])
print('x_test values',x_test[0:10])
print('y_train values',y_train[0:10])
print('y_test values',y_test[0:10])
print('length of x_train sequences',len(x_train))
print('length of x_test sequences',len(x_test))
'''
Calculating the number of classes because this is required to categorize to classify.
We are doing num_classes+1 because indexes start from 0
'''
num_classes = np.max(y_train) + 1
print('****************************')
print(num_classes, 'classes')
print('****************************')
'''
********************************************************************************************
Tokenizer API is used here.
Read more on this : https://keras.io/preprocessing/text/
                    http://www.orbifold.net/default/2017/01/10/embedding-and-tokenizer-in-keras/
num_words=max_words: Maximum number of words to work with 
sequences_to_matrix:returns numpy array of shape of the model -> len(x_train)* num_words
Arguments:
sequences: list of sequences to vectorize.
********************************************************************************************
'''
print()
print('****************************')
print('Sequence that has to be vectorized')
tokenizer = Tokenizer(num_words=max_words)
x_train = tokenizer.sequences_to_matrix(x_train)
x_test = tokenizer.sequences_to_matrix(x_test)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
print('****************************')
print()
'''
********************************************************************************************
Now, we have the input values-x values vectorized and represented by matrix
We need to now vectorize y-values, represented by 1-d matrix. 
The values of the matrix are binary values. 
1- if the documnet belongs to a particular class
0- if the documnet does not belong to a particular class
indices of this vecor will represent the class number
********************************************************************************************
'''
print('****************************')
print('Converting class vector to binary class matrix')      
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)
print('****************************')
print()
'''
********************************************************************************************
You can create an instance of Sequential model. 
This instance is called and a set of layers are stacked up.
First layer is a fully-connected layer with 700 hidden units and the model will take the array of input shape is (*,max_words) and output arrays of shape (*, 700)
Read more: https://keras.io/layers/core/#dense
Added Relu as the activation layer
We drop out some of the neurons in the hidden layer,used to prevent over-fitting while training neural nets
dropout=0.2 indicated we are removing 20% of the neurons
Read more on Fit: https://machinelearningmastery.com/overfitting-and-underfitting-with-machine-learning-algorithms/
This hidden layer is connected to another set of hidden layers with hidden units = number of classes
The activation unit accross this layer is softmax. Because the output of the softmax function is equivalent to a categorical probability distribution, it tells you the probability that any of the classes are true.
How to decide the number of hidden layers, unit? : https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw

********************************************************************************************
'''
print('****************************')
print('Building model')
model = Sequential()
model.add(Dense(700, input_shape=(max_words,)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes))
model.add(Activation('softmax'))
'''
In order to compile the model, since it's a categorical classification we make use of categorical crossentropy
Read more:(only upto Cross Entropy Error) https://visualstudiomagazine.com/articles/2014/04/01/neural-network-cross-entropy-error.aspx
Optimization Techniques: https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/
Metrics measured here: accuracy
'''
print('Compiling model')
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
print('Fitting the data to the model')
batch_size = 20
epochs = 7
'''
Fit the data to the model by passing the training data, #batch size , #epoch , verbose, validation split for thr training dataset
To read more on verbose : https://stackoverflow.com/questions/47902295/what-is-the-use-of-verbose-in-keras-while-validating-the-model
'''
history = model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose=1,validation_split=0.1)
print('Evaluating the test data on the model')
score = model.evaluate(x_test, y_test,batch_size=batch_size, verbose=1)
print('Test accuracy:', score[1])
