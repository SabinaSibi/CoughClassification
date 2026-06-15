#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 12:00:54 2023

@author: sabina
"""


#importing python libaries

import pandas as pd 
import numpy as np 
import scipy as sp
import tensorflow as tf
import matplotlib.pyplot as plt
import os 
import keras
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


spectros = '/home/export/externos/sabina/Downloads/VirCorona/Spectros'

label_map = {'neg': 0, 'pos': 1}


 # load the spectrograms into memory
X = []
Y = []

for fina in os.listdir(spectros):
     if fina.endswith('.png'): # load the spectrograms' file
         spect_path = os.path.join(spectros, fina)
         spect = plt.imread(spect_path)
         
         # add the spectrogram to X
         X.append(spect)
         
         # extract the label from the filename
         label = (fina.split('-')[0])
         label = label_map[label]
         
         # add the label to Y
         Y.append(label)

 # convert Y to a numpy array
Y = np.array(Y)

 # one-hot encode the labels
Y = to_categorical(Y)

 # convert X and Y to numpy arrays
X = np.array(X)
Y = np.array(Y)


 #spliting data into train and test data

train_data, test_data, train_labels, test_labels = train_test_split(
             X, Y, test_size=0.9, shuffle=True)


num_classes = 2


 # define the neural network architecture
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=X.shape[1:]))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.7))
model.add(Dense(num_classes, activation='sigmoid'))

 # Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

 # Train the model
treno = model.fit(train_data, train_labels, batch_size=16, epochs=50)



# plot loss 
plt.plot(treno.history['loss'], color = 'red')
plt.xlabel('epochs')
plt.ylabel('percentage/100')
plt.legend(['loss'])
plt.show()
