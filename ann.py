# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 21:19:38 2019

@author: USER
"""

# Ann for solving classification problem making use of 
# averaged feature vectors


import numpy as np
import pandas as pd

train_data = pd.read_csv('labeledTrainData.tsv',delimiter = '\t', quoting = 3)
test_data = pd.read_csv('testData.tsv',delimiter = '\t', quoting = 3)



with open('test_matrix_file', 'rb') as fp:
    X_test  = np.load(fp)

with open('train_matrix_file', 'rb') as f:
    X_train = np.load(f)
    

# Building an ann
import keras
from keras.layers import Dense
from keras.models import Sequential

model = Sequential()

model.add(Dense(units = 150, activation = 'relu', kernel_initializer = 'uniform', input_dim = 300)) 
model.add(Dense(units = 150, activation = 'relu',kernel_initializer = 'uniform'))
model.add(Dense(units = 150, activation = 'relu',kernel_initializer = 'uniform'))
model.add(Dense(units = 1, activation = 'sigmoid'))   

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

model.summary()

model.fit(x = X_train, y= train_data['sentiment'], batch_size = 25000, epochs =150)


y_pred = model.predict(X_test)

y_pred = y_pred > 0.5
y_pred = y_pred*1
y_pred = [item[0] for item in y_pred]
output = pd.DataFrame(data ={'id' : test_data['id'], 'sentiment' : y_pred}, )


output.to_csv('ann_3_layers_150epochs.csv', index = False, quoting = 3)

