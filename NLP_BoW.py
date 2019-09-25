# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 13:04:44 2019

@author: USER
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
dataset = pd.read_csv('labeledTrainData.tsv',delimiter = '\t')
lemmatizer = WordNetLemmatizer()

from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

corpus = []
count = 0


for review in dataset['review']:
    review = re.sub('[^a-zA-Z]'," ", review)
    review = review.lower()
    tokens = review.split()

    tokens = [ps.stem(token) for token in tokens if token not in set(stopwords.words('English'))]
    review = " ".join(tokens) 
    corpus.append(review)
    count +=1
    if count%100 == 0:
        print (count)
        
   
from sklearn.feature_extraction.text import CountVectorizer
cv  = CountVectorizer(max_features = 40000)
X = cv.fit_transform(corpus).toarray()

y = dataset['sentiment'].values

# Dump Corpus into file


with open('corpus.pkl','wb') as f:
    pickle.dump(corpus,f)


corpus = []
with open('corpus.pkl', 'rb') as f:
    corpus = pickle.load(f)



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)


# Using Random Forest for classification problem
from sklearn.ensemble import RandomForestClassifier
classifier= RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

accuracy = (cm[0,0]+cm[1,1])*100/200






# Preparing test Data
test_dataset = pd.read_csv('testData.tsv',delimiter = '\t')

corpus_test = []
count = 0


for review in test_dataset['review']:
    review = re.sub('[^a-zA-Z]'," ", review)
    review = review.lower()
    tokens = review.split()

    tokens = [ps.stem(token) for token in tokens if token not in set(stopwords.words('English'))]
    review = " ".join(tokens) 
    corpus_test.append(review)
    count +=1
    if count%100 == 0:
        print (count) 
        
with open('corpus_test.pkl','wb') as ct:
    pickle.dump(corpus_test,ct)

corpus_test = []    
with open('corpus.pkl', 'rb') as f:
    corpus = pickle.load(f)
    
X_test = cv.transform(corpus).toarray()


y_pred =classifier.predict(X_test)

# Creating the pandas dataframe of the output
        
output =  pd.DataFrame(data = {'id':test_dataset['id'], 'sentiment': y_pred})

#writting the output to the file
output.to_csv('my_sub_RandomForest.csv',index = False, quoting =3)

