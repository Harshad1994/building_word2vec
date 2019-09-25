# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 10:56:56 2019

@author: USER
"""

from gensim.models import Word2Vec
import numpy as np
import re
from nltk.corpus import stopwords
import pandas as pd
import pickle

model = Word2Vec.load('300features_40minwords_10context')

train = pd.read_csv('labeledTrainData.tsv',delimiter = '\t', quoting = 3)
test = pd.read_csv('testData.tsv',delimiter = '\t', quoting = 3)

# Vector averaging

def makeFeatureVecs(words, model, num_features):
    
    # add up all feature vecs in a paragraph and average it
    
    featureVec = np.zeros((num_features,), dtype = 'float32')
    
    index2word_set = set(model.wv.index2word)
    
    n_words = 0
    
    for word in words:
        if word in index2word_set:
            n_words +=1
            featureVec = np.add(featureVec,model[word])
            
    featureVec = np.divide(featureVec,n_words)
    return featureVec

def getAvgFeatureVec(reviews, model, num_features):
    
    reviewFeatureVec = np.zeros((len(reviews),num_features), dtype = 'float32')
    counter = 0
    
    
    for review in reviews:
        reviewFeatureVec[counter] = makeFeatureVecs(review,model, num_features)
        
        counter +=1
        
        if counter % 1000 ==0:
            print('review %d of  %d'% (counter, len(reviews)))
        
    return reviewFeatureVec


def review_to_wordlist(review, remove_stopwords = False):
    review = re.sub('[^a-zA-Z]'," ",review)
    tokens = review.lower().split()
    if remove_stopwords:
        tokens = [token for token in tokens if token not in set(stopwords.words('English'))]
    
    return tokens

'''
# Creating Feature Vecs for train_reviews
counter = 0    
clean_train_reviews = []
for review in train['review']:
    clean_train_reviews.append(review_to_wordlist(review, True))
    counter +=1        
    if counter % 500 ==0:
            print('review %d of  %d'% (counter, len(train['review'])))
'''
    
corpus = []
clean_train_reviews = []
with open('corpus.pkl', 'rb') as f:
    corpus = pickle.load(f)

for review in corpus:
    clean_train_reviews.append(review.split())
    
corpus_test = []
clean_test_reviews = []
with open('corpus_test.pkl', 'rb') as f:
    corpus_test = pickle.load(f)
    
for review in corpus_test:
    clean_test_reviews.append(review.split())
    
    
# Get Train and Test Data vectors
trainDataVector = getAvgFeatureVec(clean_train_reviews, model, 300)
testDataVector = getAvgFeatureVec(clean_test_reviews, model, 300)


# Using TrainData Vector to train the Random Forest classifier
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators = 100,criterion = 'entropy',n_jobs =4)
classifier.fit(trainDataVector, train['sentiment'])

y_pred = classifier.predict(testDataVector)

output = pd.DataFrame(data = {'id': test['id'], 'sentiment' : y_pred})  

output.to_csv('my_sub_w2v_rf.csv', index = False, quoting = 3)
        