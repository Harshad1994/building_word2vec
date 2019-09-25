# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 14:13:40 2019

@author: USER
"""


import numpy as np
import pandas as pd

import pickle

from gensim.models import word2vec
import nltk

from nltk.corpus import stopwords
import re

train_data = pd.read_csv('labeledTrainData.tsv',delimiter = '\t', quoting = 3)
unlabeled_data = pd.read_csv('unlabeledTrainData.tsv',delimiter = '\t', quoting = 3)
test_data = pd.read_csv('testData.tsv',delimiter = '\t', quoting = 3)

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

def Sentence_to_words(sentence, remove_stopwords = False):
    sentence = re.sub('[^a-zA-Z]'," ",sentence)
    tokens = sentence.lower().split()
    if remove_stopwords:
        tokens = [token for token in tokens if token not in set(stopwords.words('English'))]
    return tokens
        

def Review_to_sentences(review,tokenizer,remove_stopwords):
    
    sentences = []
    raw_sentences = tokenizer.tokenize(review.strip())
    
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            words = Sentence_to_words(raw_sentence, remove_stopwords)
            sentences.append(words)
        
    return sentences
    

count = 0
sentences = []
for review in train_data['review']:
    sentences += Review_to_sentences(review, tokenizer,False)
    count +=1
    if count%1000 == 0:
        print (count)
        
for review in unlabeled_data['review']:
    sentences += Review_to_sentences(review, tokenizer,False)
    count +=1
    if count%1000 == 0:
        print (count)




#Creating the model
import logging
logging.basicConfig(format = '%(asctime)s : %(levelname)s : %(message)s', level = logging.INFO)


# Selecting the parameters for model

num_of_features = 300
down_sampling = 0.001
min_word_count = 40
processors = 4
context = 10


# Initialize and train the model
from gensim.models import word2vec
model = word2vec.Word2Vec(sentences, workers = processors, size = num_of_features, min_count= min_word_count , window = context, sample = down_sampling)


model.init_sims(replace = True)
model.save('newly_tried_model')


model.most_similar('queen')




# buildign sentiment analyzer using word2vec

import pickle
corpus = []
with open('corpus.pkl', 'rb') as cf:
    corpus = pickle.load(cf)
    
test_corpus = []
with open('corpus_test.pkl', 'rb') as ctf:
    test_corpus = pickle.load(ctf)    



# Building a average vector for each review.
    

def averaged_review_vector(review):
    
    n_words = 0
    
    feature_vector = np.zeros((300,) ,dtype = 'float32')
    
    index2word_Set = model.wv.index2word
    
    words = review.split()
    
    for word in words:
        if word in index2word_Set:
            n_words +=1
            feature_vector = np.add(feature_vector, model[word])
    
    return np.divide(feature_vector, n_words)
    
def create_feature_matrix(reviews):
    
    feature_matrix = np.zeros((len(reviews), 300) ,dtype = 'float32')
    
    counter = 0
    
    for review in reviews:
        feature_matrix[counter] = averaged_review_vector(review)
        counter +=1
        
        if counter%1000 == 0:
            print ("Review %d of %d" % (counter, len(reviews)))
           
    return feature_matrix
    
train_feature_matrix = create_feature_matrix(corpus)

with open('train_matrix_file', 'wb') as f:
    np.save(f,train_feature_matrix, allow_pickle = True)


test_feature_matrix = create_feature_matrix(test_corpus)
with open('test_matrix_file', 'wb') as f:
    np.save(f,test_feature_matrix, allow_pickle = True)
    
    
# Building a random forest model
    
    
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 100, criterion ='entropy')

classifier.fit(train_feature_matrix, train_data['sentiment'])


y_pred = classifier.predict(test_feature_matrix)


output = pd.DataFrame(data ={'id' : test_data['id'], 'sentiment' : y_pred}, )


output.to_csv('retry.csv', index = False, quoting = 3)

    



    
    







