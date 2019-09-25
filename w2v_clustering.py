# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 15:28:33 2019

@author: USER
"""

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
import time
import pickle

model = Word2Vec.load('300features_40minwords_10context')

word_vectors = model.wv.syn0
from sklearn.cluster import KMeans

train = pd.read_csv('labeledTrainData.tsv',delimiter = '\t', quoting = 3)
test = pd.read_csv('testData.tsv',delimiter = '\t', quoting = 3)

# Fixing the number of clusters

start = time.time()
num_clusters = int(word_vectors.shape[0] / 5)

kmeans_clustering = KMeans(n_clusters = num_clusters)
idx = kmeans_clustering.fit_predict(word_vectors)

end = time.time()

elapsed = end-start
word_centroid_map = dict(zip( model.wv.index2word, idx ))


def createBagOfCentroids(word_list,word_centroid_map):
    num_centroids = max(word_centroid_map.values()) + 1
    
    bag_of_centroids = np.zeros(num_centroids, dtype = 'float32')
    
    for word in word_list:
        if word in word_centroid_map:
            index = word_centroid_map[word]
            bag_of_centroids[index] += 1
    return bag_of_centroids


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
    
    
    
# Creating the bag of centroids for Train Reviews
train_centroids = np.zeros((len(clean_train_reviews), num_clusters),dtype = 'float32')

counter = 0
for review in clean_train_reviews:
    train_centroids[counter] = createBagOfCentroids(review, word_centroid_map)
    counter +=1

test_centroids = np.zeros((len(clean_test_reviews), num_clusters),dtype = 'float32')

counter = 0
for review in clean_test_reviews:
    test_centroids[counter] = createBagOfCentroids(review, word_centroid_map)
    counter +=1
    
    
# Using TrainData Vector to train the Random Forest classifier
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators = 100,criterion = 'entropy',n_jobs =4)
classifier.fit(train_centroids, train['sentiment'])

y_pred = classifier.predict(test_centroids)

output = pd.DataFrame(data = {'id': test['id'], 'sentiment' : y_pred})  

output.to_csv('my_sub_w2v_clustering_rf.csv', index = False, quoting = 3)





