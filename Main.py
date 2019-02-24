# -*- coding: utf-8 -*-
"""
@author: ausilianapoli
"""

from Dataset import Dataset
from Classifier import Classifier
import numpy as np
from sklearn.preprocessing import Normalizer
from sklearn.cluster import MiniBatchKMeans as KMeans
import pickle
import warnings
warnings.filterwarnings("ignore")

#Step 1: load and check the dataset
dataset_dir = "my_dataset"
dataset = Dataset(dataset_dir)
classes = dataset.getClasses()
print(classes)
print(dataset.getLength())
print(dataset.getNumberOfClasses())
print(dataset.getClassLength("pizza"))
print(dataset.getClassLength("other"))
 
#Step 2: split the dataset in training set and test set
training_set, test_set = dataset.splitTrainingTest(0.7)
print(training_set.getClasses())
print(training_set.getLength())
print(training_set.getClassLength("pizza"))
print(training_set.getClassLength("other"))
print(test_set.getClasses())
print(test_set.getLength())
print(test_set.getClassLength("pizza"))
print(test_set.getClassLength("other"))
 
#Step 3: extract features using DAISY and create the vocabulary of visual words using KMeans
training_local_features = training_set.extractFeatures()
kmeans = KMeans(700)
kmeans.fit(training_local_features)
 
 #Step 4: assign each local descriptor to nearest centroid
X_training, Y_training, paths_training = training_set.describeDataset(kmeans)
X_test, Y_test, paths_test = test_set.describe_dataset(kmeans)
for i in range(2):
    dataset.displayImageAndRepresentation(X_training, Y_training, paths_training, classes, i)
 
#Step 5: normalize features
presence = (X_training>0).astype(int)
df = presence.sum(axis = 0)
n = len(X_training)
idf = np.log(float(n)/(1 + df))
X_training_tfidf = X_training * idf
X_test_tfidf = X_test * idf
norm = Normalizer(norm = "l2")
X_training_tfidf_l2 = norm.transform(X_training_tfidf)
X_test_tfidf_l2 = norm.transform(X_test_tfidf)
 
#Step 6: save .pkl file in order to speed up computation times for next steps
with open("PizzaORNOTPizza_bovw.pkl", "wb") as out:
    pickle.dump({
            "X_training": X_training,
            "X_training_tfidf_l2": X_training_tfidf_l2,
            "Y_training": Y_training,
            "paths_training": paths_training,
            "X_test": X_test,
            "X_test_tfidf_l2": X_test_tfidf_l2,
            "Y_test": Y_test,
            "paths_test": paths_test,
            "classes": classes,
            "kmeans": kmeans
            }, out)
     
#Step 7: in order to speed up, it needs to use pre-computed features for next steps
with open("PizzaORNOTPizza_bovw.pkl", "rb") as inp:
    data = pickle.load(inp)
  
X_training = data["X_training"]
X_training_tfidf_l2 = data["X_training_tfidf_l2"]
Y_training = data["Y_training"]
paths_training = data["paths_training"]
X_test = data["X_test"]
X_test_tfidf_l2 = data["X_test_tfidf_l2"]
Y_test = data["Y_test"]
paths_test = data["paths_test"]
classes = data["classes"]
kmeans = data["kmeans"]

#Step 8: create Classifier object and call its various methods
classifier = Classifier()
k = 1
while k <= 5:
    classifier.knn(k, X_training_tfidf_l2, Y_training, X_test_tfidf_l2, Y_test)
    k += 2
classifier.naiveBayes(X_training_tfidf_l2, Y_training, X_test_tfidf_l2, Y_test)
classifier.logisticRegression(X_training_tfidf_l2, Y_training, X_test_tfidf_l2, Y_test)

