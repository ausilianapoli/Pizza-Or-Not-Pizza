# -*- coding: utf-8 -*-
"""
@author: ausilianapoli
"""

import pickle

class Inference:
    
    def __init__(self, img):
        self.img = img
        
    def indexToNameClass(self, tag):
        if(tag == 0):
            return "other"
        else:
            return "pizza"
        
    def knnInference(self, k):
        filename = "./pkl/{}nn_training.pkl".format(k)
        model = pickle.load(open(filename, "rb"))
        tag = model.predict(self.img)
        name = self.indexToNameClass(tag)
        print("{}nn predicts class {} --> {}".format(k, tag, name))
    
    def NaiveBayesInference(self):
        filename = "./pkl/NaiveBayes_training.pkl"
        model = pickle.load(open(filename, "rb"))
        tag = model.predict(self.img)
        name = self.indexToNameClass(tag)
        print("Naive Bayes predicts class {} --> {}".format(tag, name))
    
    def LogisticRegressionInference(self):
        filename = "./pkl/LogisticRegression_training.pkl"
        model = pickle.load(open(filename, "rb"))
        filename = "./pkl/LogisticRegression_PCA.pkl"
        pca = pickle.load(open(filename, "rb"))
        img_pca = pca.transform(self.img)
        tag = model.predict(img_pca)
        name = self.indexToNameClass(tag)
        print("Logistic Regression predicts class {} --> {}".format(tag, name))
        
