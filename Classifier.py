# -*- coding: utf-8 -*-
"""
@author: ausilianapoli
"""

from time import time
import numpy as np
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.naive_bayes import MultinomialNB as NB

class Classifier:
        
    def knn (self, k, X_train, y_train, X_test, y_test):
        t1 = time()
        nnk = KNN(k)
        nnk.fit(X_train, y_train)
        t2 = time()
        elapsed_time = t2-t1
        accuracy = nnk.score(X_test, y_test)
        print("{0}-nn Classifier:\n\taccuracy score:{1:0.2f}\n\telapsed time:{2:0.2f} sec"\
              .format(k, accuracy, elapsed_time))
        
    def naiveBayes (self, X_train, y_train, X_test, y_test):
        t1 = time()
        nb = NB()
        nb.fit(X_train, y_train)
        t2 = time()
        elapsed_time = t2-t1
        accuracy = nb.score(X_test, y_test)
        print("Naive Bayes Classifier:\n\taccuracy score:{0:0.2f}\n\telapsed time:{1:0.2f} sec"\
              .format(accuracy, elapsed_time))
        
    def logisticRegression (self, X_train, y_train, X_test, y_test):
        t1 = time()
        pca = PCA()
        pca.fit(X_train)
        X_train_pca = pca.transform(X_train)
        X_test_pca = pca.transform(X_test)
        lr = LogisticRegression()
        lr.fit(X_train_pca, y_train)
        t2 = time()
        #in order to plot
        theta_0 = lr.intercept_
        theta_1_n = lr.coef_
        xs = X_train_pca[:,0]
        ys = X_train_pca[:,1]
        zs = X_train_pca[:,2]
        theta_1 = theta_1_n[0][0]
        theta_2 = theta_1_n[0][1]
        theta_3 = theta_1_n[0][2]
        fig = plt.figure()
        plt.subplot(111, projection = "3d")
        plt.plot(xs[y_train == 0], ys[y_train == 0], zs[y_train == 0], "or")
        plt.plot(xs[y_train == 1], ys[y_train == 1], zs[y_train == 1], "xb")
        x = np.arange(-0.6, 0.8)
        y = np.arange(-0.6, 0.8)
        x, y = np.meshgrid(x, y)
        z = -(theta_0 + theta_1*x + theta_2*y)/theta_3
        plt.gca().plot_surface(x, y, z, shade = False, color="y")
        plt.title("Decision Boundary on Train Data")
        plt.show()
        elapsed_time = t2-t1
        accuracy = lr.score(X_test_pca, y_test)
        print("Logistic Regression Classifier:\n\taccuracy score:{0:0.2f}\n\telapsed time:{1:0.2f} sec (it doesn't include plot time)"\
              .format(accuracy, elapsed_time))
