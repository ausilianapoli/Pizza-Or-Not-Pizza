# -*- coding: utf-8 -*-
"""
@author: ausilianapoli
"""

import os
import glob
from skimage import io as sio
from matplotlib import pyplot as plt
import numpy as np
from copy import copy
from skimage.feature import daisy
from skimage.color import rgb2gray
from time import time

class Dataset:
    def __init__(self, path_to_dataset):
        self.path_to_dataset = path_to_dataset
        classes = os.listdir(path_to_dataset)
        self.paths = dict()
        for cl in classes:
            current_paths = sorted(glob.glob(os.path.join(path_to_dataset, cl,"*.jpg")))
            self.paths[cl] = current_paths
            
    def getImagePath(self, cl, idx):
        return self.paths[cl][idx]
    
    def getClasses(self):
        return sorted(self.paths.keys())
    
    def getNumberOfClasses(self):
        return len(self.paths)
    
    def getClassLength(self, cl):
        return len(self.paths[cl])
    
    def getLength(self):
        return sum([len(x) for x in self.paths.values()])
    
    def showImage(self, class_name, image_number):
        im = sio.imread(self.getImagePath(class_name, image_number))
        plt.figure()
        plt.imshow(im)
        plt.show()
    
    def splitTrainingTest(self, percent_train):
        training_paths = dict()
        test_paths = dict()
        for cl in self.getClasses():
            paths = self.paths[cl]
            shuffled_paths = np.random.permutation(paths)
            split_idx = int(len(shuffled_paths)*percent_train)
            training_paths[cl] = shuffled_paths[0:split_idx]
            test_paths[cl] = shuffled_paths[split_idx::]
        training_dataset = copy(self)
        training_dataset.paths = training_paths
        test_dataset = copy(self)
        test_dataset.paths = test_paths
        return training_dataset, test_dataset
    
    def extractFeatures(self):
        tmp_list = list()
        t1 = time()
        print("Extract features...")
        for cl in self.getClasses():
            n_tot = self.getClassLength(cl)
            n_tmp = 1
            paths = self.paths[cl]
            for impath in paths:
                im = sio.imread(impath, as_gray = True)
                daisy_features = daisy(im, step = 6).reshape((-1, 200))
                tmp_list.append(daisy_features)
                print("[{}/{}]".format(n_tmp, n_tot))
                n_tmp += 1
        concatenated_features = np.vstack(tmp_list)
        t2 = time()
        print("Done! Elapsed time: %0.2f sec" %(t2-t1))
        return concatenated_features
    
    def extractAndDescribe(self, img, kmeans):
        img_grey = rgb2gray(img)
        daisy_features = daisy(img_grey, step = 10).reshape((-1, 200))
        assignments = kmeans.predict(daisy_features)
        bovw_representation, _ = np.histogram(assignments, bins = 500, range = (0, 499))
        return bovw_representation
    
    def describeDataset(self, kmeans):
        Y = list()
        X = list()
        paths = list()
        classes = self.getClasses()
        print("Describe Dataset...")
        total_number_images = self.getLength()
        n_i = 0
        t1 = time()
        for cl in classes:
            for path in self.paths[cl]:
                img = sio.imread(path, as_gray = True)
                feat = self.extractAndDescribe(img, kmeans)
                X.append(feat)
                Y.append(classes.index(cl))
                paths.append(path)
                n_i += 1
                print("Processing Image {}/{}".format(n_i, total_number_images))
        X = np.array(X)
        Y = np.array(Y)
        t2 = time()
        print("Done! Elapsed time {0:0.2f} sec"\
              .format(t2 - t1))
        return X, Y, paths
    
    def displayImageAndRepresentation(self, X, Y, paths, classes, idx):
        path = paths[idx]
        im = sio.imread(path)
        plt.figure(figsize = (12,4))
        plt.suptitle("Class {0} - Image n° {1}"\
                  .format(classes[Y[idx]], idx))
        plt.subplot(121)
        plt.imshow(im)
        plt.subplot(122)
        plt.plot(X[idx])
        plt.show()




