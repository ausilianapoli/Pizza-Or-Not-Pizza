Pizza Or Not Pizza?  
===================
I have coded simple classifier for image showing pizza or other. My little project compares different classifiers on **accuracy** and **elapsed time**.  

Structure  
---------
The repository contains following modules:  
- *Main*: it contains the instructions to run the project;  
- *Dataset*: it creates datasets and image representation using bovw technique;  
- *Classifier*: it contains code for various classification techniques chosen;  
- *Inference*: it contains code for predict class on new image. 

Dataset  
--------
I have used following dataset:  
>{bossard14,  
>  title = {Food-101 -- Mining Discriminative Components with Random Forests},  
>  author = {Bossard, Lukas and Guillaumin, Matthieu and Van Gool, Luc},  
>  booktitle = {European Conference on Computer Vision},  
>  year = {2014}  
>}  

[Web site](https://www.vision.ee.ethz.ch/datasets_extra/food-101/)  
I have built my dataset copying 'pizza' folder and creating 'other' folder with various images from the other folders: each folder has 1000 element.  

Classifier 
-----------
It contains code for follows classification techniques:  
- KNN (k is not pre-fixed);  
- Naive Bayes Classifier:  
- Logistic Regression (it also plots the decision boundary in 3D space).  

All training models are saved in .pkl files.  

Inference
-----------
Its input is the bovw representation of an image and loading the classification models (previously trained) it predicts the class.  

Main  
------
Steps done are:  
1. Load and check the dataset;  
2. Split the dataset in training set and test set;  
3. Extract features using DAISY and create the vocabulary of visual words using KMeans;  
4. Assign each local descriptor to nearest centroid;  
5. Normalize features through tf-idf and l2 norma;  
6. Save .pkl file in order to speed up computation times for next steps;  
7. It needs to use pre-computed features for next steps inn order to make faster the computation;  
8. Create Classifier object and call its various methods;  
9. Inference on new image that is pre-processed by feature detector and descriptor.  

Technical Data
----------------
The dataset contains 1000 images fairly divided into two classes.  
The parameter step of DAISY is setted to 6. KMeans uses 700 centroids.  
Relevant times for extracting local features and describing images with created vocabulary:  
- extract feature: 482.77 seconds;  
- describe training set: 328.79 seconds;  
- describe test set: 133.15 seconds.  

Hardwares used for this project are:  
- Architecture: x86_64  
- CPU: Intel(R) Core(TM) i7-4558U CPU @2.80 GHz  
- RAM size: 8 GB  

Usage
------
It is need to run 'Main.py'.  
If you want to use .pkl files to speed up computation, comments lines from 21 to 74 of 'Main.py'.  

How to improve the project?
----------------------------
Add:  
- Inference module that allows you to know the answer to famous question 'Pizza Or Not Pizza?' **DONE**  
- Other classification techniques;  
- Other feature descriptors;  
- Any your *idea*!  

Enjoy eating pizza :)
=====================
