# -*- coding: utf-8 -*-
"""
Created on Wed Mar 05 00:13:32 2014

@author: Deadman
"""

import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
import img_processing_utils


os.system('cls')

############################ FEATURE VECTORS AND LABEL VECTORS #############################
train_X = []
train_y = []
val_X = []
val_y = []
test_X = []
test_y = []


############################ OPERATIONS ON TRAINING DATA ###################################

print "="*35
print "OPERATING ON TRAINING DATA"
print "="*35


img_dir = "Train/Images/"
img_processing_utils.get_features_from_images(img_dir,"training")

print "\nRetrieving image features from files"
d = dict()
f2 = open("training_extracted_features.txt","r")
for record in f2:
    row = record.split()
    d[row[0]] = row[1:]

f2.close()

print "Combining all features"
f1 = open("Train/feature_vectors.txt","r")
for record in f1:
    row = record.split()
    features = row[1:] + d[row[0]]
    train_X.append(features)
    
f1.close()

img_labels = []
print "\nGetting class labels for training data"
f = open("Train/labels.txt","r+")    
for record in f:
    row = record.split()
    img_labels.append(row[0])
    train_y.append(row[1])
    
f.close()

print "\nPre-processing the training data"
train_X=np.array(train_X).astype(float)
train_X = preprocessing.scale(train_X)

train_y=np.array(train_y) 


print "\nMaking the model please wait..."

rf = RandomForestClassifier(n_estimators=630, criterion='entropy', min_samples_leaf=1,\
                            max_depth=None, min_samples_split=1, random_state=0, \
                            max_features=None)

rf = rf.fit(train_X, train_y)

print "Model created...\n\n"


############################ OPERATIONS ON VALIDATION DATA ###################################

print "="*35
print "OPERATING ON VALIDATION DATA"
print "="*35

img_dir = "Validation/Images/"
img_processing_utils.get_features_from_images(img_dir,"validation")

print "\nRetrieving image features from files"
f2 = open("validation_extracted_features.txt","r")
d = dict()
for record in f2:
    row = record.split()
    d[row[0]] = row[1:]

f2.close()

print "Combining all features"
img_labels = []
f1 = open("Validation/feature_vectors.txt","r")
for record in f1:
    row = record.split()
    img_labels.append(row[0])
    features = row[1:] + d[row[0]]
    val_X.append(features)
    
f1.close()

print "\nPre-processing the validation data"
val_X=np.array(val_X).astype(float)
val_X = preprocessing.scale(val_X)

print "\nPerforming prediction on validation data now..."
val_y = rf.predict(val_X)


print "Prediction completed writing results to file now...\n\n"
f = open("result_validation.txt","w")  
for i in xrange(len(img_labels)):
    s = str(img_labels[i]+" "+val_y[i]+"\n")
    f.write(s)
    
f.close()


############################ OPERATIONS ON TESTING DATA ###################################


print "="*35
print "OPERATING ON TEST DATA"
print "="*35

img_dir = "Test/Images/"
img_processing_utils.get_features_from_images(img_dir,"testing")

print "\nRetrieving image features from files"
f2 = open("testing_extracted_features.txt","r")
d = dict()
for record in f2:
    row = record.split()
    d[row[0]] = row[1:]

f2.close()

print "Combining all features"
img_labels = []
f1 = open("Test/feature_vectors.txt","r")
for record in f1:
    row = record.split()
    img_labels.append(row[0])
    features = row[1:] + d[row[0]]
    test_X.append(features)
    
f1.close()


test_X=np.array(test_X).astype(float)
test_X = preprocessing.scale(test_X)

print "\nPerforming prediction on test data now..."
test_y = rf.predict(test_X)


print "Prediction completed writing results to file now..."
f = open("result_testing.txt","w")  
for i in xrange(len(img_labels)):
    s = str(img_labels[i]+" "+test_y[i]+"\n")
    f.write(s)
    
f.close()

print "All Done :)"    


##########################################################################################

