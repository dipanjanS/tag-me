# -*- coding: utf-8 -*-
"""
Created on Wed Mar 05 14:29:10 2014

@author: Deadman
"""
import numpy as np
from PIL import Image
import os
from sklearn.decomposition import RandomizedPCA
import time
import mahotas as mh
from mahotas.features import surf


################################### NOT USED IN FINAL CODE #################################

STANDARD_SIZE = (240, 240)

def img_to_matrix(filename, verbose=False):
    
    """
    Takes a filename and turns it into a numpy array of RGB pixels.
    """
    
    img = Image.open(filename)
    if verbose==True:
        print "changing size from %s to %s" % (str(img.size), str(STANDARD_SIZE))
    img = img.resize(STANDARD_SIZE)
    img = list(img.getdata())
    img = map(list, img)
    img = np.array(img)
    return img
    

def flatten_image(img):
    
    """
    Takes in an (m, n) numpy array and flattens it 
    into an array of shape (1, m * n).
    """
    
    s = img.shape[0] * img.shape[1]
    img_wide = img.reshape(1, s)
    return img_wide[0]


def get_features_from_images_PCA(img_dir,data_set):
    
    """
    Takes in a directory and gets all the images from
    it and extracts the pixel values, flattens the matrix
    into an array and performs principle component analysis
    to get representative subset of features from the pixel
    values of the image.
    """
    
    print "\nExtracting features from given images..."
    img_names = [f for f in os.listdir(img_dir)]
    images = [img_dir+ f for f in os.listdir(img_dir)]
    #print images
    
    print "\nConverting images to vectors"
    data = []
    for image in images:
#        print image
        img = img_to_matrix(image)
        img = flatten_image(img)
        data.append(img)
    
    print "Converting image data to numpy array"
    
    time.sleep(5)
    data = np.array(data)
    print "Finished Conversion"
    time.sleep(5)
    
    print "\nPerforming PCA to get reqd features"
    features = []
    pca = RandomizedPCA(n_components=14)
    for i in xrange(len(data)/100):
        if features == []:
            split = data[0:100]
            features = pca.fit_transform(split)
        else:
            split = data[100*i:100*(i+1)]
            features = np.concatenate((features,pca.fit_transform(split)),axis=0)
    
    print "Writing feature data to file"
    f = open(data_set+"_extracted_features.txt","w")  
    for i in xrange(len(img_names)):
        s = str(img_names[i])
        for value in features[i]:
            s += " "+str(value)
        s += "\n"
        f.write(s)
    
    f.close()
    print "Write completed"


##########################################################################################


########################## UTILITY METHODS USED IN FINAL CODE ############################

def edginess_sobel(image):
    
    """
    Computes edges of the image using Sobel's algorithm 
    and lighter areas appear more edgier. Computed edge points 
    are squared and added to give one global feature per image.
    """    
    
    edges = mh.sobel(image,just_filter=True)
    edges = edges.ravel()
    return np.sqrt(np.dot(edges,edges))



def get_features_from_images(img_dir,data_set):
    
    """
    Uses sobel's algorithm for edge detection to get a global feature 
    and also uses SURF to extract more relevant features per image and
    writes it to a file.
    """
    
    print "\nExtracting features from given images..."
    img_names = [f for f in os.listdir(img_dir)]
    images = [img_dir+ f for f in os.listdir(img_dir)]
    
    features = []
    print "\nApplying edginess algorithm and SURF to images"
    for im in images:
        image = mh.imread(im)
        image_sobel = mh.imread(im,as_grey=True)
        descriptors = surf.dense(image_sobel,spacing=16)[::32].ravel()
        features.append(np.concatenate((mh.features.haralick(image).mean(0),[edginess_sobel(image_sobel)],descriptors),))
    
    features = np.array(features)
    
    print "Writing feature data to file"
    f = open(data_set+"_extracted_features.txt","w")  
    for i in xrange(len(img_names)):
        s = str(img_names[i])
        for value in features[i]:
            s += " "+str(value)
        s += "\n"
        f.write(s)
    
    f.close()
    print "Write completed"    


##########################################################################################


############################################    TEST AREA    ############################# 
#img_dir = "Train/Images/"
#image = mh.imread(img_dir+"0a137d3e0fbc21e7dee2efad873610f8.jpg")
#a = mh.features.haralick(image).mean(0)
#print a
#
##r,g,b = image.transpose(2,0,1)
##r12 = mh.gaussian_filter(r,12.)
##g12 = mh.gaussian_filter(g,12.)
##b12 = mh.gaussian_filter(b,12.)
##
##im12 = mh.as_rgb(r12,g12,b12)
##h,w = r.shape
##Y,X = np.mgrid[:h,:w]
##
##Y = Y-h/2
##Y = Y/Y.max()
##X = X-w/2
##X = X/X.max()
##W = np.exp(-2.*(X**2+Y**2))
##
###W = W - W.min()
###W = W / W.ptp()
##
##w = W[:,:,None]
##ringed = mh.stretch(image*w + (1-w)*im12)
#
#image2 = mh.imread(img_dir+"0a137d3e0fbc21e7dee2efad873610f8.jpg",as_grey=True)
#edges = mh.sobel(image2,just_filter=True)
#edges = edges.ravel()
#b = np.sqrt(np.dot(edges,edges))
#print b
#
#descriptors = surf.dense(image2,spacing=16)[::49].ravel()
#descriptors = np.array(descriptors)
#print descriptors
#
#f = np.concatenate((a,[b],descriptors))
#print f
#get_features_from_images(img_dir,"tst")


############################################################################################    