# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 21:06:14 2019

@author: giles
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from scipy import stats
from pathlib import Path, PureWindowsPath


def extract_dataset_info(data_path):
    # extract information from train.txt
    f = open(os.path.join(data_path, "train.txt"), "r")
    contents_train = f.readlines()
    label_classes, label_train_list, img_train_list = [], [], []
    for sample in contents_train:
        sample = sample.split()
        label, img_path = sample[0], sample[1]
        if label not in label_classes:
            label_classes.append(label)
        label_train_list.append(sample[0])
        img_train_list.append(os.path.join(data_path, Path(PureWindowsPath(img_path))))
    print('Classes: {}'.format(label_classes))

    # extract information from test.txt
    f = open(os.path.join(data_path, "test.txt"), "r")
    contents_test = f.readlines()
    label_test_list, img_test_list = [], []
    for sample in contents_test:
        sample = sample.split()
        label, img_path = sample[0], sample[1]
        label_test_list.append(label)
        img_test_list.append(os.path.join(data_path, Path(PureWindowsPath(img_path))))  # you can directly use img_path if you run in Windows

    return label_classes, label_train_list, img_train_list, label_test_list, img_test_list

def get_tiny_image(img, output_size):
    # To do
    
    (w, h) = output_size
    feature = np.zeros((h,w))
    
    for i in range(0,h):
        for j in range(0,w):
            size_h = int(img.shape[0]/h)
            size_w = int(img.shape[1]/w)
            s = 0
            for m in range(0,size_h):
                for n in range(0,size_w):
                    s+=img[i*size_h+m][j*size_w+n]
                    
            feature[i][j]=s/(size_h*size_w)
    feature = np.resize(feature,(1,h*w))
    return feature


def predict_knn(feature_train, label_train, feature_test, k):
    # To do
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(feature_train,label_train)
    label_test_pred = neigh.predict(feature_test)
    
    return label_test_pred


def classify_knn_tiny(label_classes, label_train_list, img_train_list, label_test_list, img_test_list):
    # To do
    #get training img feature
    feature_train = get_tiny_image(cv2.imread(img_train_list[0],0),(16,16))
    for i in range(1,len(img_train_list)):
        src = cv2.imread(img_train_list[i],0)
        feature = get_tiny_image(src,(16,16))
        feature_train = np.vstack((feature_train,feature))
    label_train = np.resize(np.asarray(label_train_list),(1500,1))
    #get testing img feature
    feature_test = get_tiny_image(cv2.imread(img_test_list[0],0),(16,16))
    for i in range(1,len(img_test_list)):
        src = cv2.imread(img_test_list[i],0)
        feature = get_tiny_image(src,(16,16))
        feature_test = np.vstack((feature_test,feature))
     
    label_test_pred = predict_knn(feature_train, label_train, feature_test, k=10)
    label_test_pred = np.resize(label_test_pred,(1500,1))
    label_test_list = np.resize(np.asarray(label_test_list),(1500,1))
    accuracy = accuracy_score(label_test_list,label_test_pred)
    confusion = confusion_matrix(label_test_list,label_test_pred)
    
    visualize_confusion_matrix(confusion, accuracy, label_classes)
    return confusion, accuracy


def compute_dsift(img, stride, size):
    # To do
    # img = cv2.imread(img_train_list[0],0)
    sift = cv2.xfeatures2d.SIFT_create()
    step_size = stride
    kp = [cv2.KeyPoint(x, y, step_size) for y in range(int(size/2), img.shape[0], step_size) 
                                    for x in range(int(size/2), img.shape[1], step_size)]
    a,b= sift.compute(img, kp)
    dense_feature = b
    return dense_feature


def build_visual_dictionary(dense_feature_list, dic_size):
    # To do
    kmeans = KMeans(n_clusters=dic_size,n_init=10,max_iter=300).fit(dense_feature_list)
    vocab = kmeans.cluster_centers_
    
    return vocab


def compute_bow(feature, vocab):
    # To do
    #bow_feature size = dic_size
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(vocab)
    matches = neigh.kneighbors(feature,n_neighbors=1, return_distance=False)
    
    bow_feature = np.zeros((1,vocab.shape[0]))
    for i in range(0,matches.shape[0]):
        t = matches[i][0]
        bow_feature[0][t]+=1
      
    bow_feature = bow_feature/matches.shape[0]
    
    return bow_feature


def classify_knn_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list):
    # To do
    #parameter
    stride=10
    size=16
    dic_size=45
    #build dense_feature_list
    dense_feature_list = compute_dsift(cv2.imread(img_train_list[0],0),stride,size)
    for i in range(1,len(img_train_list)):
        src=cv2.imread(img_train_list[i],0)
        dense_feature = compute_dsift(src,stride,size)
        dense_feature_list = np.vstack((dense_feature_list,dense_feature))
        
    vocab = build_visual_dictionary(dense_feature_list, dic_size)
    #get bow_feature of each image
    #train data
    bow_feature_train = compute_bow(compute_dsift(cv2.imread(img_train_list[0],0), stride, size), vocab)
    for i in range(1,len(img_train_list)):
        src = cv2.imread(img_train_list[i],0)
        feature = compute_dsift(src, stride, size)
        bow_feature = compute_bow(feature, vocab)
        bow_feature_train = np.vstack((bow_feature_train,bow_feature))
    #test data
    bow_feature_test = compute_bow(compute_dsift(cv2.imread(img_test_list[0],0), stride, size), vocab)
    for i in range(1,len(img_test_list)):
        src = cv2.imread(img_test_list[i],0)
        feature = compute_dsift(src, stride, size)
        bow_feature = compute_bow(feature, vocab)
        bow_feature_test = np.vstack((bow_feature_test,bow_feature))
    #knn clasify
    label_train = np.resize(np.asarray(label_train_list),(1500,1))
    label_test_pred = predict_knn(bow_feature_train, label_train, bow_feature_test, k=10)
    label_test_pred = np.resize(label_test_pred,(1500,1))
    label_test_list = np.resize(np.asarray(label_test_list),(1500,1))
    accuracy = accuracy_score(label_test_list,label_test_pred)
    confusion = confusion_matrix(label_test_list,label_test_pred)
    
    visualize_confusion_matrix(confusion, accuracy, label_classes)
    return confusion, accuracy


def predict_svm(feature_train, label_train, feature_test, n_classes): 
    # To do
    #label to int
    le = preprocessing.LabelEncoder()
    le.fit(label_train)
    label_train_int = le.transform(label_train)
    label_train_int = np.resize(label_train_int,(1500,1))
    #
    result_prob = np.zeros((1500,15))
    for i in range(0,n_classes):
        #i = 0 #feature_train=bow_feature_train #feature_test=bow_feature_test
        label_train_single = label_train_int
        for j in range(0,label_train_int.shape[0]):
            if label_train_single[j][0] == i:
                label_train_single[j][0]= i
            if label_train_single[j][0] != i:
                label_train_single[j][0]= 16

        svm = SVC(C=5,gamma=15,degree=5,probability=True)#gamma='scale'
        svm.fit(feature_train, label_train_single)
        label_test_pred = svm.predict(feature_test)
        label_test_pred_prob = svm.predict_proba(feature_test)
        result_prob[:,i] = label_test_pred_prob[:,0]
        
        le = preprocessing.LabelEncoder()
        le.fit(label_train)
        label_train_int = le.transform(label_train)
        label_train_int = np.resize(label_train_int,(1500,1))
    
    result_prob=np.argmax(result_prob, axis=1)
    #le.inverse_transform to get label
    label_test_pred = le.inverse_transform(result_prob)
    label_test_pred = np.resize(label_test_pred,(1500,1))
    return label_test_pred


def classify_svm_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list):
    # To do
    stride=6
    size=16
    dic_size=45
    n_classes=15
    #build dense_feature_list
    dense_feature_list = compute_dsift(cv2.imread(img_train_list[0],0),stride,size)
    for i in range(1,len(img_train_list)):
        src=cv2.imread(img_train_list[i],0)
        dense_feature = compute_dsift(src,stride,size)
        dense_feature_list = np.vstack((dense_feature_list,dense_feature))
        
    vocab = build_visual_dictionary(dense_feature_list, dic_size)
    #get bow_feature of each image
    #train data
    bow_feature_train = compute_bow(compute_dsift(cv2.imread(img_train_list[0],0), stride, size), vocab)
    for i in range(1,len(img_train_list)):
        src = cv2.imread(img_train_list[i],0)
        feature = compute_dsift(src, stride, size)
        bow_feature = compute_bow(feature, vocab)
        bow_feature_train = np.vstack((bow_feature_train,bow_feature))
    #test data
    bow_feature_test = compute_bow(compute_dsift(cv2.imread(img_test_list[0],0), stride, size), vocab)
    for i in range(1,len(img_test_list)):
        src = cv2.imread(img_test_list[i],0)
        feature = compute_dsift(src, stride, size)
        bow_feature = compute_bow(feature, vocab)
        bow_feature_test = np.vstack((bow_feature_test,bow_feature))
    #svm clasify 
    label_train = np.resize(np.asarray(label_train_list),(1500,1))
    label_test_pred = predict_svm(bow_feature_train, label_train, bow_feature_test, n_classes)
    label_test_list = np.resize(np.asarray(label_test_list),(1500,1))
    accuracy = accuracy_score(label_test_list,label_test_pred)
    confusion = confusion_matrix(label_test_list,label_test_pred)
    visualize_confusion_matrix(confusion, accuracy, label_classes)
    return confusion, accuracy


def visualize_confusion_matrix(confusion, accuracy, label_classes):
    plt.title("accuracy = {:.3f}".format(accuracy))
    plt.imshow(confusion)
    ax, fig = plt.gca(), plt.gcf()
    plt.xticks(np.arange(len(label_classes)), label_classes)
    plt.yticks(np.arange(len(label_classes)), label_classes)
    # set horizontal alignment mode (left, right or center) and rotation mode(anchor or default)
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="center", rotation_mode="default")
    # avoid top and bottom part of heatmap been cut
    ax.set_xticks(np.arange(len(label_classes) + 1) - .5, minor=True)
    ax.set_yticks(np.arange(len(label_classes) + 1) - .5, minor=True)
    ax.tick_params(which="minor", bottom=False, left=False)
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    # %matplotlib qt
    # To do: replace with your dataset path
    #dirname = os.path.dirname(os.path.abspath(__file__))
    #filename = os.path.join(dirname, "./scene_classification_data")
    label_classes, label_train_list, img_train_list, label_test_list, img_test_list = extract_dataset_info("./scene_classification_data")
    
    classify_knn_tiny(label_classes, label_train_list, img_train_list, label_test_list, img_test_list)

    classify_knn_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list)
    
    classify_svm_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list)


