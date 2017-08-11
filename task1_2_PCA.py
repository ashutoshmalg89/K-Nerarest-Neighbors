import numpy as np
import os
import scipy.misc
import glob
import matplotlib.pyplot as plt
from numpy import mean,cov,cumsum,dot,linalg,size,flipud
import math
from scipy.linalg import eigh
from scipy.sparse.linalg import eigs
from sklearn.utils import shuffle

corpus='data'

train_label = []
test_label = []
train_feature = []
test_feature = []
# print train_label, test_label

files = os.listdir(corpus)


def k_fold_validation(data_set,K):
    subset = len(data_set)/K
    for i in range(K):
        test_data = data_set[i * subset:][:subset]
        train_data = files[:i * subset] + files[(i+1) * subset:]

        print train_data, test_data

def euclidean_distance(img1, img2):
    distance = 0.0
    #print len(img1)
    difference = img1 -img2
    #for i in range(len(img1)):
    sum = np.sum(difference**2)
    #    distance += math.pow((img1[i]-img2[i]),2)
    distance = np.sqrt(sum)

    return distance



def task1(path):
    K=5
    for i in range(K):

        dataset = os.listdir(path)
        data = np.random.random([10304,1])
        target = np.random.random([1])
        for labels in dataset:
            files = os.listdir(path + "/" + labels)
            test_target=[]
            for filename in files:
                img = scipy.misc.imread(path + '/' + labels + '/' + filename).astype(np.float32)
                img = img / 255
                img = img.reshape(-1, 1)
                data = np.hstack([data, img])
                target = np.hstack([target,int(labels[1:])])


        data=data[:,1:].T
        target=target[1:]

        target = target.reshape(400,1)

        data,target = shuffle(data, target)
        subset = len(data) / K
        start = i * subset
        end = subset + start

        testing_set = data[start:end,:]
        test_class = target[start:end, :]
        myrange = [x for x in  range(start,end)]
        training_set = np.delete(data, myrange, axis=0)
        train_class = np.delete(target, myrange, axis=0)


        print testing_set.shape, test_class.shape, training_set.shape, train_class.shape

        correct_predict = 0
        accuracy = 0.0

        eig_val, eig_vec = eigs(cov(training_set.T),40)
        #plt.plot(eig_val)
        #plt.show()
        new_train = np.dot(eig_vec.T,training_set.T)
        new_test = np.dot(eig_vec.T,testing_set.T)
        print new_train.shape, new_test.shape

        for idx, val in enumerate(testing_set):
            # print idx
            actual_label = int(test_class[idx])
            dist_mat = []
            for idx1, val1 in enumerate(training_set):
                dist = euclidean_distance(val, val1)
                dist_mat.append(dist)
            ind = np.argsort(dist_mat)

            predicted_label = int(train_class[ind[0]])

            if (actual_label == predicted_label):
                correct_predict += 1
        #print "Correct Predictions   " + str(correct_predict)
        accuracy = (correct_predict * 100) / len(testing_set)
        print accuracy

def task2(path):
    K=5

    average_acc = 0.0
    for i in range(K):

        dataset = os.listdir(path)
        data = np.random.random([2576,1])
        target = np.random.random([1])
        for labels in dataset:
            files = os.listdir(path + "/" + labels)
            test_target=[]
            for filename in files:
                img = scipy.misc.imread(path + '/' + labels + '/' + filename).astype(np.float32)
                img = img / 255
                img = scipy.misc.imresize(img, size=(56, 46))
                img = img.reshape(-1, 1)
                data = np.hstack([data, img])
                target = np.hstack([target,int(labels[1:])])


        data=data[:,1:].T
        target=target[1:]

        target = target.reshape(400,1)

        data,target = shuffle(data, target)
        subset = len(data) / K
        start = i * subset
        end = subset + start

        testing_set = data[start:end,:]
        test_class = target[start:end, :]
        myrange = [x for x in  range(start,end)]
        training_set = np.delete(data, myrange, axis=0)
        train_class = np.delete(target, myrange, axis=0)


        print testing_set.shape, test_class.shape, training_set.shape, train_class.shape

        correct_predict = 0
        accuracy = 0.0

        eig_val, eig_vec = eigs(cov(training_set.T),40,which='LM')
        new_train = np.dot(eig_vec.T,training_set.T)
        new_test = np.dot(eig_vec.T,testing_set.T)
        print new_train.shape, new_test.shape



        for idx, val in enumerate(testing_set):
            # print idx
            actual_label = int(test_class[idx])
            dist_mat = []
            for idx1, val1 in enumerate(training_set):
                dist = euclidean_distance(val, val1)
                dist_mat.append(dist)
            ind = np.argsort(dist_mat)

            predicted_label = int(train_class[ind[0]])

            if (actual_label == predicted_label):
                correct_predict += 1

        print "Correct Predictions   " + str(correct_predict)
        accuracy = (correct_predict * 100) / len(testing_set)
        print accuracy







#print task1(corpus)
print task2(corpus)








