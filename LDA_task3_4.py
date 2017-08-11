import numpy as np
import os
import scipy.misc
import matplotlib.pyplot as plt
from numpy import mean,cov,cumsum,dot,linalg
import math
from scipy.linalg import eigh
from scipy.linalg import pinv
from scipy.sparse.linalg import eigs,eigsh
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



def task3(path):


    dataset = os.listdir(path)

    K=5

    for i in range(K):
        training_set = []
        testing_set =[]
        train_label = []
        test_label = []
        train_img_array = []
        train_mean_dic = {}
        scatter_w = 0.0
        scatter_b = 0.0

        for labels in dataset:
            files = os.listdir(path + "/" + labels)

            subset = len(files) / K
            test_data = files[i * subset:][:subset]

            train_data = files[:i * subset] + files[(i + 1) * subset:]
            class_label = labels[1:]


            for train_img in train_data:
                train_label.append(class_label)
                img = scipy.misc.imread(path + '/' + labels + '/' + train_img).astype(np.float32)
                img = img / 255
                img = img.reshape(-1, 1)
                train_img_array.append(img)
                training_set.append(img)

            for test_img in test_data:
                test_label.append(class_label)
                img = scipy.misc.imread(path + '/' + labels + '/' + test_img).astype(np.float32)
                img = img / 255
                img = img.reshape(-1, 1)
                testing_set.append(img)


            train_img_array=np.array(train_img_array)
            train_img_array = train_img_array.reshape(8,10304)
            train_mean_dic[class_label] = train_img_array.mean(axis=0)
            class_scatter_mat = np.zeros((10304,10304))
            for k,v in enumerate(train_img_array):
                x = v.reshape(10304,1)
                mv =train_img_array.mean(axis=0).reshape(10304,1)
                class_scatter_mat+=(x-mv).dot((x-mv).T)
            scatter_w+= class_scatter_mat
            train_img_array = []
        print scatter_w.shape
        training_set=np.array(training_set).reshape(320,10304)
        testing_set = np.array(testing_set).reshape(80,10304)
        train_label = np.array(train_label).reshape(320,1)
        test_label = np.array(test_label).reshape(80,1)

        training_set, train_label = shuffle(training_set, train_label)
        testing_set, test_label = shuffle(testing_set, test_label)

        print training_set.shape, testing_set.shape, train_label.shape, test_label.shape

        overall_mean = training_set.mean(axis=0)
        overall_mean = overall_mean.reshape(10304,1)

        for i in train_mean_dic:
            mv = train_mean_dic[i].reshape(10304,1)
            scatter_b+= 8 * (mv - overall_mean).dot((mv - overall_mean).T)

        print scatter_b.shape

        eig_vals, eig_vecs = eigsh((np.linalg.inv(scatter_w).dot(scatter_b)),39)

        print eig_vecs.shape

        correct_predict = 0
        accuracy = 0.0
        new_train = np.dot(eig_vecs.T, training_set.T)
        new_test = np.dot(eig_vecs.T, testing_set.T)
        print "New Training and Testing " + str(new_train.shape) + "------" + str(new_test.shape)

        for idx, val in enumerate(new_test.T):
            # print idx
            actual_label = int(test_label[idx][0])
            # print "Actual Label "+str(actual_label)
            dist_mat = []
            for idx1, val1 in enumerate(new_train.T):
                dist = euclidean_distance(val, val1)
                dist_mat.append(dist)
            ind = np.argsort(dist_mat)

            predicted_label = int(train_label[ind[0]][0])
            # print "Predicted Label "+str(predicted_label)

            if (actual_label == predicted_label):
                correct_predict += 1
        # print "Correct Predictions   " + str(correct_predict)
        accuracy = (correct_predict * 100) / len(testing_set)
        print "Accuracy " + str(accuracy)



def task4(path):
    K=5
    train_mean_dic = {}
    scatter_w = 0.0
    scatter_b = 0.0
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

        eig_val, eig_vec = eigsh(cov(training_set.T),40)

        new_train = np.dot(eig_vec.T,training_set.T).T
        new_test = np.dot(eig_vec.T,testing_set.T).T
        print new_train.shape, new_test.shape


        for i, j in enumerate(train_class):
            if j[0] not in train_mean_dic:
                train_mean_dic[j[0]] = new_train[i]
            else:
                train_mean_dic[j[0]] += new_train[i]

        for j, k in enumerate(train_mean_dic):
            train_mean_dic[k] = train_mean_dic[k]/8


        for idx, val in enumerate(train_class):
            x = new_train[idx].reshape(40,1)
            mv = train_mean_dic[int(val)].reshape(40,1)
            diff = x- mv
            scatter_w+= np.dot(diff,diff.T)


        print scatter_w.shape

        overall_mean = new_train.mean(axis=0).reshape(40,1)


        for id, vl in enumerate(train_mean_dic):
            diff = train_mean_dic[int(vl)] - overall_mean
            scatter_b+=np.dot(diff,diff.T)

        print scatter_b.shape


        scatter_w_inv = np.linalg.inv(scatter_w)

        print scatter_w_inv.shape

        eig_val, eig_vec = eigsh(np.dot(scatter_w_inv,scatter_b),39)

        new_train_01 = np.dot(eig_vec.T,new_train.T).T
        new_test_01 = np.dot(eig_vec.T,new_test.T).T

        print new_train_01.shape, new_test_01.shape

        for idx, val in enumerate(new_test_01):
            # print idx
            actual_label = int(test_class[idx][0])
            # print "Actual Label "+str(actual_label)
            dist_mat = []
            for idx1, val1 in enumerate(new_train_01):
                dist = euclidean_distance(val, val1)
                dist_mat.append(dist)
            ind = np.argsort(dist_mat)

            predicted_label = int(train_class[ind[0]][0])
            # print "Predicted Label "+str(predicted_label)

            if (actual_label == predicted_label):
                correct_predict += 1
        # print "Correct Predictions   " + str(correct_predict)
        accuracy = (correct_predict * 100) / len(testing_set)
        print "Accuracy " + str(accuracy)







#print task3(corpus)
print task4(corpus)







