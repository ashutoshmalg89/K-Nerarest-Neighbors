import numpy as np
import scipy.misc
import os
import glob
import math
import cvxopt
from sklearn.utils import shuffle
from scipy.linalg import eigh

from numpy import mean,cov,cumsum,dot,linalg,size,flipud
from scipy.sparse.linalg import eigs


corpus = "data"
train_images={}
test_images={}
train_ratio=0.5
train_feature=[]
test_feature=[]


def load_and_split_data(path,i,K):
    dataset = os.listdir(path)
    data = np.random.random([10304, 1])
    target = np.random.random([1])
    for labels in dataset:
        files = os.listdir(path + "/" + labels)
        test_target = []
        for filename in files:
            img = scipy.misc.imread(path + '/' + labels + '/' + filename).astype(np.float32)
            img = img / 255
            img = img.reshape(-1, 1)
            data = np.hstack([data, img])
            target = np.hstack([target, int(labels[1:])])

    data = data[:, 1:].T
    target = target[1:]

    target = target.reshape(400, 1)

    data, target = shuffle(data, target)
    subset = len(data) / K
    start = i * subset
    end = subset + start

    testing_set = data[start:end, :]
    test_class = target[start:end, :]
    myrange = [x for x in range(start, end)]
    training_set = np.delete(data, myrange, axis=0)
    train_class = np.delete(target, myrange, axis=0)

    return training_set, train_class,testing_set, test_class


def perform_pca(train_data, test_data,features):

    eig_val, eig_vec = eigs(cov(train_data.T), features, which='LR')
    new_train = np.dot(eig_vec.T, train_data.T)
    new_test = np.dot(eig_vec.T, test_data.T)

    return new_train.T, new_test.T



def train_svm(train_feature, train_label):
    C = 100.00
    samples_count, feature_count = train_feature.shape
    mera_k = np.zeros((samples_count, samples_count))

    for idx1, val1 in enumerate(train_feature):
        for idx2, val2 in enumerate(train_feature):
            mera_k[idx1, idx2] = np.dot(val1, val2)

    P = cvxopt.matrix(np.outer(train_label, train_label.T) * mera_k)
    q = cvxopt.matrix(np.ones(samples_count) * -1)

    gstnd = cvxopt.matrix(np.diag(np.ones(samples_count) * -1))
    hstnd = cvxopt.matrix(np.zeros(samples_count))
    gSlack = cvxopt.matrix(np.diag(np.ones(samples_count)))
    hSlack = cvxopt.matrix(np.ones(samples_count) * C)
    G = cvxopt.matrix(np.vstack((gstnd, gSlack)))
    h = cvxopt.matrix(np.vstack((hstnd, hSlack)))

    A = cvxopt.matrix(train_label.astype('double'), (1, samples_count))
    b = cvxopt.matrix(0.0)

    soln = cvxopt.solvers.qp(P, q, G, h, A, b)

    alpha = np.ravel(soln['x'])
    suppor_vectors = alpha > 0.0

    alpha_sv = alpha[suppor_vectors]
    sv_train_feature = train_feature[suppor_vectors]
    sv_train_label = train_label[suppor_vectors]

    weight = np.zeros(feature_count)
    for n in range(len(alpha_sv)):

        weight += alpha_sv[n] * sv_train_label[n] * sv_train_feature[n].astype(np.float64)

    biases = 0.0
    for id, val in enumerate(sv_train_label):
        if val > -1:
            biases += (1 - (np.dot(weight,sv_train_feature[id])))
            break
    print biases

    return weight, biases

def modify_train_labels(train_l, i):

    new_train_labels = np.array(train_l)
    new_train_labels[new_train_labels != i + 1] = -1
    new_train_labels[new_train_labels ==i+1]=1
    return new_train_labels

def generate_w_b(train_features, train_labels):
    w_mat=[]
    b_mat=[]
    length = len(np.unique(train_labels[:,0]))

    for i in range(int(length)):
        new_train_labels = modify_train_labels(train_labels,i)
        w, b = train_svm(train_features, new_train_labels)
        w_mat.append(w)
        b_mat.append(b)

    return w_mat,b_mat

def predict(test_features,test_labels, weights, biases):
    result=[]
    correct = 0
    accuracy =0.0
    weights = np.array(weights)
    biases = np.array(biases).reshape(40,1)

    for j in range(len(test_features)):
        temp = []
        for i in range(len(weights)):
            output = np.dot(weights[i],test_features[j])+biases[i]
            temp.append(output)

        max_index = temp.index(max(temp))
        result.append(max_index+1)
    #print result

    for i in range(len(result)):
        if result[i]==test_labels[i][0]:
            correct += 1

    accuracy = (float(correct)/len(result))*100
    return accuracy

def task5():
    acc = 0.0
    average_accuracy = 0.0
    K=5
    for i in range(K):
        train_data, train_label, test_data, test_label = load_and_split_data(corpus,i,K)
        weights, biases = generate_w_b(train_data, train_label)
        acc = predict(test_data, test_label, weights, biases)
        print acc

def task6():
    acc = 0.0
    average_accuracy = 0.0
    K = 5
    for i in range(K):
        train_data, train_label, test_data, test_label = load_and_split_data(corpus, i, K)

        new_train, new_test = perform_pca(train_data, test_data, 40)
        weights, biases = generate_w_b(new_train, train_label)
        acc = predict(new_test, test_label, weights, biases)

        print acc



#print task5()
print task6()