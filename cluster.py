import numpy as np
import sklearn
from sklearn.cluster import KMeans
from munkres import Munkres
from collections import Counter
import csv
import glob
import os

def compute_acc(cluster, target_cluster, k):
    """ Compute error between cluster and target cluster
    :param cluster: proposed cluster
    :param target_cluster: target cluster
    :return: error
    """
    n = np.shape(target_cluster)[0]
    M = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            M[i][j] = np.sum(np.logical_and(cluster == i, target_cluster == j))
    m = Munkres()
    indexes = m.compute(-M)
    corresp = []
    for i in range(k):
        corresp.append(indexes[i][1])
    pred_corresp = [corresp[int(predicted)] for predicted in cluster]
    acc = np.sum(pred_corresp == target_cluster) / float(len(target_cluster))
    return acc

def convert_num(labels, label2id=None):
    ctr = Counter(labels)
    if label2id == None:
        label2id = {}
        for label in ctr:
            label2id[label] = len(label2id)
    
    label_id_lt = []
    for label in labels:
        label_id = label2id[label]
        label_id_lt.append(label_id)

    return np.array(label_id_lt), label2id

def compute_acc_from_rep(train_path, test_path):
    train_matrix = []
    train_labels = []
    print('load train rep')
    with open(train_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            rep = row[:-1]
            rep = [float(item) for item in rep]
            train_matrix.append(rep)
            train_labels.append(row[-1])
            line_count += 1
    
    train_mat = np.array(train_matrix)
    train_labels, train_label2id =  convert_num(train_labels)
    
    test_matrix = []
    test_labels = []
    
    print('load test rep')
    with open(test_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            rep = row[:-1]
            rep = [float(item) for item in rep]
            test_matrix.append(rep)
            test_labels.append(row[-1])
            line_count += 1

    test_mat = np.array(test_matrix)
    test_labels, test_label2id =  convert_num(test_labels, train_label2id)

    print('fit pca model')
    pca = sklearn.decomposition.PCA(train_mat.shape[1], whiten=True)
    pca.fit(train_mat)
    train_mat = pca.transform(train_mat)
    test_mat = pca.transform(test_mat)
    
    num_classes = len(train_label2id)
    n_init=100
    print(f'cluster num: {num_classes}')
    kmeans = KMeans(init='random', n_clusters=num_classes, 
            max_iter=1000, n_init=n_init)

    #pred = kmeans.fit_predict(mat)
    #pred = np.argmax(pred, axis=1)
    print('fit kmean model')
    kmeans.fit(train_mat)
    train_pred = kmeans.predict(train_mat)
    test_pred = kmeans.predict(test_mat)

    train_acc = compute_acc(train_labels, train_pred, num_classes)
    test_acc = compute_acc(test_labels, test_pred, num_classes)
    return train_acc, test_acc

train_path = './rep_ae_train.csv'
test_path = './rep_ae_test.csv'
train_acc, test_acc = compute_acc_from_rep(train_path, test_path)
print(f'ae train acc: {train_acc}, test acc: {test_acc}')

train_path = './rep_acae_train.csv'
test_path = './rep_acae_test.csv'
train_acc, test_acc = compute_acc_from_rep(train_path, test_path)
print(f'acae train acc: {train_acc}, test acc: {test_acc}')


