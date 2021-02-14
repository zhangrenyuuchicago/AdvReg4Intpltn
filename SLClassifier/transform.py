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

def transform_rep(train_path, test_path):
    train_matrix = []
    raw_train_labels = []
    with open(train_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                pass
            else:
                rep = row[:-1]
                rep = [float(item) for item in rep]
                train_matrix.append(rep)
                raw_train_labels.append(row[-1])
            line_count += 1

    train_mat = np.array(train_matrix)
    train_labels, train_label2id =  convert_num(raw_train_labels)
    
    test_matrix = []
    raw_test_labels = []
    with open(test_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                pass
            else:
                rep = row[:-1]
                rep = [float(item) for item in rep]
                test_matrix.append(rep)
                raw_test_labels.append(row[-1])
            line_count += 1

    test_mat = np.array(test_matrix)
    test_labels, test_label2id =  convert_num(raw_test_labels, train_label2id)
    
    pca = sklearn.decomposition.PCA(train_mat.shape[1], whiten=True)
    pca.fit(train_mat)
    train_mat = pca.transform(train_mat)
    test_mat = pca.transform(test_mat)
    
    train_path = os.path.basename(train_path)
    test_path = os.path.basename(test_path)
    rep_len = train_mat.shape[1]
    
    with open(train_path, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        header = [i for i in range(1, rep_len+1)] + ['label']
        writer.writerow(header)
        for i in range(len(train_labels)):
            line = list(train_mat[i]) + [raw_train_labels[i]]
            writer.writerow(line)

    with open(test_path, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        header = [i for i in range(1, rep_len+1)] + ['label']
        writer.writerow(header)
        for i in range(len(test_labels)):
            line = list(test_mat[i]) + [raw_test_labels[i]]
            writer.writerow(line)

uid2paths = {}

for path in glob.glob('../rep*.csv'):
    #print(f'Compute {path}')
    basename = os.path.basename(path)
    basename = basename[:-4]
    array = basename.split('_')
    if array[1] == 'ae' :
        uid = array[16]
    elif array[1] == 'vae':
        uid = array[18]
    elif array[1] == 'acae':
        uid = array[20]
    else:
        print('error')
    
    if array[2] == 'train':
        if uid in uid2paths:
            uid2paths[uid][0] = path
        else:
            uid2paths[uid] = [path, None]
    elif array[2] == 'test':
        if uid in uid2paths:
            uid2paths[uid][1] = path
        else:
            uid2paths[uid] = [None, path]
 
for uid in uid2paths:
    train_path, test_path = uid2paths[uid][0], uid2paths[uid][1]
    transform_rep(train_path, test_path)
    print(f'finish: {uid}')    

