import torch
import torch.utils.data
import numpy as np
import csv 
import random
import scvi
import scanpy as sc
import json
from collections import Counter
import json
import os

def convert_num(labels):
    if os.path.exists('label2id.json'):
        with open('label2id.json', 'r') as j:
            label2id = json.load(j)
    else:
        label2id = {}
        ctr = Counter(labels)
        for label in ctr:
            label2id[label] = len(label2id)
        with open('label2id.json', 'w') as j:
            json.dump(label2id, j)
    label_id_lt = []
    for label in labels:
        label_id = label2id[label]
        label_id_lt.append(label_id)
    return np.array(label_id_lt)

class RepDataset(torch.utils.data.Dataset):
    def __init__(self, rep_path):
        matrix = []
        labels = []

        with open(rep_path, 'r') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                rep = row[:-1]
                rep = [float(item) for item in rep]
                matrix.append(rep)
                labels.append(row[-1])
                line_count += 1

        self.mat = np.array(matrix)
        self.labels = convert_num(labels)
        
        print(f'rep num: {len(self.labels)}')

    def get_weight(self):
        label_num = {}
        for label in self.labels:
            if label in label_num:
                label_num[label] += 1
            else:
                label_num[label] = 1
        weight = []
        for i in range(len(label_num)):
            weight.append(1.0/label_num[i])
        weight = np.array(weight)
        weight = weight/ np.sum(weight)
        return weight

    def get_rep_dim(self):
        return self.mat.shape[1]
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        view = torch.Tensor(self.mat[idx])
        label = torch.LongTensor([int(self.labels[idx])])
        return (view, label)

