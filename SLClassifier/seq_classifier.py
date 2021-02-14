import os
import glob
import torch
from torch import nn as nn
import torch.utils.data
from torch import optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from csv import reader
import numpy as np
from RepLoader import RepDataset
import sys
import scvi
import scanpy as sc
import csv
from sklearn.metrics import accuracy_score

CUDA = True
SEED = 1
BATCH_SIZE = 128
LOG_INTERVAL = 10

EPOCHS = 1000
DROPOUT_RATE=0.4
#INPUT_DIMS=4000
PATIENCE=20
torch.manual_seed(SEED)

def train(epoch, train_loader, classifier, cl_optimizer, weight):
    sum_cl_loss = 0
    count = 0
    classifier.train()
    pred_lt = []
    true_lt = []

    for batch_idx, (x, target) in enumerate(train_loader):
        x = Variable(x).float()
        if CUDA:
            x = x.cuda()
            target = target.cuda().view(-1)

        cl_optimizer.zero_grad()
                
        pred = classifier(x)
        cl_loss = F.cross_entropy(pred, target, weight=weight)
        cl_loss.backward()
        cl_optimizer.step()
        sum_cl_loss += cl_loss.cpu().data.numpy()
        count += 1

        soft_pred = torch.nn.Softmax(dim=1)(pred)
        pred_lt += list(soft_pred.data.cpu().numpy())
        true_lt += list(target.data.cpu().numpy())

    sum_cl_loss /= count
    pred_lt = np.array(pred_lt)
    true_lt = np.array(true_lt)

    pred_lt = np.argmax(pred_lt, axis=1) 
    acc = accuracy_score(pred_lt, true_lt)

    return acc, sum_cl_loss


def test(test_loader, classifier, weight):
    # toggle model to train mode
    classifier.eval()
    sum_ae_loss = 0
    pred_lt = []
    true_lt = []
    for batch_idx, (x, target) in enumerate(test_loader):
        x = Variable(x).float()
        if CUDA:
            x = x.cuda()
            target = target.cuda().view(-1)

        pred = classifier(x)
        cl_loss = F.cross_entropy(pred, target, weight=weight)
        soft_pred = torch.nn.Softmax(dim=1)(pred)
        pred_lt += list(soft_pred.data.cpu().numpy())
        true_lt += list(target.data.cpu().numpy())

    pred_lt = np.array(pred_lt)
    true_lt = np.array(true_lt)

    pred_lt = np.argmax(pred_lt, axis=1) 
    acc = accuracy_score(pred_lt, true_lt)

    return acc 
 
def evaluate_rep(train_rep_file, test_rep_file):
    if CUDA:
        torch.cuda.manual_seed(SEED)
    # DataLoader instances will load tensors directly into GPU memory
    #kwargs = {'num_workers': 2, 'pin_memory': True} if CUDA else {}

    train_dataset = RepDataset(train_rep_file)  
    test_dataset = RepDataset(test_rep_file)
    weight = train_dataset.get_weight()
    weight = torch.Tensor(weight)
    if CUDA:
        weight = weight.cuda()

    train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True)
    test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False)
    ZDIMS = train_dataset.get_rep_dim()

    classifier = nn.Sequential(#nn.LayerNorm(ZDIMS, elementwise_affine=True),
            nn.BatchNorm1d(ZDIMS),
            nn.Linear(ZDIMS, 10)
            )

    if CUDA:
        classifier.cuda()

    cl_optimizer = optim.Adam(list(classifier.parameters()), lr=0.001)

    best_test_acc = 0
    best_train_acc = 0
    best_epoch = 0
    for epoch in range(1, EPOCHS + 1):
        train_acc, train_loss = train(epoch, train_loader, classifier, cl_optimizer, weight)
        test_acc = test(test_loader, classifier, weight)
        print(f'EPOCH: {epoch}, train_acc: {train_acc}, test_acc: {test_acc}')
        if best_test_acc < test_acc:
            best_epoch = epoch
            best_test_acc = test_acc
            best_train_acc = train_acc
        else:
            if epoch - best_epoch > PATIENCE:
                print('patience extrausted')
                break

    checkpoint = {'epoch': EPOCHS,
            'classifier': classifier.state_dict(),
            'best_test_acc': best_test_acc,
            'best_train_acc': best_train_acc
            }
    
    basename = os.path.basename(train_rep_file)
    basename = basename[:-4]
    
    #with open(f'{basename}_checkpoint.pt','wb') as f:
    #    torch.save(checkpoint, f)

    return best_train_acc, best_test_acc

train_acc, test_acc = evaluate_rep('../rep_ae_train.csv', '../rep_ae_test.csv')

print(f'train_acc: {train_acc}, test_acc: {test_acc}')

'''
for path in glob.glob('Rep/rep*.csv'):
    basename = os.path.basename(path)
    basename = basename[:-4]
    array = basename.split('_')
    #print(len(array))
    if array[1] == 'ae':
        fout = open(array[1] + '.csv', 'w')
        fout.write('epoch,zdims,dropout-rate,input-dims,n-hidden,n-layers,uid,train-acc,test-acc\n')
        fout.close()
    
    if array[1] == 'vae':
        fout = open(array[1] + '.csv', 'w')
        fout.write('epoch,zdims,dropout-rate,input-dims,n-hidden,n-layers,beta,uid,train-acc,test-acc\n')
        fout.close()

    if array[1] == 'acae':
        fout = open(array[1] + '.csv', 'w')
        fout.write('epoch,zdims,dropout-rate,input-dims,n-hidden,n-layers,lambda,gamma,uid,train-acc,test-acc\n')
        fout.close()

uid2paths = {}

for path in glob.glob('Rep/rep*.csv'):
    print(f'Compute {path}')
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
    train_acc, test_acc = evaluate_rep(train_path, test_path)
    basename = os.path.basename(train_path)
    basename = basename[:-4]
    array = basename.split('_')
    if array[1] == 'ae':
        fout = open(array[1] + '.csv', 'a')
        fout.write(f'{array[4]},{array[6]},{array[8]},{array[10]},{array[12]},{array[14]},{array[16]},{train_acc},{test_acc}\n')
        fout.close()
    if array[1] == 'vae':
        fout = open(array[1] + '.csv', 'a')
        fout.write(f'{array[4]},{array[6]},{array[8]},{array[10]},{array[12]},{array[14]},{array[16]},{array[18]},{train_acc},{test_acc}\n')
        fout.close()
    if array[1] == 'acae':
        fout = open(array[1] + '.csv', 'a')
        fout.write(f'{array[4]},{array[6]},{array[8]},{array[10]},{array[12]},{array[14]},{array[16]},{array[18]},{array[20]},{train_acc},{test_acc}\n')
        fout.close()

    print(f'... train_acc: {train_acc}; test_acc: {test_acc}')
'''

