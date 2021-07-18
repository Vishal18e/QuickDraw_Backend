import Configure as Con;
import torch as th;
import torch.nn as nn;
import torch.nn.functional as F;
from torchvision import datasets,transforms;
from torch.utils.data import Dataset,DataLoader,TensorDataset;
from PIL import Image;
import os;
import numpy as np
import matplotlib.pyplot as plt;
import pandas as pd

from Data_loader import getDataLoader;
from Model import network
from tqdm import tqdm


def Training(itr,model,train_loader,optimizer,criterion):
    # start_time = time.time();
    # trn_losses = [];
    # tst_losses = [];
    iterator = tqdm(train_loader)
    # for itr in range(epochs):
    trn_corr = 0;
    # tst_corr = 0;
    for tr, (img_trn, label_trn, img_path) in enumerate(iterator):
        # if tr == 1000:
        #     break;
        # else:
        predictions_trn = model(img_trn.float());
        loss_trn = criterion(predictions_trn, label_trn);
        actual_prediction_trn = th.max(predictions_trn, 1).indices;
        trn_corr = trn_corr + sum(actual_prediction_trn == label_trn);
        Accuracy = 100 * (trn_corr / (Con.batchSize['train'] * (tr + 1)));
        #             if tr%50==0:
        # print(f'itr={itr}; train_batch ={tr}; Accuracy={100 * (trn_corr / (25 * (tr + 1)))}');
        optimizer.zero_grad();
        loss_trn.backward();
        optimizer.step();
        iterator.set_description('Epoch:{} | batch:{} |trn_correct:{} | Loss:{}| Accuracy:{}'.format(itr,tr,trn_corr,loss_trn.item(),Accuracy));
    # trn_losses.append(loss_trn.item());

def Testing(model,test_loader,criterion):
    model.eval();
    tst_corr = 0;
    with th.no_grad():
        for ts, (img_tst, label_tst,path_tst) in enumerate(test_loader):
            predictions_tst = model(img_tst.float());
            loss_tst = criterion(predictions_tst, label_tst);
            actual_prediction_tst = th.max(predictions_tst, 1).indices;
            tst_corr = tst_corr + sum(actual_prediction_tst == label_tst);
            Accuracy_tst=100 * (tst_corr / (Con.batchSize['test'] * (ts + 1)));
            # if ts % 25 == 0:
            print(f'test_batch ={ts};tst_corr={tst_corr} Accuracy={Accuracy_tst}| Loss_tst = {loss_tst}');

def seeds(val):
    th.manual_seed(val);
    np.random.seed(val);


def main():
    model = network();
    seeds(Con.Seed_value);

    optimizer = th.optim.Adam(model.parameters(),lr=Con.LR);
    criterion = nn.CrossEntropyLoss();

    train_DataLoader = getDataLoader(type_="train");
    test_DataLoader = getDataLoader(type_="test");
    epoch =Con.Epochs;
    # print(test_DataLoader);
    model.load_state_dict(th.load('MY_QDR.pt'));

    for itr in range(epoch):
        # Training(itr,model,train_DataLoader,optimizer,criterion);
        Testing(model,test_DataLoader,criterion);
        # th.save(model.state_dict(),'MY_QDR3.pt')

if __name__ =='__main__':
    main()
