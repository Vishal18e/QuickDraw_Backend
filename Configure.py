import torch as th
import os
import numpy as np

drop=0.25;
Inp =1;
Seed_value =0;
base = {
    'train':'GoogleDataImages_train',
    'test':'GoogleDataImages_test'
};
Epochs =1;
resz_fig =64;

LR = 0.00001;
batchSize ={
    'train':25,
    'test':10
};

trn_batch =25;
tst_batch =10;

numworkers={
    'train':3,
    'test':3
};

layers= np.array([128,128,128,128,128,128]);
