import numpy as np
import tqdm as tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils 
from torch.autograd import Variable

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from torchvision import transforms

import pandas as pd
from matplotlib import pyplot as plt

import torch.utils.data as Data
import os


import argparse
parser = argparse.ArgumentParser(description='compare program')
parser.add_argument('DIRECTORY', type=str)
parser.add_argument('DATASET', type=str)
parser.add_argument('MODE', type=int)
args = parser.parse_args()

from MyDataLoader import MyDataLoader
mdl = MyDataLoader(args.DATASET)
train_loader, val_loader, test_loader = mdl.get_dataloaders()
Nc, Ns, Nt = mdl.Nc, mdl.Ns, mdl.Nt

import myconfig
test_dir = args.DIRECTORY
models = myconfig.models

#default
hour = np.arange(Nt)
pset = np.arange(8)

import sys
mode = 0
print('mode:', args.MODE)
if args.MODE == 1:
    mode = 1
    hour = [0, 1, 2, 10, 11]
    pset = [0, 1, 2, 3, 4]
elif args.MODE == 2:
    mode = 2
    hour = [4, 5, 6, 7, 8, 9, 10]
    pset = [5, 6, 7]

from sklearn.metrics import mean_squared_error, mean_absolute_error
def MAPE(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100 

def model_test_result(model_name, fname):
    MyModel = models[model_name]

    Ac = mdl.Ac
    As = mdl.As
    cmask = As@As@Ac + As@Ac + Ac
    cmask[cmask != 0] = 1
    smask = (mdl.DAs + mdl.DAs.T + np.eye(Ns))


    cmask = 1-cmask
    smask = 1-smask
    n2v = mdl.N2V
    nAc = mdl.Ac / mdl.Ac.sum(axis=0)
    c2v = nAc.T @ mdl.N2V
    Ms = mdl.Ms

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #print(device)
    #device = torch.device("cpu")
    #print(device)
    
    n2v = torch.tensor(n2v).to(device).float()
    c2v = torch.tensor(c2v).to(device).float()
    cmask = torch.tensor(cmask).to(device).float()
    smask = torch.tensor(smask).to(device).float()
    Ac = torch.tensor(mdl.Ac).to(device).float()
    As = torch.tensor(mdl.As).to(device).float()
    Ms = torch.tensor(Ms).to(device).float()
    G = MyModel(Nc, Ns, Nt, c2v, n2v, Ms, Ac, As, cmask, smask).to(device)
    
    G.load_state_dict(torch.load(fname, map_location=device))
    G.eval()

    
    __y_true = []
    __y_pred = []
    __p_info = []

    for step, (b_x, b_m, b_c, b_p) in enumerate(test_loader):
        batch_size = b_x.size(0)              # batch size of sequences
        b_x = Variable(b_x.to(device)).float()     # put tensor in Variable
        b_m = Variable(b_m.to(device)).float()
        b_c = Variable(b_c.to(device)).float()
        b_p = Variable(b_p.to(device)).float()

        ########################### Train Generator #############################
        G.zero_grad()

        G_data = G(b_m, b_c, b_p)
        __p_info.append(b_p.detach().cpu().numpy())

        __y_true.append(b_x.detach().cpu().numpy())
        __y_pred.append(G_data.detach().cpu().numpy())


    t_y_pred = np.concatenate(__y_pred)
    np.save(fname+'_y_pred.npy', t_y_pred)

    t_y_true = np.concatenate(__y_true)
    np.save(fname+'_y_true.npy', t_y_true)


    _p_info = np.concatenate(__p_info)
    p_cond = _p_info[:, pset].sum(axis=1) > 0

    __y_true = np.concatenate(__y_true)[p_cond][:, :, hour]
    __y_pred = np.concatenate(__y_pred)[p_cond][:, :, hour]
    _y_true = mdl.denormalize(__y_true)
    _y_pred = mdl.denormalize(__y_pred)

    y_true = _y_true.reshape(-1)
    y_pred = _y_pred.reshape(-1)


    mse = mean_squared_error(y_true, y_pred, squared=True)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)
    mape = MAPE(y_true, y_pred)

    del G, y_true, y_pred

    return (mse, rmse, mae, mape)



perf_dict = dict()
fname_dict = dict()
for fname in sorted(os.listdir(test_dir)):
    if fname[-4:] == '.txt' and fname[:6] != 'record':
        df = pd.read_csv(test_dir+'/'+fname, sep='\t')
        name = '_'.join(fname.split('.')[0].split('_')[:-2])
        if name not in models:
            continue
        
        if name not in perf_dict or perf_dict[name] > df[df.columns[4]].min():
            target = fname[:-3] + 'pkl'
            if target not in os.listdir(test_dir):
                continue
            perf_dict[name] = df[df.columns[4]].min()
            fname_dict[name] = target



records = []
for name in models:
    if name not in fname_dict:
        continue
    test_result = model_test_result(name, test_dir+'/'+fname_dict[name])

    record = (name+'                         ')[:20] +  '\t' + \
                '\t'.join(['{:8.2f}'.format(s) for s in test_result])
    records.append(record)
    print(record)

with open(args.DIRECTORY + '/' + 'record_{}.txt'.format(mode), 'w') as fp:
    for rec in records:
        fp.write(rec + '\n')
        #print(rec)
