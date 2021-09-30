import os, sys
import datetime
import numpy as np
import pandas as pd
import tqdm as tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils 
import torch.utils.data as Data
from torch.autograd import Variable
from torchvision import transforms
from sklearn.metrics import mean_squared_error, mean_absolute_error

def MAPE(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100 


import myconfig
from MyDataLoader import MyDataLoader

now_checkpoint = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

import argparse
parser = argparse.ArgumentParser(description='test program')
parser.add_argument('GPU', type=int, default=0)
parser.add_argument('MODEL', type=str)
parser.add_argument('DATASET', type=str)
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPU)

mdl = MyDataLoader(args.DATASET)
train_loader, val_loader, test_loader = mdl.get_dataloaders()
Nc, Ns, Nt = mdl.Nc, mdl.Ns, mdl.Nt

MyModel = myconfig.models[args.MODEL]
num_epoch = myconfig.num_epoch
epoch_decay = myconfig.epoch_decay
learning_rate = myconfig.learning_rate

if not os.path.isdir(myconfig.testdir):
    os.mkdir(myconfig.testdir)


G_loadfile = None
G_savename = '{}/{}_{}.pkl'.format(myconfig.testdir, args.MODEL, now_checkpoint)


last_mae = 100
curr_early_iter = 0
def model_test(epoch, G, device, mdl, model_name, val_loader):
    global last_mae, curr_early_iter
    __y_true = []
    __y_pred = []

    for step, (b_x, b_m, b_c, b_p) in enumerate(val_loader):
        batch_size = b_x.size(0)              # batch size of sequences
        b_x = Variable(b_x.to(device)).float()     # put tensor in Variable
        b_m = Variable(b_m.to(device)).float()
        b_c = Variable(b_c.to(device)).float()
        b_p = Variable(b_p.to(device)).float()

        ########################### Train Generator #############################
        G.zero_grad()

        G_data = G(b_m, b_c, b_p)

        __y_true.append(b_x.detach().cpu().numpy())
        __y_pred.append(G_data.detach().cpu().numpy())

    __y_true = np.concatenate(__y_true)
    __y_pred = np.concatenate(__y_pred)
    _y_true = mdl.denormalize(__y_true)
    _y_pred = mdl.denormalize(__y_pred)


    y_true = _y_true.reshape(-1)
    y_pred = _y_pred.reshape(-1)


    mse = mean_squared_error(y_true, y_pred, squared=True)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)
    mape = MAPE(y_true, y_pred)

    print("%d\tMODEL_VAL:\t%5.3f\t%5.3f\t%5.3f\t%5.3f\t"%(epoch, mse, rmse, mae, mape))

    with open('{}/{}_{}.txt'.format(myconfig.testdir, model_name, now_checkpoint), 'a+') as fp:
        fp.write(("%d\tMODEL_TEST:\t%6.4f\t%6.4f\t%6.4f\t%6.4f\t\n"%(epoch, mse, rmse, mae, mape)))


    if not myconfig.early_stop:
        torch.save(G.state_dict(), G_savename)
    else:
        if last_mae > mae:
            last_mae = mae
            torch.save(G.state_dict(), G_savename)
            curr_early_iter = 0
        curr_early_iter += 1
        if curr_early_iter > myconfig.early_iter:
            sys.exit(0)
            
    return mae


def main():
    print("G_savename:", G_savename)
    print(Nc, Ns, Nt)


    Ac = mdl.Ac
    As = mdl.As
    cmask = As@As@Ac + As@Ac + Ac
    cmask[cmask != 0] = 1
    smask = (mdl.DAs + mdl.DAs.T + np.eye(Ns))
    
    Ac = Ac/Ac.sum(axis=0,keepdims=1)
    As = As/As.sum(axis=0,keepdims=1)

    nAc = mdl.Ac / mdl.Ac.sum(axis=0)
    c2v = nAc.T @ mdl.N2V

    cmask = 1-cmask
    smask = 1-smask
    n2v = mdl.N2V
    Ms = mdl.Ms

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    
    n2v = torch.tensor(n2v).to(device).float()
    c2v = torch.tensor(c2v).to(device).float()
    cmask = torch.tensor(cmask).to(device).float()
    smask = torch.tensor(smask).to(device).float()
    Ac = torch.tensor(mdl.Ac).to(device).float()
    As = torch.tensor(mdl.As).to(device).float()
    Ms = torch.tensor(Ms).to(device).float()
    G = MyModel(Nc, Ns, Nt, c2v, n2v, Ms, Ac, As, cmask, smask).to(device)

    BCE_loss = nn.BCELoss()
    L1_loss = nn.L1Loss()
    MSE_loss = nn.MSELoss()


    if G_loadfile != None:
        G.load_state_dict(torch.load(G_loadfile))

    opt_G = torch.optim.Adam(G.parameters(), lr=learning_rate)
    # training
    lr = 1
    G_losses = []
    T_losses = []
    train_x = []
    test_x = []
    for epoch in tqdm.tqdm(range(num_epoch)):
        if epoch == epoch_decay:  # or epoch == 15:
            opt_G.param_groups[0]['lr'] /= 10
            
        for step, (b_x, b_m, b_c, b_p) in enumerate(train_loader):
            batch_size = b_x.size(0)              # batch size of sequences
            b_x = Variable(b_x.to(device)).float()     # put tensor in Variable
            b_m = Variable(b_m.to(device)).float()
            b_c = Variable(b_c.to(device)).float()
            b_p = Variable(b_p.to(device)).float()
            
            ########################### Train Generator #############################
            G.zero_grad()
            G_data = G(b_m, b_c, b_p)
            G_loss = L1_loss(G_data, b_x)
            G_loss.backward()
            opt_G.step()

            
        G_losses.append(G_loss.item())
        train_x.append(epoch)
        
        if epoch%1 == 0:
            test_loss = model_test(epoch, G, device, mdl, args.MODEL, val_loader)
            test_x.append(epoch)
            T_losses.append(test_loss)
            plt.figure()
            plt.plot(train_x, G_losses)
            plt.plot(test_x, T_losses)
            plt.savefig('{}/{}_{}.png'.format(myconfig.testdir, args.MODEL, now_checkpoint))
            plt.close()


if __name__ == "__main__":
    # execute only if run as a script
    main()

