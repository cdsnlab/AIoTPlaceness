import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from torch.autograd import Variable
import torchvision
from torchvision import transforms
from torchvision import datasets
import numpy as np
import copy

import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from mymodels.submodels import *




class BiLSTMemb(nn.Module):
    def __init__(self, Nc, Ns, Nt, c2v, n2v, Ms, Ac, As, cmask, smask, 
                 d_m=10, d_ms=9, d_v=64, d_h=64, d_f=1024, d_p=8, heads=1):
        super(BiLSTMemb, self).__init__()
        
        self.Nc, self.Ns, self.Nt = Nc, Ns, Nt
        self.n2v = n2v
        self.Ms = Ms
        self.cmask = cmask
        self.smask = smask
        
        self.d_m = d_m
        self.d_ms = d_ms
        self.d_v = d_v
        self.d_h = d_h
        self.d_f = d_f
        self.d_p = d_p
        self.heads = heads
         
        self.model_c = nn.Sequential(
            nn.Linear(2, self.d_h),
            nn.ReLU(),
            nn.Linear(self.d_h, self.d_h),
            nn.ReLU(),
        )

        self.model_cs = nn.Sequential(
            nn.Linear(self.Nc, self.d_f),
            nn.ReLU(),
            nn.Linear(self.d_f, self.Ns),
            nn.ReLU(),
        )
        #self.model_cnn1 = torch.nn.Conv1d(self.Nc*2, 1024, kernel_size=5, padding=2)
        #self.model_cnn2 = torch.nn.Conv1d(1024, self.Ns*self.d_h, kernel_size=5, padding=2)
        self.model_rnn1 = nn.LSTM(2, self.d_h//2, 2, bidirectional=True)

        #self.embS = self.n2v #Parameter(torch.eye(self.Ns, self.d_h), requires_grad=True)
        self.embT = Parameter(torch.randn(self.Nt, self.d_h), requires_grad=True)

        self.model_embS = nn.Sequential(
            nn.Linear(self.d_h, self.d_h),
            nn.ReLU(),
            nn.Linear(self.d_h, self.d_h),
            nn.ReLU(),
        )
        self.model_embP = nn.Sequential(
            nn.Linear(self.d_p, self.d_h),
            nn.ReLU(),
            nn.Linear(self.d_h, self.d_h),
            nn.ReLU(),
        )
        self.model_embT = nn.Sequential(
            nn.Linear(self.d_h*2, self.d_h),
            nn.ReLU(),
            nn.Linear(self.d_h, self.d_h),
            nn.ReLU(),
        )
        self.model_final = nn.Sequential(
            nn.Linear(self.d_h*3, self.d_f),
            nn.ReLU(),
            nn.Linear(self.d_f, 1),
            nn.ReLU(),
        )

    def forward(self, m, c, p):
        Nc, Ns, Nt = self.Nc, self.Ns, self.Nt
        bs = m.shape[0]

        c = c.permute(0, 1, 3, 2)
        c = c.reshape(bs*self.Nc, 2, self.Nt)
        c = c.permute(2, 0, 1)
        #c = self.model_cnn1(c)
        c,_ = self.model_rnn1(c)
        c = c.view(self.Nt, bs, self.Nc, self.d_h)
        #c = c.permute(1, 2, 0, 3)
        c = c.permute(1, 0, 3, 2)
        s = self.model_cs(c)
        s = s.permute(0, 3, 1, 2)



        #c = c.view(bs, self.Ns, self.Nt)
        #c = self.model_cnn2(c)   
        #c = c.view(bs, self.Ns, self.d_h, self.Nt)
        #s = c.permute(0, 1, 3, 2)

        embS = self.model_embS(self.n2v).view(1, self.Ns, 1, self.d_h).repeat(bs, 1, self.Nt, 1)
        embP = self.model_embP(p.view(bs, 1, 1, self.d_p)).repeat(1, 1, self.Nt, 1)
        embT = self.embT.view(1, 1, self.Nt, self.d_h).repeat(bs, 1, 1, 1)
        embT = self.model_embT(torch.cat((embT, embP), dim=-1)).repeat(1, self.Ns, 1, 1)

        x = torch.cat((s, embS, embT), dim=-1)
        x = self.model_final(x).view(bs, Ns, Nt)
        return x
