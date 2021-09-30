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





class ModelFNNatt(nn.Module):
    def __init__(self, Nc, Ns, Nt, c2v, n2v, Ms, cmask, smask, 
                 d_m=10, d_ms=9, d_v=64, d_h=64, d_f=64, heads=1):
        super(ModelFNNatt, self).__init__()
        
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
        self.heads = heads
        
                        
        self.model = nn.Sequential(
            nn.Linear(Ns*7+1, 2048),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, Ns*Nt)
        )
        
        self.embdim = 64
        self.embSTR = Parameter(torch.FloatTensor(Nc*Nt*2, self.embdim).normal_(-1, 1), requires_grad=True)
        #self.embS = Parameter(torch.FloatTensor(Nc, self.embdim).normal_(-1, 1), requires_grad=True)
        #self.embT = Parameter(torch.FloatTensor(Nt, self.embdim).normal_(-1, 1), requires_grad=True)
        
        self.modelS = nn.Sequential(
            nn.Linear(self.embdim, self.embdim),
            nn.ReLU(),
        )
        self.modelT = nn.Sequential(
            nn.Linear(self.embdim, self.embdim),
            nn.ReLU(),
        )
        self.modelSTR = nn.Sequential(
            nn.Linear(self.embdim, self.embdim),
            nn.ReLU(),
            nn.Linear(self.embdim, self.embdim),
            nn.ReLU(),
        )
        
        self.attn1 = torch.nn.MultiheadAttention(self.embdim+1, 1)
        self.attn2 = torch.nn.MultiheadAttention(self.embdim+1, 1)
        self.attn3 = torch.nn.MultiheadAttention(self.embdim+1, 1)
        
        self.model_x2 = nn.Sequential(
            nn.Linear(self.embdim+1, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        
        self.model_x1 = nn.Sequential(
            nn.Linear(Nc*Nt*2, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, Ns*Nt),
            nn.ReLU(),
        )
        
    def forward(self, m, c, p):
        Nc, Ns, Nt = self.Nc, self.Ns, self.Nt
        bs = m.shape[0]
        
        embSTR = self.modelSTR(self.embSTR)
        e = embSTR.unsqueeze(0)
        e = e.repeat(bs, 1, 1)
        
        c = c.view(bs, -1, 1)
        
        
        ce = torch.cat((c, e), dim=-1)
        x = ce.permute(1, 0, 2)
        
        x2, _ = self.attn1(x, x, x)
        x = x + x2
        x2, _ = self.attn2(x, x, x)
        x = x + x2
        x2, _ = self.attn3(x, x, x)
        x = x + x2
        
        
        x = x.permute(1, 2, 0)
        x = self.model_x1(x)
        x = x.permute(0, 2, 1)
        x = self.model_x2(x)
        x = x.squeeze(-1)
        x = x.view(bs, Ns, Nt)
        #print(x.shape)
        
        return x

