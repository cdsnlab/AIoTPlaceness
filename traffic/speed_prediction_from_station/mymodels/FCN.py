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




class FCN(nn.Module):
    def __init__(self, Nc, Ns, Nt, c2v, n2v, Ms, Ac, As, cmask, smask, 
                 d_m=10, d_ms=9, d_v=64, d_h=64, d_f=1024, d_p=8, heads=1):
        super(FCN, self).__init__()
        
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
         
        self.model = nn.Sequential(
            nn.Linear(self.Nc*self.Nt*2, 2048),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(2048, self.Ns*self.Nt),
        )

    def forward(self, m, c, p):
        Nc, Ns, Nt = self.Nc, self.Ns, self.Nt
        bs = m.shape[0]
        c = c.view(bs, -1)
        x = self.model(c).view(bs, Ns, Nt)
        return x
