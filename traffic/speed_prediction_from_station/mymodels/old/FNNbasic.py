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




class FNNbasic(nn.Module):
    def __init__(self, Nc, Ns, Nt, c2v, n2v, Ms, cmask, smask, 
                 d_m=10, d_ms=9, d_v=64, d_h=64, d_f=64, heads=1):
        super(FNNbasic, self).__init__()
        
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
            nn.Linear(Nc*Nt*2+8, 2048),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(1024, Ns*Nt)
        )

    def forward(self, m, c, p):
        Nc, Ns, Nt = self.Nc, self.Ns, self.Nt
        bs = m.shape[0]

        c = c.view(bs, Nc*Nt*2)
        x = torch.cat((c, p), dim=-1)

        x = self.model(x).view(bs, Ns, Nt)
        return x
