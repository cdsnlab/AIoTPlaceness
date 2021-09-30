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




class FNNemb(nn.Module):
    def __init__(self, Nc, Ns, Nt, c2v, n2v, Ms, Ac, As, cmask, smask, 
                 d_m=10, d_ms=9, d_v=64, d_h=64, d_f=1024, d_p=8, heads=1):
        super(FNNemb, self).__init__()
        
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

        c = self.model_c(c)
        c = c.permute(0, 3, 2, 1)
        s = self.model_cs(c)
        s = s.permute(0, 3, 2, 1)

        embS = self.model_embS(self.n2v).view(1, self.Ns, 1, self.d_h).repeat(bs, 1, self.Nt, 1)
        embP = self.model_embP(p.view(bs, 1, 1, self.d_p)).repeat(1, 1, self.Nt, 1)
        embT = self.embT.view(1, 1, self.Nt, self.d_h).repeat(bs, 1, 1, 1)
        embT = self.model_embT(torch.cat((embT, embP), dim=-1)).repeat(1, self.Ns, 1, 1)

        x = torch.cat((s, embS, embT), dim=-1)
        x = self.model_final(x).view(bs, Ns, Nt)
        return x
