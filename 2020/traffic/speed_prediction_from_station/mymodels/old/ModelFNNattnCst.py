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



class ModelFNNattnCst(nn.Module):
    def __init__(self, Nc, Ns, Nt, c2v, n2v, Ms, cmask, smask, 
                 d_m=10, d_ms=9, d_v=64, d_h=64, d_f=256, d_p=8, heads=1):
        super(ModelFNNattnCst, self).__init__()
        
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

        self.model_c2 = nn.Sequential(
            nn.Linear(self.d_h*2, self.d_h),
            nn.ReLU(),
        )

        #self.embS = self.n2v #Parameter(torch.eye(self.Ns, self.d_h), requires_grad=True)
        self.embT = Parameter(torch.randn(self.Nt, self.d_h), requires_grad=True)
        self.embC = Parameter(torch.randn(self.Nc, self.d_h), requires_grad=True)

        self.attn_cs = torch.nn.MultiheadAttention(self.d_h, 1)
        self.norm_cs1 = Norm(self.d_h)
        self.norm_cs2 = Norm(self.d_h)
        self.ffn_cs = nn.Sequential(nn.Linear(self.d_h, self.d_f), nn.ReLU(), nn.Linear(self.d_f, self.d_h), nn.ReLU())

        self.model_embC = nn.Sequential(
            nn.Linear(self.d_h, self.d_h),
            nn.ReLU()
        )
        self.model_embS = nn.Sequential(
            nn.Linear(self.d_h, self.d_h),
            nn.ReLU(),
        )
        self.model_embP = nn.Sequential(
            nn.Linear(self.d_p, self.d_h),
            nn.ReLU(),
        )
        self.model_embT = nn.Sequential(
            nn.Linear(self.d_h*2, self.d_h),
            nn.ReLU(),
        )
        self.model_embST = nn.Sequential(
            nn.Linear(self.d_h*2, self.d_h),
            nn.ReLU(),
        )
        self.model_embCT = nn.Sequential(
            nn.Linear(self.d_h*2, self.d_h),
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

        embC = self.model_embC(self.embC).view(1, self.Nc, 1, self.d_h)
        embC = embC.repeat(bs, 1, self.Nt, 1)
        embS = self.model_embS(self.n2v).view(1, self.Ns, 1, self.d_h).repeat(bs, 1, self.Nt, 1)
        embP = self.model_embP(p.view(bs, 1, 1, self.d_p)).repeat(1, 1, self.Nt, 1)
        embT = self.embT.view(1, 1, self.Nt, self.d_h).repeat(bs, 1, 1, 1)
        embT = self.model_embT(torch.cat((embT, embP), dim=-1))
        embT4S = embT.repeat(1, self.Ns, 1, 1)
        embT4C = embT.repeat(1, self.Nc, 1, 1)

        embST = self.model_embST(torch.cat((embS, embT4S), dim=-1))
        embCT = self.model_embCT(torch.cat((embC, embT4C), dim=-1))

        c = self.model_c2(torch.cat((c, embC), dim=-1))
        c = c.permute(1, 2, 0, 3).reshape(self.Nc*self.Nt, bs, self.d_h)
        es = embST.permute(1, 2, 0, 3).reshape(self.Ns*self.Nt, bs, self.d_h)
        ec = embCT.permute(1, 2, 0, 3).reshape(self.Nc*self.Nt, bs, self.d_h)

        s = self.attn_cs(es, c, c)[0] #es, ec, c
        s = self.norm_cs2(s + self.ffn_cs(s))

        s = s.view(self.Ns, self.Nt, bs, self.d_h).permute(2, 0, 1, 3)

        x = torch.cat((s, embS, embT4S), dim=-1)
        x = self.model_final(x).view(bs, Ns, Nt)
        return x
