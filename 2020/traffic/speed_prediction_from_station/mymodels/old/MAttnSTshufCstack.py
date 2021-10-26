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





class MAttnSTshufCstack(nn.Module):
    def __init__(self, Nc, Ns, Nt, c2v, n2v, Ms, cmask, smask, 
                 d_m=10, d_ms=9, d_v=64, d_h=48, d_f=256, d_p=8, heads=1):
        super(MAttnSTshufCstack, self).__init__()
        
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
         
        self.model_m = nn.Sequential(
            nn.Linear(self.d_m, self.d_h),
            nn.ReLU(),
            nn.Linear(self.d_h, self.d_h),
            nn.ReLU()
        )
        
        self.model_ms = nn.Sequential(
            nn.Linear(self.d_ms, self.d_h),
            nn.ReLU(),
            nn.Linear(self.d_h, self.d_h),
            nn.ReLU()
        )
        self.model_c = nn.Sequential(
            nn.Linear(2, self.d_h),
            nn.ReLU(),
            nn.Linear(self.d_h, self.d_h),
            nn.ReLU(),
        )

        self.model_cs = nn.Sequential(
            nn.Linear(self.Nc*self.Nt*2, self.d_f),
            nn.ReLU(),
            nn.Linear(self.d_f, self.Ns*self.Nt*self.d_h),
            nn.ReLU(),
        )

        #self.embS = self.n2v #Parameter(torch.eye(self.Ns, self.d_h), requires_grad=True)
        self.embT = Parameter(torch.randn(self.Nt, self.d_h), requires_grad=True)

        self.model_embS = nn.Sequential(
            nn.Linear(self.d_v, self.d_h),
            nn.ReLU(),
            nn.Linear(self.d_h, self.d_h),
            nn.ReLU(),
        )
        self.model_embST = nn.Sequential(
            nn.Linear(self.d_h*2, self.d_h*2),
            nn.ReLU(),
            nn.Linear(self.d_h*2, self.d_h),
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
            nn.Linear(self.d_h*2, self.d_f),
            nn.ReLU(),
            nn.Linear(self.d_f, 1),
            nn.ReLU(),
        )

        self.attnS1 = SelfAttnBlock(self.Ns, self.Nt, self.d_h, 1, 0)
        self.attnT1 = SelfAttnBlock(self.Ns, self.Nt, self.d_h, 1, 1)
        self.attnS2 = SelfAttnBlock(self.Ns, self.Nt, self.d_h, 1, 0)
        self.attnT2 = SelfAttnBlock(self.Ns, self.Nt, self.d_h, 1, 1)
        self.attnS3 = SelfAttnBlock(self.Ns, self.Nt, self.d_h, 1, 0)
        self.attnT3 = SelfAttnBlock(self.Ns, self.Nt, self.d_h, 1, 1)
        self.model_ST1 = nn.Sequential(nn.Linear(self.d_h*2, self.d_h), nn.ReLU())
        self.model_ST2 = nn.Sequential(nn.Linear(self.d_h*2, self.d_h), nn.ReLU())
        self.model_ST3 = nn.Sequential(nn.Linear(self.d_h*2, self.d_h), nn.ReLU())

        self.model_mss = nn.Sequential(
            nn.Linear(self.d_h*5, self.d_f),
            nn.ReLU(),
            nn.Linear(self.d_f, self.d_h),
            nn.ReLU()
        )

    def forward(self, m, c, p):
        Nc, Ns, Nt = self.Nc, self.Ns, self.Nt
        bs = m.shape[0]

        mv = self.model_m(m)
        mv = mv.unsqueeze(2)#.repeat(1, 1, Nt, 1)
        
        ms = self.Ms.view(1, self.Ns, 1, self.d_ms)
        ms = self.model_ms(ms)
        ms = ms.repeat(bs, 1, 1, 1)

        embMS = torch.cat((mv, ms), dim=-1)
        embMS = embMS.repeat(1, 1, self.Nt, 1)
        
        embS = self.model_embS(self.n2v).view(1, self.Ns, 1, self.d_h).repeat(bs, 1, self.Nt, 1)
        embP = self.model_embP(p.view(bs, 1, 1, self.d_p)).repeat(1, 1, self.Nt, 1)
        embT = self.embT.view(1, 1, self.Nt, self.d_h).repeat(bs, 1, 1, 1)
        embT = self.model_embT(torch.cat((embT, embP), dim=-1)).repeat(1, self.Ns, 1, 1)
        embST = self.model_embST(torch.cat((embS, embT), dim=-1))

        embMST = torch.cat((embMS, embS, embT), dim=-1)


        c = c.view(bs, Nc*Nt*2)
        s = self.model_cs(c)
        s = s.view(bs, self.Ns, self.Nt, self.d_h)

        s = self.model_mss(torch.cat((s, embMST), dim=-1))


        x = s
        x = self.attnS1(x, embST)
        x = self.attnT1(x, embST)
        #x = self.model_ST2(torch.cat((self.attnS2(x, embST), self.attnT2(x, embST)), dim=-1))
        #x = self.model_ST3(torch.cat((self.attnS3(x, embST), self.attnT3(x, embST)), dim=-1))

        x = torch.cat((x, embST), dim=-1)
        x = self.model_final(x).view(bs, Ns, Nt)
        return x

