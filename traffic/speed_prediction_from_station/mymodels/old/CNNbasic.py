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



class CNNbasic(nn.Module):
    def __init__(self, Nc, Ns, Nt, c2v, n2v, Ms, cmask, smask, 
                 d_m=10, d_ms=9, d_v=64, d_h=48, d_f=256, d_p=8, heads=1):
        super(CNNbasic, self).__init__()
        
        self.Nc, self.Ns, self.Nt = Nc, Ns, Nt
        self.n2v = n2v
        self.c2v = c2v
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
        self.model_cnn1 = torch.nn.Conv1d(self.Nc*2, 1024, kernel_size=5, padding=2)
        self.model_cnn2 = torch.nn.Conv1d(1024, self.Ns, kernel_size=5, padding=2)

        self.model_c2 = nn.Sequential(
            nn.Linear(self.d_h*2, self.d_h),
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
        self.embC = Parameter(torch.randn(self.Nc, self.d_h), requires_grad=True)
        #self.embC = self.c2v
        self.embT = Parameter(torch.randn(self.Nt, self.d_h), requires_grad=True)

        self.model_embS = nn.Sequential(
            nn.Linear(self.d_v, self.d_h),
            nn.ReLU(),
        )
        self.model_embC = nn.Sequential(
            nn.Linear(self.d_h, self.d_h),
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
        self.model_embMMST = nn.Sequential(
            nn.Linear(self.d_h*4, self.d_h*2),
            nn.ReLU(),
            nn.Linear(self.d_h*2, self.d_h),
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
        self.model_final = nn.Sequential(
            nn.Linear(self.d_h*2, self.d_f),
            nn.ReLU(),
            nn.Linear(self.d_f, 1),
            nn.ReLU(),
        )

        
        self.attnTC = SelfAttnBlock(self.Nc, self.Nt, self.d_h, 1, 1)

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
            nn.Linear(self.d_h*4, self.d_f),
            nn.ReLU(),
            nn.Linear(self.d_f, self.d_h),
            nn.ReLU()
        )
        self.attn_cs = nn.MultiheadAttention(self.d_h*self.heads, self.heads)
        self.norm_cs = Norm(self.d_h)
        self.ffn_cs = nn.Sequential(nn.Linear(self.d_h, self.d_f), nn.ReLU(), nn.Dropout(0.1),
                                    nn.Linear(self.d_f, self.d_h), nn.ReLU())

    def forward(self, m, c, p):
        Nc, Ns, Nt = self.Nc, self.Ns, self.Nt
        bs = m.shape[0]

        c = c.permute(0, 1, 3, 2)
        c = c.reshape(bs, self.Nc*2, self.Nt)
        c = self.model_cnn1(c)   
        #c = c.view(bs, self.Ns, self.Nt)
        c = self.model_cnn2(c)   
        c = c.view(bs, self.Ns, self.Nt)

        return c


