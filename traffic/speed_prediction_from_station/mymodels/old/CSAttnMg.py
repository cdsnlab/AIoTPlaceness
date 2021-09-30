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



class CSAttnMg(nn.Module):
    def __init__(self, Nc, Ns, Nt, c2v, n2v, Ms, cmask, smask, 
                 d_m=10, d_ms=9, d_v=64, d_h=48, d_f=256, d_p=8, heads=1):
        super(CSAttnMg, self).__init__()
        
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

        mv = self.model_m(m)
        mv = mv.unsqueeze(2)#.repeat(1, 1, Nt, 1)
        
        ms = self.Ms.view(1, self.Ns, 1, self.d_ms)
        ms = self.model_ms(ms)
        ms = ms.repeat(bs, 1, 1, 1)

        embMM = torch.cat((mv, ms), dim=-1)
        embMM = embMM.repeat(1, 1, self.Nt, 1)
        
        embS = self.model_embS(self.n2v).view(1, self.Ns, 1, self.d_h).repeat(bs, 1, self.Nt, 1)
        embP = self.model_embP(p.view(bs, 1, 1, self.d_p)).repeat(1, 1, self.Nt, 1)
        embT = self.embT.view(1, 1, self.Nt, self.d_h).repeat(bs, 1, 1, 1)
        embT = self.model_embT(torch.cat((embT, embP), dim=-1))
        embT4S = embT.repeat(1, self.Ns, 1, 1)
        embT4C = embT.repeat(1, self.Nc, 1, 1)
        
        embC = self.model_embC(self.embC)
        embC = embC.view(1, self.Nc, 1, self.d_h).repeat(bs, 1, self.Nt, 1)
        embST = self.model_embST(torch.cat((embS, embT4S), dim=-1))
        embCT = self.model_embCT(torch.cat((embC, embT4C), dim=-1))
        embMMST = self.model_embMMST(torch.cat((embMM, embS, embT4S), dim=-1))


        c = self.model_c(c)
        c = self.attnTC(c, embCT)
        
        c = self.model_c2(torch.cat((c, embCT), dim=-1))
        c = c.permute(1, 0, 2, 3).reshape(self.Nc, bs*self.Nt, self.d_h)
        es = embMMST.permute(1, 0, 2, 3).reshape(self.Ns, bs*self.Nt, self.d_h)
        ec = embCT.permute(1, 0, 2, 3).reshape(self.Nc, bs*self.Nt, self.d_h)

        s = self.attn_cs(es, ec, c, attn_mask=self.cmask)[0]
        s = self.norm_cs(s + self.ffn_cs(s))

        s = s.view(self.Ns, bs, self.Nt, self.d_h).permute(1, 0, 2, 3)
        #s = self.model_mss(torch.cat((s, embMST), dim=-1))


        x = s
        x = self.model_ST1(torch.cat((self.attnS1(x, embMMST), self.attnT1(x, embMMST)), dim=-1))
        #x = self.attnT1(x, embMMST)
        #x = self.attnS1(x, embMMST)

        x = torch.cat((x, embMMST), dim=-1)
        x = self.model_final(x).view(bs, Ns, Nt)
        return x


