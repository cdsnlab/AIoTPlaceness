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


class GCNFast(nn.Module):
    def __init__(self, Nc, Ns, Nt, d_h, Ac, As):
        super(GCNFast, self).__init__()
        self.Nc, self.Ns, self.Nt, self.d_h = Nc, Ns, Nt, d_h
        self.Ac, self.As = Ac, As
        self.AA = As@As@Ac + As@Ac + Ac
        self.AA = self.AA.masked_fill(self.AA != 0, 1)
        #self.AA = self.AA / self.AA.sum(0, keepdim=True)[0]
        
        
        self.gcn_mask = Parameter(torch.eye(self.Nt)
                                  .repeat_interleave(self.Ns, dim=0)
                                  .repeat_interleave(self.Nc, dim=1), requires_grad=False)
        
        self.GCW = Parameter(torch.randn(self.Ns*self.Nt, self.Nc*self.Nt), requires_grad=True)
        self.GCB = Parameter(torch.randn(self.Ns*self.Nt, self.d_h), requires_grad=True)
        self.GCL = nn.Linear(self.d_h, self.d_h, bias=True)
        
    def forward(self, h, e):
        bs = h.shape[0]
        #h = torch.cat((h, e), dim=-1)
        A = F.relu(self.AA.repeat(self.Nt, self.Nt)  * self.GCW)
        h = h.permute(0, 2, 1, 3)
        h = h.reshape(bs, self.Nt*self.Nc, self.d_h)
        h = F.relu(A @ h + self.GCB)
        h = h.view(bs, self.Nt, self.Ns, self.d_h)
        h = h.permute(0, 2, 1, 3)
        h = h.reshape(bs, self.Ns, self.Nt, self.d_h)
        return h
    

class ETBlock(nn.Module):
    def __init__(self, Nc, Ns, Nt, d_h, d_f):
        super(ETBlock, self).__init__()
        self.Nc, self.Ns, self.Nt, self.d_h, self.d_f = Nc, Ns, Nt, d_h, d_f
        self.attnTC = SelfAttnBlock(self.Nc, self.Nt, self.d_h, 1, 1)

    def forward(self, h, e):
        bs = h.shape[0]
        h = self.attnTC(h, e)
        return h
        

class ESBlock(nn.Module):
    def __init__(self, Nc, Ns, Nt, d_h, d_f, Ac, As):
        super(ESBlock, self).__init__()
        self.Nc, self.Ns, self.Nt, self.d_h, self.d_f = Nc, Ns, Nt, d_h, d_f
        self.Ac, self.As = Ac, As
        
        self.mGCNFast = GCNFast(Nc, Ns, Nt, d_h, Ac, As)
        
        
    def forward(self, h, e):
        bs = h.shape[0]
        h = self.mGCNFast(h, e)
        return h
        
        
class DTBlock(nn.Module):
    def __init__(self, Nc, Ns, Nt, d_h, d_f):
        super(DTBlock, self).__init__()
        self.Nc, self.Ns, self.Nt, self.d_h, self.d_f = Nc, Ns, Nt, d_h, d_f
        self.attnT = SelfAttnBlock(self.Ns, self.Nt, self.d_h, 1, 1)

    def forward(self, h, e):
        bs = h.shape[0]
        h = self.attnT(h, e)
        return h
        
        
class DSBlock(nn.Module):
    def __init__(self, Nc, Ns, Nt, d_h, d_f, As):
        super(DSBlock, self).__init__()
        self.Nc, self.Ns, self.Nt, self.d_h, self.d_f = Nc, Ns, Nt, d_h, d_f
        self.As = As
        self.attnS = SelfAttnBlock(self.Ns, self.Nt, self.d_h, 1, 0)

    def forward(self, h, e):
        bs = h.shape[0]
        h = self.attnS(h, e)
        return h
        
        

class OurModelFast(nn.Module):
    def __init__(self, Nc, Ns, Nt, c2v, n2v, Ms, cmask, smask, Ac, As, 
                 d_mv=117, d_ms=9, d_v=64, d_p=8, d_h=64, d_f=256, heads=2, optM=False):
        super(OurModelFast, self).__init__()
        
        self.Nc, self.Ns, self.Nt = Nc, Ns, Nt
        self.Ac, self.As = Ac, As
        self.n2v, self.Ms = n2v, Ms
        self.cmask, self.smask = cmask, smask
        self.d_mv, self.d_ms, self.d_mm = d_mv, d_ms, d_mv + d_ms
        self.d_v, self.d_h, self.d_f, self.d_p = d_v, d_h, d_f, d_p
        self.heads = heads
        self.optM = optM
         
        
        self.embQ = Parameter(torch.randn(self.Nt, self.d_h), requires_grad=True)
        self.embC = Parameter(torch.randn(self.Nc, self.d_h), requires_grad=True)
        self.model_embP = nn.Sequential(nn.Linear(self.d_p, self.d_h//4), nn.ReLU())
        self.model_embQ = nn.Sequential(nn.Linear(self.d_h, self.d_h//4), nn.ReLU())
        self.model_embC = nn.Sequential(nn.Linear(self.d_h, self.d_h//2), nn.ReLU())
        
        if self.optM:
            self.model_embM = nn.Sequential(nn.Linear(self.d_mm, self.d_h//4), nn.ReLU())
            self.model_embS = nn.Sequential(nn.Linear(self.d_v, self.d_h//4), nn.ReLU())
        else:
            self.model_embS = nn.Sequential(nn.Linear(self.d_v, self.d_h//2), nn.ReLU())
            
        self.model_front = nn.Sequential(
            nn.Linear(2, self.d_f),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(self.d_f, self.d_h),
            nn.ReLU(),
        )
                
        self.model_final = nn.Sequential(
            nn.Linear(self.d_h, self.d_f),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(self.d_f, 1),
            nn.ReLU(),
        )
        
        self.mETBlock = ETBlock(Nc, Ns, Nt, d_h, d_f)
        self.mESBlock = ESBlock(Nc, Ns, Nt, d_h, d_f, Ac, As)
        self.mDTBlock1 = DTBlock(Nc, Ns, Nt, d_h, d_f)
        self.mDSBlock = DSBlock(Nc, Ns, Nt, d_h, d_f, As)
        self.mDTBlock2 = DTBlock(Nc, Ns, Nt, d_h, d_f)

    def forward(self, m, c, p):
        Nc, Ns, Nt = self.Nc, self.Ns, self.Nt
        bs = m.shape[0]

        if self.optM:
            ms = self.Ms.view(1, self.Ns, self.d_ms).repeat(bs, 1, 1)
            mv = m.view(bs, self.Ns, self.d_mv)
            m = torch.cat((mv, ms), dim=-1)
            embM = self.model_embM(m).view(bs, self.Ns, 1, self.d_h//4)
            embS = self.model_embS(self.n2v).view(1, self.Ns, 1, self.d_h//4).repeat(bs, 1, 1, 1)
            embS = torch.cat((embS, embM), dim=-1).repeat(1, 1, self.Nt, 1)
        else:
            embS = self.model_embS(self.n2v).view(1, self.Ns, 1, self.d_h//2).repeat(bs, 1, self.Nt, 1)
            
        embP = self.model_embP(p.view(bs, 1, 1, self.d_p)).repeat(1, 1, self.Nt, 1)
        embQ = self.model_embQ(self.embQ).view(1, 1, self.Nt, self.d_h//4).repeat(bs, 1, 1, 1)
        embT = torch.cat((embP, embQ), dim=-1)
        embT4S = embT.repeat(1, self.Ns, 1, 1)
        embT4C = embT.repeat(1, self.Nc, 1, 1)
        embC = self.model_embC(self.embC)
        embC = embC.view(1, self.Nc, 1, self.d_h//2).repeat(bs, 1, self.Nt, 1)
        embCT = torch.cat((embC, embT4C), dim=-1)
        embST = torch.cat((embS, embT4S), dim=-1)        
        
        
        x = self.model_front(c)
        
        x = self.mETBlock (x, embCT)
        x = self.mESBlock (x, embCT)
        x = self.mDTBlock1(x, embST)
        x = self.mDSBlock (x, embST)
        x = self.mDTBlock2(x, embST)
        
        #x = torch.cat((x, embST), dim=-1)
        x = self.model_final(x).view(bs, Ns, Nt)
        return x
