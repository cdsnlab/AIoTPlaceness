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

class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.size = d_model
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))

        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm

class SelfAttnBlockN(nn.Module):
    def __init__(self, Ns, Nt, d_h, heads, direction, dropout=0.1):
        super(SelfAttnBlockN, self).__init__()

        self.Ns = Ns
        self.Nt = Nt
        self.d_h = d_h
        self.heads = heads
        self.direction = direction #0: spatio, 1: temporal

        self.attn = torch.nn.MultiheadAttention(self.d_h*2*self.heads, self.heads)
        self.pos_ffn = nn.Sequential(
            nn.Linear(self.d_h*2, self.d_h*2), 
            nn.ReLU(), 
            nn.Dropout(0.1),
            nn.Linear(self.d_h*2, self.d_h), 
            nn.ReLU()
        )

        self.norm_1 = Norm(self.d_h*2*self.heads)
        self.norm_2 = Norm(self.d_h*self.heads)
        self.dropout = dropout
        if self.dropout != None:
            self.dropout_1 = nn.Dropout(dropout)
            self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, e):  # here x includes c
        # input x <- (batch_size, sequence_length, pixel_num * feature_num)
        bs = x.shape[0]
        xori = x
        x = torch.cat((x, e), dim=-1)

        if self.direction == 0:
            x = x.permute(1, 0, 2, 3).reshape(self.Ns, bs*self.Nt, self.d_h*2)
            xori = xori.permute(1, 0, 2, 3).reshape(self.Ns, bs*self.Nt, self.d_h)
        elif self.direction == 1:
            x = x.permute(2, 0, 1, 3).reshape(self.Nt, bs*self.Ns, self.d_h*2)
            xori = xori.permute(2, 0, 1, 3).reshape(self.Nt, bs*self.Ns, self.d_h)

        if self.dropout != None:
            x = self.dropout_1(self.attn(x, x, x)[0])
        else:
            x = self.attn(x, x, x)[0]
        x = self.norm_1(x)
        if self.dropout != None:
            x = xori + self.dropout_2(self.pos_ffn(x))
        else:
            x = xori + self.pos_ffn(x)
        x = self.norm_2(x)
        
        if self.direction == 0:
            x = x.view(self.Ns, bs, self.Nt, self.d_h*self.heads).permute(1, 0, 2, 3)
        elif self.direction == 1:
            x = x.view(self.Nt, bs, self.Ns, self.d_h*self.heads).permute(1, 2, 0, 3)
        return x


class SelfAttnBlockBackup(nn.Module):
    def __init__(self, Ns, Nt, d_h, heads, direction, dropout=None):
        super(SelfAttnBlockBackup, self).__init__()

        self.Ns = Ns
        self.Nt = Nt
        self.d_h = d_h
        self.heads = heads
        self.direction = direction #0: spatio, 1: temporal

        self.attn = torch.nn.MultiheadAttention(self.d_h*self.heads, self.heads)
        self.pos_ffn = nn.Sequential(
            nn.Linear(self.d_h, self.d_h*2), 
            nn.ReLU(), 
            nn.Dropout(0.1),
            nn.Linear(self.d_h*2, self.d_h), 
            nn.ReLU()
        )
        self.model_emb = nn.Sequential(
            nn.Linear(self.d_h*2, self.d_h*2),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(self.d_h*2, self.d_h),
            nn.ReLU()
        )

        self.norm_1 = Norm(self.d_h*self.heads)
        self.norm_2 = Norm(self.d_h*self.heads)
        self.dropout = dropout
        if self.dropout != None:
            self.dropout_1 = nn.Dropout(dropout)
            self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, e):  # here x includes c
        # input x <- (batch_size, sequence_length, pixel_num * feature_num)
        bs = x.shape[0]
        x = self.model_emb(torch.cat((x, e), dim=-1))

        if self.direction == 0:
            x = x.permute(1, 0, 2, 3).reshape(self.Ns, bs*self.Nt, self.d_h)
        elif self.direction == 1:
            x = x.permute(2, 0, 1, 3).reshape(self.Nt, bs*self.Ns, self.d_h)

        if self.dropout != None:
            x = x + self.dropout_1(self.attn(x, x, x)[0])
        else:
            x = x + self.attn(x, x, x)[0]
        x = self.norm_1(x)
        if self.dropout != None:
            x = x + self.dropout_2(self.pos_ffn(x))
        else:
            x = x + self.pos_ffn(x)
        x = self.norm_2(x)
        
        if self.direction == 0:
            x = x.view(self.Ns, bs, self.Nt, self.d_h*self.heads).permute(1, 0, 2, 3)
        elif self.direction == 1:
            x = x.view(self.Nt, bs, self.Ns, self.d_h*self.heads).permute(1, 2, 0, 3)
        return x



class SelfAttnBlock(nn.Module):
    def __init__(self, Ns, Nt, d_h, heads, direction, dropout=None):
        super(SelfAttnBlock, self).__init__()

        self.Ns = Ns
        self.Nt = Nt
        self.d_h = d_h
        self.heads = heads
        self.direction = direction #0: spatio, 1: temporal

        self.attn = torch.nn.MultiheadAttention(self.d_h*2*self.heads, self.heads)
        self.pos_ffn = nn.Sequential(
            nn.Linear(self.d_h*2*self.heads, self.d_h*2), 
            nn.ReLU(), 
            nn.Dropout(0.1),
            nn.Linear(self.d_h*2, self.d_h), 
            nn.ReLU()
        )

        self.linear_h = nn.Linear(self.d_h, self.d_h*2)
        self.norm_1 = Norm(self.d_h*2*self.heads)
        self.norm_2 = Norm(self.d_h)
        self.dropout = dropout
        if self.dropout != None:
            self.dropout_1 = nn.Dropout(dropout)
            self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, e):  # here x includes c
        # input x <- (batch_size, sequence_length, pixel_num * feature_num)
        bs = x.shape[0]

        xori = x
        tx  = self.linear_h(x)
        x = torch.cat((x, e), dim=-1)

        if self.direction == 0:
            xori = xori.permute(1, 0, 2, 3).reshape(self.Ns, bs*self.Nt, self.d_h)
            x = x.permute(1, 0, 2, 3).reshape(self.Ns, bs*self.Nt, self.d_h*2)
            tx = tx.permute(1, 0, 2, 3).reshape(self.Ns, bs*self.Nt, self.d_h*2)
        elif self.direction == 1:
            xori = xori.permute(2, 0, 1, 3).reshape(self.Nt, bs*self.Ns, self.d_h)
            x = x.permute(2, 0, 1, 3).reshape(self.Nt, bs*self.Ns, self.d_h*2)
            tx = tx.permute(2, 0, 1, 3).reshape(self.Nt, bs*self.Ns, self.d_h*2)

        x = x.repeat(1, 1, self.heads)
        tx = tx.repeat(1, 1, self.heads)
        if self.dropout != None:
            x = x + self.dropout_1(self.attn(x, x, tx)[0])
        else:
            x = x + self.attn(x, x, tx)[0]
        x = self.norm_1(x)
        if self.dropout != None:
            x = xori + self.dropout_2(self.pos_ffn(x))
        else:
            x = xori + self.pos_ffn(x)
        x = self.norm_2(x)
        
        if self.direction == 0:
            x = x.view(self.Ns, bs, self.Nt, self.d_h).permute(1, 0, 2, 3)
        elif self.direction == 1:
            x = x.view(self.Nt, bs, self.Ns, self.d_h).permute(1, 2, 0, 3)
        return x