import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init
from torch.autograd import Variable

import math
import numpy as np

from model.deconv_autoencoder.silu import SiLU
from model.deconv_autoencoder.maxout import Maxout
from model.deconv_autoencoder.ptanh import PTanh


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class ConvolutionEncoder(nn.Module):
    def __init__(self, embedding, sentence_len, filter_size, filter_shape, latent_size):
        super(ConvolutionEncoder, self).__init__()
        self.embed = embedding
        self.convs1 = nn.Sequential(
                nn.Conv2d(1, filter_size, (filter_shape, self.embed.weight.size()[1]), stride=(2,1)),
                nn.BatchNorm2d(filter_size),
                PTanh(penalty=0.25)
            )

        self.convs2 = nn.Sequential(
                nn.Conv2d(filter_size, filter_size * 2, (filter_shape, 1), stride=(2,1)),
                nn.BatchNorm2d(filter_size * 2),
                PTanh(penalty=0.25)
            )

        self.convs3 = nn.Sequential(
                nn.Conv2d(filter_size * 2, latent_size, (sentence_len, 1), stride=(1,1)),
                nn.Tanh()
            )

        # weight initialize for conv layer
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0.001)

    def __call__(self, x):
        x = self.embed(x)
        # x.size() is (L, emb_dim) if batch_size is 1.
        # So interpolate x's dimension if batch_size is 1.
        if len(x.size()) < 3:
            x = x.view(1, *x.size())
        # reshape for convolution layer
        x = x.view(x.size()[0], 1, x.size()[1], x.size()[2])
        # x = F.relu(x)
        h1 = self.convs1(x)
        h2 = self.convs2(h1)
        h3 = self.convs3(h2)
        return h3


class DeconvolutionDecoder(nn.Module):
    def __init__(self, embedding, tau, sentence_len, filter_size, filter_shape, latent_size):
        super(DeconvolutionDecoder, self).__init__()
        self.tau = tau
        self.embed = embedding
        self.deconvs1 = nn.Sequential(
                nn.ConvTranspose2d(latent_size, filter_size * 2, (sentence_len, 1), stride=(1,1)),
                nn.BatchNorm2d(filter_size * 2),
                PTanh(penalty=0.25)
            )
        self.deconvs2 = nn.Sequential(
                nn.ConvTranspose2d(filter_size * 2, filter_size, (filter_shape, 1), stride=(2,1)),
                nn.BatchNorm2d(filter_size),
                PTanh(penalty=0.25)
            )
        self.deconvs3 = nn.Sequential(
                nn.ConvTranspose2d(filter_size, 1, (filter_shape, self.embed.weight.size()[1]), stride=(2,1)),
                nn.Tanh()
            )
        self.log_softmax = nn.LogSoftmax(dim=2)
        self.CosineSimilarity = nn.CosineSimilarity(dim=2)

        # weight initialize for conv_transpose layer
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):      
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0.001)

    def __call__(self, h):
        h2 = self.deconvs1(h)
        h1 = self.deconvs2(h2)
        x_hat = self.deconvs3(h1)
        x_hat = x_hat.squeeze()
        W = Variable(self.embed.weight.data).to(device)

        # x.size() is (L, emb_dim) if batch_size is 1.
        # So interpolate x's dimension if batch_size is 1.
        if len(x_hat.size()) < 3:
            x_hat = x_hat.view(1, *x_hat.size())
            
        # normalize
        norm_x_hat = torch.norm(x_hat, 2, dim=2, keepdim=True)
        norm_W = torch.norm(W, 2, dim=1, keepdim=True)

        normalized_x_hat = x_hat / norm_x_hat
        normalized_W = W / norm_W

        # calculate logit and softmax
        prob_logits = torch.tensordot(normalized_x_hat, normalized_W, dims=([2], [1]))
        log_prob = self.log_softmax(prob_logits / self.tau)
        return log_prob