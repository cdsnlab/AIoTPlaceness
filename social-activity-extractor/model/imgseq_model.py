import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init
from torch.autograd import Variable

import math
import numpy as np

from model.component import SiLU, Maxout, PTanh

class RNNEncoder(nn.Module):
	def __init__(self, embed_dim, num_layers, latent_size, bidirectional):
		super(RNNEncoder, self).__init__()
		self.latent_size = latent_size
		self.num_layers = num_layers
		self.lstm = nn.LSTM(embed_dim, latent_size, num_layers, batch_first=True, dropout=0.2, bidirectional=False)

		# initialize weights
		#nn.init.xavier_uniform_(self.lstm.weight_ih_l0, gain=np.sqrt(2))
		#nn.init.xavier_uniform_(self.lstm.weight_hh_l0, gain=np.sqrt(2))
		#nn.init.orthogonal_(self.lstm.weight_ih_l0, gain=np.sqrt(2))
		#nn.init.orthogonal_(self.lstm.weight_hh_l0, gain=np.sqrt(2))

	def __call__(self, x):

		# forward propagate lstm
		h, _ = self.lstm(x) 
		return h[:, -1, :]

class RNNDecoder(nn.Module):
	def __init__(self, sequence_len, embed_dim, num_layers, latent_size, bidirectional):
		super(RNNDecoder, self).__init__()
		self.embed_dim = embed_dim
		self.num_layers = num_layers
		self.sequence_len = sequence_len
		self.lstm = nn.LSTM(latent_size, embed_dim, num_layers, batch_first=True, dropout=0.2, bidirectional=False)

		# initialize weights
		#nn.init.xavier_uniform_(self.lstm.weight_ih_l0, gain=np.sqrt(2))
		#nn.init.xavier_uniform_(self.lstm.weight_hh_l0, gain=np.sqrt(2))
		#nn.init.orthogonal_(self.lstm.weight_ih_l0, gain=np.sqrt(2))
		#nn.init.orthogonal_(self.lstm.weight_hh_l0, gain=np.sqrt(2))

	def __call__(self, h):

		# forward propagate lstm
		x_hat, _ = self.lstm(h.unsqueeze(dim=1).expand(-1, self.sequence_len, -1))
		return x_hat

class ConvolutionEncoder(nn.Module):
	def __init__(self, embedding_dim, t3, filter_size, filter_shape, latent_size):
		super(ConvolutionEncoder, self).__init__()
		self.convs1 = nn.Sequential(
				nn.Conv2d(1, filter_size, (filter_shape, embedding_dim), stride=(2,1)),
				nn.BatchNorm2d(filter_size),
				nn.SELU()
			)

		self.convs2 = nn.Sequential(
				nn.Conv2d(filter_size, filter_size * 2, (filter_shape, 1), stride=(2,1)),
				nn.BatchNorm2d(filter_size * 2),
				nn.SELU()
			)

		self.convs3 = nn.Sequential(
				nn.Conv2d(filter_size * 2, latent_size, (t3, 1), stride=(1,1)),
				nn.Tanh()
			)

		# weight initialize for conv layer
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.xavier_uniform_(m.weight)
				if m.bias is not None:
					torch.nn.init.constant_(m.bias, 0.001)

	def __call__(self, x):
		# x.size() is (L, emb_dim) if batch_size is 1.
		# So interpolate x's dimension if batch_size is 1.
		if len(x.size()) < 3:
			x = x.view(1, *x.size())
		# reshape for convolution layer
		x = x.view(x.size()[0], 1, x.size()[1], x.size()[2])
		# x = F.relu(x)
		h1 = self.convs1(x)
		h2 = self.convs2(h1)
		h = self.convs3(h2).squeeze().squeeze()
		return h

class DeconvolutionDecoder(nn.Module):
	def __init__(self, embedding_dim, t3, filter_size, filter_shape, latent_size):
		super(DeconvolutionDecoder, self).__init__()
		self.deconvs1 = nn.Sequential(
				nn.ConvTranspose2d(latent_size, filter_size * 2, (t3, 1), stride=(1,1)),
				nn.BatchNorm2d(filter_size * 2),
				nn.SELU()
			)
		self.deconvs2 = nn.Sequential(
				nn.ConvTranspose2d(filter_size * 2, filter_size, (filter_shape, 1), stride=(2,1)),
				nn.BatchNorm2d(filter_size),
				nn.SELU()
			)
		self.deconvs3 = nn.Sequential(
				nn.ConvTranspose2d(filter_size, 1, (filter_shape, embedding_dim), stride=(2,1)),
				nn.Tanh()
			)

		# weight initialize for conv_transpose layer
		for m in self.modules():
			if isinstance(m, nn.ConvTranspose2d):      
				nn.init.xavier_uniform_(m.weight)
				if m.bias is not None:
					torch.nn.init.constant_(m.bias, 0.001)

	def __call__(self, h):
		h2 = self.deconvs1(h.unsqueeze(dim=-1).unsqueeze(dim=-1))
		h1 = self.deconvs2(h2)
		x_hat = self.deconvs3(h1).squeeze()
		
		return x_hat

class ImgseqAutoEncoder(nn.Module):
	def __init__(self, encoder, decoder):
		super(ImgseqAutoEncoder, self).__init__()
		self.encoder = encoder
		self.decoder = decoder

	def forward(self, x):
		h = self.encoder(x)
		x_hat = self.decoder(h)
		return x_hat