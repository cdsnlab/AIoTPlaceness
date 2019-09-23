import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init
from torch.autograd import Variable

import math
import numpy as np

from model.component import SiLU, Maxout, PTanh

class CRNNEncoder(nn.Module):
	def __init__(self, encoder_list, embed_dim, num_layers, latent_size, bidirectional):
		super(RNNEncoder, self).__init__()
		self.latent_size = latent_size
		self.num_layers = num_layers
		self.lstm = nn.LSTM(embed_dim, int(latent_size/2), num_layers, batch_first=True, dropout=0.2, bidirectional=bidirectional)

		# initialize weights
		#nn.init.xavier_uniform_(self.lstm.weight_ih_l0, gain=np.sqrt(2))
		#nn.init.xavier_uniform_(self.lstm.weight_hh_l0, gain=np.sqrt(2))
		#nn.init.orthogonal_(self.lstm.weight_ih_l0, gain=np.sqrt(2))
		#nn.init.orthogonal_(self.lstm.weight_hh_l0, gain=np.sqrt(2))

	def __call__(self, x):

		# forward propagate lstm
		h, _ = self.lstm(x) 
		return h[:, -1, :]

class CRNNDecoder(nn.Module):
	def __init__(self, decoder_list, sequence_len, embed_dim, num_layers, latent_size, bidirectional):
		super(RNNDecoder, self).__init__()
		self.embed_dim = embed_dim
		self.num_layers = num_layers
		self.sequence_len = sequence_len
		self.lstm = nn.LSTM(latent_size, int(embed_dim/2), num_layers, batch_first=True, dropout=0.2, bidirectional=bidirectional)

		# initialize weights
		#nn.init.xavier_uniform_(self.lstm.weight_ih_l0, gain=np.sqrt(2))
		#nn.init.xavier_uniform_(self.lstm.weight_hh_l0, gain=np.sqrt(2))
		#nn.init.orthogonal_(self.lstm.weight_ih_l0, gain=np.sqrt(2))
		#nn.init.orthogonal_(self.lstm.weight_hh_l0, gain=np.sqrt(2))

	def __call__(self, h):

		# forward propagate lstm
		x_hat, _ = self.lstm(h.unsqueeze(dim=1).expand(-1, self.sequence_len, -1))
		return x_hat

class CRNNAutoEncoder(nn.Module):
	def __init__(self, encoder, decoder):
		super(ImgseqAutoEncoder, self).__init__()
		self.encoder = encoder
		self.decoder = decoder

	def forward(self, x):
		h = self.encoder(x)
		x_hat = self.decoder(h)
		return x_hat

class ResNetDecoder(nn.Module):
	def __init__(self):
		super(Decoder,self).__init__()
		self.dfc3 = nn.Linear(zsize, 4096)
		self.bn3 = nn.BatchNorm2d(4096)
		self.dfc2 = nn.Linear(4096, 4096)
		self.bn2 = nn.BatchNorm2d(4096)
		self.dfc1 = nn.Linear(4096,256 * 6 * 6)
		self.bn1 = nn.BatchNorm2d(256*6*6)
		self.upsample1=nn.Upsample(scale_factor=2)
		self.dconv5 = nn.ConvTranspose2d(256, 256, 3, padding = 0)
		self.dconv4 = nn.ConvTranspose2d(256, 384, 3, padding = 1)
		self.dconv3 = nn.ConvTranspose2d(384, 192, 3, padding = 1)
		self.dconv2 = nn.ConvTranspose2d(192, 64, 5, padding = 2)
		self.dconv1 = nn.ConvTranspose2d(64, 3, 12, stride = 4, padding = 4)

		self.relu = nn.ReLU(inplace=True)
		self.tanh = nn.Sigmoid(inplace=True)

	def forward(self,x):
		
		x = self.dfc3(x)
		x = self.relu(self.bn3(x))		
		x = self.dfc2(x)
		x = self.relu(self.bn2(x))
		x = self.dfc1(x)
		x = self.relu(self.bn1(x))
		x = self.view(batch_size,256,6,6)
		x = self.upsample1(x)
		x = self.dconv5(x)
		x = self.relu(x)
		x = self.relu(self.dconv4(x))
		x = self.relu(self.dconv3(x))
		x = self.upsample1(x)
		x = self.dconv2(x)
		x = self.relu(x)
		x = self.upsample1(x)
		x = self.dconv1(x)
		x = self.tanh(x)
		return x