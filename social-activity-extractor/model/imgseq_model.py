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

class ImgseqAutoEncoder(nn.Module):
	def __init__(self, encoder, decoder):
		super(ImgseqAutoEncoder, self).__init__()
		self.encoder = encoder
		self.decoder = decoder

	def forward(self, x):
		h = self.encoder(x)
		x_hat = self.decoder(h)
		return x_hat