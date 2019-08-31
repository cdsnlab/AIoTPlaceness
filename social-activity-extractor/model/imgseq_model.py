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
		self.lstm = nn.LSTM(embed_dim, int(latent_size/2), num_layers, batch_first=True, dropout=0.2, bidirectional=bidirectional)

		# initialize weights
		#nn.init.xavier_uniform_(self.lstm.weight_ih_l0, gain=np.sqrt(2))
		#nn.init.xavier_uniform_(self.lstm.weight_hh_l0, gain=np.sqrt(2))
		#nn.init.orthogonal_(self.lstm.weight_ih_l0, gain=np.sqrt(2))
		#nn.init.orthogonal_(self.lstm.weight_hh_l0, gain=np.sqrt(2))

	def __call__(self, x):

		# forward propagate lstm
		h, _ = self.lstm(x) 
		return h[:, -1, :].unsqueeze(1)

class RNNDecoder(nn.Module):
	def __init__(self, embed_dim, num_layers, latent_size, bidirectional):
		super(RNNDecoder, self).__init__()
		self.embed_dim = embed_dim
		self.num_layers = num_layers
		self.lstm = nn.LSTM(latent_size, int(embed_dim/2), num_layers, batch_first=True, dropout=0.2, bidirectional=bidirectional)

		# initialize weights
		#nn.init.xavier_uniform_(self.lstm.weight_ih_l0, gain=np.sqrt(2))
		#nn.init.xavier_uniform_(self.lstm.weight_hh_l0, gain=np.sqrt(2))
		#nn.init.orthogonal_(self.lstm.weight_ih_l0, gain=np.sqrt(2))
		#nn.init.orthogonal_(self.lstm.weight_hh_l0, gain=np.sqrt(2))

	def __call__(self, h):

		# forward propagate lstm
		x_hat, _ = self.lstm(h)
		normalized_x_hat = F.normalize(x_hat, p=2, dim=2)
		return normalized_x_hat

class ImgseqAutoEncoder(nn.Module):
	def __init__(self, encoder, decoder, sequence_len):
		super(ImgseqAutoEncoder, self).__init__()
		self.encoder = encoder
		self.decoder = decoder
		self.sequence_len = sequence_len

	def forward(self, x):
		h = self.encoder(x).expand(-1, self.sequence_len, -1)
		x_hat = self.decoder(h)
		return x_hat