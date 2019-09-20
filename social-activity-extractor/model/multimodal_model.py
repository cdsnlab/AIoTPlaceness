import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init
from torch.autograd import Variable

import math
import numpy as np

from model.component import SiLU, Maxout, PTanh

class MultimodalEncoder(nn.Module):
	def __init__(self, text_encoder, imgseq_encoder, latent_size, normalize=False, add_latent=False):
		super(MultimodalEncoder, self).__init__()
		self.text_encoder = text_encoder
		self.imgseq_encoder = imgseq_encoder
		self.latent_size = latent_size
		self.multimodal_encoder = nn.Sequential(
			nn.Linear(latent_size*2, int(latent_size*2/3)),
			nn.SELU(),
			nn.Linear(int(latent_size*2/3), latent_size),
			nn.Tanh())
		self.normalize = normalize
		self.add_latent = add_latent
	def __call__(self, text, imgseq):
		text_h = self.text_encoder(text)
		imgseq_h = self.imgseq_encoder(imgseq)

		# if batch_size == 1, unsqueeze
		if len(text_h.size()) < 2:
			text_h = text_h.view(1, *text_h.size())
		if len(imgseq_h.size()) < 2:
			imgseq_h = imgseq_h.view(1, *imgseq_h.size())

		if self.normalize:
			text_h = F.normalize(text_h, p=2, dim=1)
			imgseq_h = F.normalize(imgseq_h, p=2, dim=1)
		if self.add_latent:
			h = text_h + imgseq_h
			h = self.multimodal_encoder(h)
		else:
			h = self.multimodal_encoder(torch.cat((text_h, imgseq_h), dim=-1))
		return h

class MultimodalDecoder(nn.Module):
	def __init__(self, text_decoder, imgseq_decoder, latent_size, sequence_len, no_decode=False):
		super(MultimodalDecoder, self).__init__()
		self.text_decoder = text_decoder
		self.imgseq_decoder =imgseq_decoder
		self.sequence_len = sequence_len
		self.latent_size = latent_size
		self.multimodal_decoder = nn.Sequential(
			nn.Linear(latent_size, int(latent_size*2/3)),
			nn.SELU(),
			nn.Linear(int(latent_size*2/3), latent_size*2),
			nn.Tanh())
		self.no_decode = no_decode
	def __call__(self, h):
		if self.no_decode:
			text_hat = self.text_decoder(h)
			imgseq_hat = self.imgseq_decoder(h)
		else:
			decode_h = torch.split(self.multimodal_decoder(h), self.latent_size, dim=-1)
			text_hat = self.text_decoder(decode_h[0])
			imgseq_hat = self.imgseq_decoder(decode_h[1])
		return text_hat, imgseq_hat

class MultimodalAutoEncoder(nn.Module):
	def __init__(self, encoder, decoder):
		super(MultimodalAutoEncoder, self).__init__()
		self.encoder = encoder
		self.decoder = decoder

	def forward(self, text, imgseq):
		h = self.encoder(text, imgseq)
		text_hat, imgseq_hat = self.decoder(h)
		return text_hat, imgseq_hat