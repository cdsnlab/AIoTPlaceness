import argparse
import config
import requests
import json
import pickle
import datetime
import os
import math
import numpy as np
import pandas as pd
from sumeval.metrics.rouge import RougeCalculator
from sumeval.metrics.bleu import BLEUCalculator
from gensim.models.keyedvectors import FastTextKeyedVectors
from gensim.similarities.index import AnnoyIndexer
from hyperdash import Experiment
from tqdm import tqdm


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import StepLR, CyclicLR
from model import util
from model import imgseq_model
from model.component import AdamW, Identity
from model.util import load_imgseq_data


CONFIG = config.Config

def slacknoti(contentstr):
	webhook_url = "https://hooks.slack.com/services/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"
	payload = {"text": contentstr}
	requests.post(webhook_url, data=json.dumps(payload), headers={'Content-Type': 'application/json'})

def main():
	parser = argparse.ArgumentParser(description='text convolution-deconvolution auto-encoder model')
	# learning
	parser.add_argument('-lr', type=float, default=3e-04, help='initial learning rate')
	parser.add_argument('-weight_decay', type=float, default=3e-05, help='initial weight decay')
	parser.add_argument('-epochs', type=int, default=80, help='number of epochs for train')
	parser.add_argument('-batch_size', type=int, default=16, help='batch size for training')
	parser.add_argument('-lr_decay_interval', type=int, default=10,
						help='how many epochs to wait before decrease learning rate')
	parser.add_argument('-log_interval', type=int, default=1000,
						help='how many steps to wait before logging training status')
	parser.add_argument('-test_interval', type=int, default=1,
						help='how many epochs to wait before testing')
	parser.add_argument('-save_interval', type=int, default=1,
						help='how many epochs to wait before saving')
	# data
	parser.add_argument('-target_dataset', type=str, default=None, help='folder name of target dataset')
	parser.add_argument('-shuffle', default=True, help='shuffle data every epoch')
	parser.add_argument('-split_rate', type=float, default=0.9, help='split rate between train and validation')
	# model
	parser.add_argument('-arch', type=str, default='resnet152', help='image embedding model')
	parser.add_argument('-latent_size', type=int, default=900, help='size of latent variable')
	parser.add_argument('-num_layer', type=int, default=4, help='layer number')

	# train
	parser.add_argument('-noti', action='store_true', default=False, help='whether using gpu server')
	parser.add_argument('-gpu', type=str, default='cuda', help='gpu number')
	# option
	parser.add_argument('-resume', type=str, default=None, help='filename of checkpoint to resume ')

	args = parser.parse_args()

	if args.noti:
		slacknoti("underkoo start using")
	train_reconstruction(args)
	if args.noti:
		slacknoti("underkoo end using")


def train_reconstruction(args):
	device = torch.device(args.gpu)
	print("Loading embedding model...")
	embedding_model = models.__dict__[args.arch](pretrained=True)
	embedding_dim = embedding_model.fc.in_features
	args.embedding_dim = embedding_dim
	print("Loading dataset...")
	train_dataset, val_dataset = load_imgseq_data(args, CONFIG)
	print("Loading dataset completed")
	train_loader, val_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=args.shuffle),\
								  DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

	imgseq_encoder = imgseq_model.RNNEncoder(embedding_dim, args.num_layer, args.latent_size, bidirectional=True)
	imgseq_decoder = imgseq_model.RNNDecoder(embedding_dim, args.num_layer, args.latent_size, bidirectional=True)
	if args.resume:
		print("Restart from checkpoint")
		checkpoint = torch.load(os.path.join(CONFIG.CHECKPOINT_PATH, args.resume), map_location=lambda storage, loc: storage)
		best_loss = checkpoint['best_loss']
		start_epoch = checkpoint['epoch']
		imageseq_encoder.load_state_dict(checkpoint['imgseq_encoder'])
		imageseq_decoder.load_state_dict(checkpoint['imgseq_decoder'])
	else:		
		print("Start from initial")
		best_loss = 999999.
		start_epoch = 0
	
	imgseq_autoencoder = imgseq_model.ImgseqAutoEncoder(imgseq_encoder, imgseq_decoder, CONFIG.MAX_SEQUENCE_LEN)
	criterion = nn.MSELoss().to(device)
	imgseq_autoencoder.to(device)

	optimizer = AdamW(imgseq_autoencoder.parameters(), lr=args.lr, weight_decay=args.weight_decay, amsgrad=True)

	if args.resume:
		optimizer.load_state_dict(checkpoint['optimizer'])


	exp = Experiment("Image-sequence autoencoder")
	try:
		avg_loss = []

		imgseq_autoencoder.train() 

		for epoch in range(start_epoch, args.epochs):
			print("Epoch: {}".format(epoch))
			for steps, batch in enumerate(train_loader):
				torch.cuda.empty_cache()
				feature = Variable(batch).to(device)
				optimizer.zero_grad()
				feature_hat = imgseq_autoencoder(feature)
				loss = criterion(feature_hat, feature)
				loss.backward()
				optimizer.step()

				if steps % args.log_interval == 0:
					print("Epoch: {} at {}".format(epoch, str(datetime.datetime.now())))
					print("Steps: {}".format(steps))
					print("Loss: {}".format(loss.detach().item()))
					exp.metric("Loss", loss.detach().item())
					input_data = feature[0]
				del feature, feature_hat, loss
			
			_avg_loss = eval_reconstruction(imgseq_autoencoder, embedding_model, criterion, val_loader, args, device)
			avg_loss.append(_avg_loss)

			if best_loss > _avg_loss:
				best_loss = _avg_loss
				util.save_models({
					'epoch': epoch + 1,
					'imgseq_encoder': imgseq_encoder.state_dict(),
					'imgseq_decoder': imgseq_decoder.state_dict(),
					'best_loss': best_loss,
					'optimizer' : optimizer.state_dict(),
				}, CONFIG.CHECKPOINT_PATH, "imgseq_autoencoder")

		eval_reconstruction(imgseq_autoencoder, embedding_model, criterion, val_loader, args, device)		
		print("Finish!!!")

	finally:
		exp.end()

def eval_reconstruction(autoencoder, embedding_model, criterion, data_iter, args, device):
	print("=================Eval======================")
	autoencoder.eval()
	step = 0
	avg_loss = 0.
	rouge_1 = 0.
	rouge_2 = 0.
	for batch in data_iter:
		torch.cuda.empty_cache()
		with torch.no_grad():
			feature = Variable(batch).to(device)
		feature_hat = autoencoder(feature)
		loss = criterion(feature_hat, feature)	
		avg_loss += loss.detach().item()
		step = step + 1
		del feature, feature_hat, loss
	avg_loss = avg_loss / step
	print("Evaluation - loss: {}".format(avg_loss))
	print("===============================================================")
	autoencoder.train()

	return avg_loss



if __name__ == '__main__':
	main()