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
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import StepLR, CyclicLR
from torchvision.utils import save_image
from model import util
from model import imgseq_model
from model.resnet_autoencoder import ResNet50Encoder, ResNet50Decoder, ResNet_autoencoder
from model.component import AdamW, cyclical_lr, ImageEncoder, ImageDecoder, ImageAutoEncoder
from model.util import load_image_pretrain_data, transform_inverse_normalize


CONFIG = config.Config

def slacknoti(contentstr):
	webhook_url = "https://hooks.slack.com/services/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"
	payload = {"text": contentstr}
	requests.post(webhook_url, data=json.dumps(payload), headers={'Content-Type': 'application/json'})

def main():
	parser = argparse.ArgumentParser(description='text convolution-deconvolution auto-encoder model')
	# learning
	parser.add_argument('-lr', type=float, default=1e-04, help='initial learning rate')
	parser.add_argument('-lr_factor', type=float, default=10, help='lr_factor for min lr')
	parser.add_argument('-half_cycle_interval', type=int, default=4, help='lr_factor step size equals to half cycle')
	parser.add_argument('-weight_decay', type=float, default=1e-05, help='initial weight decay')
	parser.add_argument('-epochs', type=int, default=100, help='number of epochs for train')
	parser.add_argument('-batch_size', type=int, default=16, help='batch size for training')
	parser.add_argument('-log_interval', type=int, default=25600,
						help='how many steps to wait before logging training status')
	parser.add_argument('-save_interval', type=int, default=1,
						help='how many epochs to wait before saving')
	# data
	parser.add_argument('-target_dataset', type=str, default=None, help='folder name of target dataset')
	parser.add_argument('-shuffle', default=True, help='shuffle data every epoch')
	parser.add_argument('-split_rate', type=float, default=0.8, help='split rate between train and validation')
	# model
	parser.add_argument('-embedding_dim', type=int, default=2048, help='embedding dimension of the model')
	parser.add_argument('-latent_size', type=int, default=1000, help='size of latent variable')

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
	print("Loading dataset...")
	train_dataset, val_dataset = load_image_pretrain_data(args, CONFIG)
	print("Loading dataset completed")
	train_loader, val_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=args.shuffle),\
								  DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

	# image_encoder = ImageEncoder()
	# image_encoder.init_weights()
	# image_decoder = ImageDecoder()
	image_encoder = ResNet50Encoder()
	image_encoder.init_weights()
	image_decoder = ResNet50Decoder()
	if args.resume:
		print("Restart from checkpoint")
		checkpoint = torch.load(os.path.join(CONFIG.CHECKPOINT_PATH, args.resume), map_location=lambda storage, loc: storage)
		start_epoch = checkpoint['epoch']
		image_encoder.load_state_dict(checkpoint['image_encoder'])
		image_decoder.load_state_dict(checkpoint['image_decoder'])
	else:		
		print("Start from initial")
		start_epoch = 0
	
	image_autoencoder = ResNet_autoencoder(image_encoder, image_decoder)
	criterion = nn.MSELoss().to(device)
	image_autoencoder.to(device)

	optimizer = AdamW(image_autoencoder.parameters(), lr=1., weight_decay=args.weight_decay, amsgrad=True)
	step_size = args.half_cycle_interval*len(train_loader)
	clr = cyclical_lr(step_size, min_lr=args.lr, max_lr=args.lr*args.lr_factor)
	scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, [clr])

	if args.resume:
		optimizer.load_state_dict(checkpoint['optimizer'])
		scheduler.load_state_dict(checkpoint['scheduler'])


	exp = Experiment("Image-sequence Component Pretrain " + str(args.latent_size), capture_io=False)

	for arg, value in vars(args).items():
		exp.param(arg, value) 
	try:
		image_autoencoder.train() 

		for epoch in range(start_epoch, args.epochs):
			print("Epoch: {}".format(epoch))
			for steps, batch in enumerate(train_loader):
				torch.cuda.empty_cache()
				feature = Variable(batch).to(device)
				optimizer.zero_grad()
				feature_hat = image_autoencoder(feature)
				loss = criterion(feature_hat, feature)
				loss.backward()
				optimizer.step()
				scheduler.step()

				if (steps * args.batch_size) % args.log_interval == 0:
					print("Epoch: {} at {} lr: {}".format(epoch, str(datetime.datetime.now()), str(scheduler.get_lr())))
					print("Steps: {}".format(steps))
					print("Loss: {}".format(loss.detach().item()))
				del feature, feature_hat, loss
			
			exp.log("\nEpoch: {} at {} lr: {}".format(epoch, str(datetime.datetime.now()), str(scheduler.get_lr())))
			_avg_loss = eval_reconstruction(image_autoencoder, criterion, val_loader, device, epoch)
			exp.log("\nEvaluation - loss: {}".format(_avg_loss))

			util.save_models({
				'epoch': epoch + 1,
				'image_encoder': image_encoder.state_dict(),
				'image_decoder': image_decoder.state_dict(),
				'avg_loss': _avg_loss,
				'optimizer' : optimizer.state_dict(),
				'scheduler' : scheduler.state_dict()
			}, CONFIG.CHECKPOINT_PATH, "image_pretrain" + str(args.latent_size))
	
		print("Finish!!!")

	finally:
		exp.end()

def eval_reconstruction(autoencoder,criterion, data_iter, device, epoch):
	print("=================Eval======================")
	autoencoder.eval()
	step = 0
	avg_loss = 0.
	rouge_1 = 0.
	rouge_2 = 0.
	for batch in tqdm(data_iter):
		torch.cuda.empty_cache()
		with torch.no_grad():
			feature = Variable(batch).to(device)
		feature_hat = autoencoder(feature)
		loss = criterion(feature_hat, feature)	
		avg_loss += loss.detach().item()
		if step == 0:
			input_data = transform_inverse_normalize(feature[0])
			output_data = transform_inverse_normalize(feature_hat[0])
			save_image([input_data, output_data], './evaluation/pretrain/' + str(epoch) +'.png', nrow=1)
			del input_data, output_data
		step = step + 1
		del feature, feature_hat, loss
	avg_loss = avg_loss / step
	print("Evaluation - loss: {}".format(avg_loss))
	print("===============================================================")
	autoencoder.train()

	return avg_loss



if __name__ == '__main__':
	main()