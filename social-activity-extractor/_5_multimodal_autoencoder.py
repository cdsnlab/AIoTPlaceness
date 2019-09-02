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
from model import text_model, imgseq_model, multimodal_model
from model.util import load_multimodal_data
from model.component import AdamW



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
	parser.add_argument('-epochs', type=int, default=40, help='number of epochs for train')
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
	parser.add_argument('-filter_size', type=int, default=300, help='filter size of convolution')
	parser.add_argument('-filter_shape', type=int, default=5,
						help='filter shape to use for convolution')
	parser.add_argument('-num_layer', type=int, default=4, help='layer number')
	parser.add_argument('-text_pt', type=str, default=None, help='filename of checkpoint of text autoencoder')
	parser.add_argument('-imgseq_pt', type=str, default=None, help='filename of checkpoint of image sequence autoencoder')

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
	image_embedding_model = models.__dict__[args.arch](pretrained=True)
	image_embedding_dim = image_embedding_model.fc.in_features
	args.image_embedding_dim = image_embedding_dim
	model_name = 'FASTTEXT_' + args.target_dataset + '.model'
	text_embedding_model = FastTextKeyedVectors.load(os.path.join(CONFIG.EMBEDDING_PATH, model_name))
	text_embedding_dim = text_embedding_model.vector_size	
	args.text_embedding_dim = text_embedding_dim
	print("Building index...")
	indexer = AnnoyIndexer(text_embedding_model, 10)
	print("Loading embedding model completed")
	print("Loading dataset...")
	train_dataset, val_dataset = load_multimodal_data(args, CONFIG, text_embedding_model)
	print("Loading dataset completed")
	train_loader, val_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=args.shuffle),\
								  DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

	# t1 = max_sentence_len + 2 * (args.filter_shape - 1)
	t1 = CONFIG.MAX_SENTENCE_LEN
	t2 = int(math.floor((t1 - args.filter_shape) / 2) + 1) # "2" means stride size
	t3 = int(math.floor((t2 - args.filter_shape) / 2) + 1)
	args.t3 = t3	

	text_encoder = text_model.ConvolutionEncoder(text_embedding_dim, t3, args.filter_size, args.filter_shape, args.latent_size)
	text_decoder = text_model.DeconvolutionDecoder(text_embedding_dim, t3, args.filter_size, args.filter_shape, args.latent_size)
	if args.text_pt:
		text_checkpoint = torch.load(os.path.join(CONFIG.CHECKPOINT_PATH, args.text_pt), map_location=lambda storage, loc: storage)
		text_encoder.load_state_dict(text_checkpoint['text_encoder'])
		text_decoder.load_state_dict(text_checkpoint['text_decoder'])
	imgseq_encoder = imgseq_model.RNNEncoder(image_embedding_dim, args.num_layer, args.latent_size, bidirectional=True)
	imgseq_decoder = imgseq_model.RNNDecoder(image_embedding_dim, args.num_layer, args.latent_size, bidirectional=True)
	if args.imgseq_pt:
		imgseq_checkpoint = torch.load(os.path.join(CONFIG.CHECKPOINT_PATH, args.imgseq_pt), map_location=lambda storage, loc: storage)
		imgseq_encoder.load_state_dict(imgseq_checkpoint['imgseq_encoder'])
		imgseq_decoder.load_state_dict(imgseq_checkpoint['imgseq_decoder'])
	multimodal_encoder = multimodal_model.MultimodalEncoder(text_encoder, imgseq_encoder, args.latent_size)
	multimodal_decoder = multimodal_model.MultimodalDecoder(text_decoder, imgseq_decoder, args.latent_size, CONFIG.MAX_SEQUENCE_LEN)
	if args.resume:
		print("Restart from checkpoint")
		checkpoint = torch.load(os.path.join(CONFIG.CHECKPOINT_PATH, args.resume), map_location=lambda storage, loc: storage)
		best_loss = checkpoint['best_loss']
		start_epoch = checkpoint['epoch']
		multimodal_encoder.load_state_dict(checkpoint['multimodal_encoder'])
		multimodal_decoder.load_state_dict(checkpoint['multimodal_decoder'])
	else:		
		print("Start from initial")
		best_loss = 999999.
		start_epoch = 0
	multimodal_autoencoder = multimodal_model.MultimodalAutoEncoder(multimodal_encoder, multimodal_decoder)
	criterion = nn.MSELoss().to(device)
	multimodal_autoencoder.to(device)

	optimizer = AdamW(multimodal_autoencoder.parameters(), lr=args.lr, weight_decay=args.weight_decay, amsgrad=True)

	if args.resume:
		optimizer.load_state_dict(checkpoint['optimizer'])


	exp = Experiment("Multimodal autoencoder")
	try:
		avg_loss = []

		multimodal_autoencoder.train() 

		for epoch in range(start_epoch, args.epochs):
			print("Epoch: {}".format(epoch))
			for steps, (text_batch, imgseq_batch) in enumerate(train_loader):
				torch.cuda.empty_cache()
				text_feature = Variable(text_batch).to(device)
				imgseq_feature = Variable(imgseq_batch).to(device)
				feature = torch.cat((text_feature.view(text_feature.size()[0], -1), imgseq_feature.view(imgseq_feature.size()[0], -1)), dim=1)
				optimizer.zero_grad()
				text_feature_hat, imgseq_feature_hat = multimodal_autoencoder(text_feature, imgseq_feature)
				feature_hat = torch.cat((text_feature_hat.contiguous().view(text_feature.size()[0], -1), imgseq_feature_hat.contiguous().view(imgseq_feature.size()[0], -1)), dim=1)
				loss = criterion(feature_hat, feature)
				loss.backward()
				optimizer.step()

				if steps % args.log_interval == 0:					
					input_data = text_feature[0]
					single_data = text_feature_hat[0]
					input_sentence = util.transform_vec2sentence(input_data.detach().cpu().numpy(), text_embedding_model, indexer)
					predict_sentence = util.transform_vec2sentence(single_data.detach().cpu().numpy(), text_embedding_model, indexer)
					print("Epoch: {} at {}".format(epoch, str(datetime.datetime.now())))
					print("Steps: {}".format(steps))
					print("Loss: {}".format(loss.detach().item()))
					exp.metric("Loss", loss.detach().item())
					print("Input Sentence:")
					print(input_sentence)
					print("Output Sentence:")
					print(predict_sentence)
					del input_data, single_data
				del text_feature, text_feature_hat, imgseq_feature, imgseq_feature_hat, feature, feature_hat, loss
			
			_avg_loss = eval_reconstruction(multimodal_autoencoder, text_embedding_model, indexer, criterion, val_loader, args, device)

			if best_loss > _avg_loss:
				best_loss = _avg_loss
				util.save_models({
					'epoch': epoch + 1,
					'multimodal_encoder': multimodal_encoder.state_dict(),
					'multimodal_decoder': multimodal_decoder.state_dict(),
					'best_loss': best_loss,
					'optimizer' : optimizer.state_dict(),
				}, CONFIG.CHECKPOINT_PATH, "multimodal_autoencoder")

		eval_reconstruction_with_rouge(multimodal_autoencoder, text_embedding_model, indexer, criterion, val_loader, args, device)
		print("Finish!!!")

	finally:
		exp.end()

def eval_reconstruction(autoencoder, embedding_model, indexer, criterion, data_iter, args, device):
	print("=================Eval======================")
	autoencoder.eval()
	step = 0
	avg_loss = 0.
	rouge_1 = 0.
	rouge_2 = 0.
	for batch in data_iter:
		torch.cuda.empty_cache()
		with torch.no_grad():			
			text_feature = Variable(text_batch).to(device)
			imgseq_feature = Variable(imgseq_batch).to(device)
			feature = torch.cat((text_feature.view(text_feature.size()[0], -1), imgseq_feature.view(imgseq_feature.size()[0], -1)), dim=1)
		text_feature_hat, imgseq_feature_hat = autoencoder(text_feature, imgseq_feature)
		feature_hat = torch.cat((text_feature_hat.contiguous().view(text_feature.size()[0], -1), imgseq_feature_hat.contiguous().view(imgseq_feature.size()[0], -1)), dim=1)
		loss = criterion(feature_hat, feature)	
		avg_loss += loss.detach().item()
		step = step + 1
		del text_feature, text_feature_hat, imgseq_feature, imgseq_feature_hat, feature, feature_hat, loss
	avg_loss = avg_loss / step
	print("Evaluation - loss: {}".format(avg_loss))
	print("===============================================================")
	autoencoder.train()

	return avg_loss

def eval_reconstruction_with_rouge(autoencoder, embedding_model, indexer, criterion, data_iter, args, device):
	print("=================Eval======================")
	autoencoder.eval()
	step = 0
	avg_loss = 0.
	rouge_1 = 0.
	rouge_2 = 0.
	for batch in data_iter:
		torch.cuda.empty_cache()
		with torch.no_grad():
			text_feature = Variable(text_batch).to(device)
			imgseq_feature = Variable(imgseq_batch).to(device)
			feature = torch.cat((text_feature.view(text_feature.size()[0], -1), imgseq_feature.view(imgseq_feature.size()[0], -1)), dim=1)
		feature_hat = autoencoder(feature)
		feature_hat = torch.cat((text_feature_hat.contiguous().view(text_feature.size()[0], -1), imgseq_feature_hat.contiguous().view(imgseq_feature.size()[0], -1)), dim=1)
		original_sentences = [util.transform_vec2sentence(sentence, embedding_model, indexer) for sentence in text_feature.detach().cpu().numpy()]		
		predict_sentences = [util.transform_vec2sentence(sentence, embedding_model, indexer) for sentence in text_feature_hat.detach().cpu().numpy()]	
		r1, r2 = calc_rouge(original_sentences, predict_sentences)		
		rouge_1 += r1 / len(batch)
		rouge_2 += r2 / len(batch)
		loss = criterion(feature_hat, feature)	
		avg_loss += loss.detach().item()
		step = step + 1
		del text_feature, text_feature_hat, imgseq_feature, imgseq_feature_hat, feature, feature_hat, loss
	avg_loss = avg_loss / step
	rouge_1 = rouge_1 / step
	rouge_2 = rouge_2 / step
	print("Evaluation - loss: {}  Rouge1: {}    Rouge2: {}".format(avg_loss, rouge_1, rouge_2))
	print("===============================================================")
	autoencoder.train()

	return avg_loss, rouge_1, rouge_2

def calc_rouge(original_sentences, predict_sentences):
	rouge_1 = 0.0
	rouge_2 = 0.0
	for original, predict in zip(original_sentences, predict_sentences):
		# Remove padding
		original, predict = original.replace("<PAD>", "").strip(), predict.replace("<PAD>", "").strip()
		rouge = RougeCalculator(stopwords=True, lang="en")
		r1 = rouge.rouge_1(summary=predict, references=original)
		r2 = rouge.rouge_2(summary=predict, references=original)
		rouge_1 += r1
		rouge_2 += r2
	return rouge_1, rouge_2


if __name__ == '__main__':
	main()