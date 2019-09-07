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
import _pickle as cPickle
from sumeval.metrics.rouge import RougeCalculator
from sumeval.metrics.bleu import BLEUCalculator
from gensim.models.keyedvectors import FastTextKeyedVectors
from gensim.similarities.index import AnnoyIndexer
from hyperdash import Experiment
from tqdm import tqdm


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import StepLR, CyclicLR
from model import util
from model import text_model
from model.util import load_text_data
from model.component import AdamW, cyclical_lr



CONFIG = config.Config

def slacknoti(contentstr):
	webhook_url = "https://hooks.slack.com/services/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"
	payload = {"text": contentstr}
	requests.post(webhook_url, data=json.dumps(payload), headers={'Content-Type': 'application/json'})

def main():
	parser = argparse.ArgumentParser(description='text convolution-deconvolution auto-encoder model')
	# learning
	parser.add_argument('-lr', type=float, default=3e-03, help='initial learning rate')
	parser.add_argument('-weight_decay', type=float, default=1e-05, help='initial weight decay')
	parser.add_argument('-epochs', type=int, default=100, help='number of epochs for train')
	parser.add_argument('-batch_size', type=int, default=16, help='batch size for training')
	parser.add_argument('-lr_decay_interval', type=int, default=10,
						help='how many epochs to wait before decrease learning rate')
	parser.add_argument('-log_interval', type=int, default=25600,
						help='how many steps to wait before logging training status')
	parser.add_argument('-tau', type=float, default=0.01, help='temperature parameter')
	# data
	parser.add_argument('-target_dataset', type=str, default=None, help='folder name of target dataset')
	parser.add_argument('-shuffle', default=True, help='shuffle data every epoch')
	parser.add_argument('-split_rate', type=float, default=0.9, help='split rate between train and validation')
	# model
	parser.add_argument('-latent_size', type=int, default=900, help='size of latent variable')
	parser.add_argument('-filter_size', type=int, default=300, help='filter size of convolution')
	parser.add_argument('-filter_shape', type=int, default=5,
						help='filter shape to use for convolution')

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
	with open(os.path.join(CONFIG.DATASET_PATH, args.target_dataset, 'word_embedding.p'), "rb") as f:
		embedding_model = cPickle.load(f)
	with open(os.path.join(CONFIG.DATASET_PATH, args.target_dataset, 'word_idx.json'), "r", encoding='utf-8') as f:
		word_idx = json.load(f)
	print("Loading embedding model completed")
	print("Loading dataset...")
	train_dataset, val_dataset = load_text_data(args, CONFIG, word2idx=word_idx[1])
	print("Loading dataset completed")
	train_loader, val_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=args.shuffle),\
								  DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

	# t1 = max_sentence_len + 2 * (args.filter_shape - 1)
	t1 = CONFIG.MAX_SENTENCE_LEN
	t2 = int(math.floor((t1 - args.filter_shape) / 2) + 1) # "2" means stride size
	t3 = int(math.floor((t2 - args.filter_shape) / 2) + 1)
	args.t3 = t3
	embedding = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_model))
	text_encoder = text_model.ConvolutionEncoder(embedding, t3, args.filter_size, args.filter_shape, args.latent_size)
	text_decoder = text_model.DeconvolutionDecoder(embedding, args.tau, t3, args.filter_size, args.filter_shape, args.latent_size, device)
	if args.resume:
		print("Restart from checkpoint")
		checkpoint = torch.load(os.path.join(CONFIG.CHECKPOINT_PATH, args.resume), map_location=lambda storage, loc: storage)
		best_loss = checkpoint['best_loss']
		start_epoch = checkpoint['epoch']
		text_encoder.load_state_dict(checkpoint['text_encoder'])
		text_decoder.load_state_dict(checkpoint['text_decoder'])
	else:		
		print("Start from initial")
		best_loss = 999999.
		start_epoch = 0
	
	text_autoencoder = text_model.TextAutoencoder(text_encoder, text_decoder)
	criterion = nn.NLLLoss().to(device)
	text_autoencoder.to(device)

	optimizer = AdamW(text_autoencoder.parameters(), lr=args.lr, weight_decay=args.weight_decay, amsgrad=True)
	step_size = 4*len(train_loader)
	clr = cyclical_lr(step_size, min_lr=args.lr/6, max_lr=args.lr)
	scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, [clr])
	if args.resume:
		optimizer.load_state_dict(checkpoint['optimizer'])
	exp = Experiment("Text autoencoder", capture_io=False)

	for arg, value in vars(args).items():
		exp.param(arg, value) 
	try:
		avg_loss = []

		text_autoencoder.train() 

		for epoch in range(start_epoch, args.epochs):
			print("Epoch: {}".format(epoch))
			for steps, batch in enumerate(train_loader):
				torch.cuda.empty_cache()
				feature = Variable(batch).to(device)
				optimizer.zero_grad()
				prob = text_autoencoder(feature)
				loss = criterion(prob.transpose(1, 2), feature)
				loss.backward()
				scheduler.step()
				optimizer.step()

				if (steps * args.batch_size) % args.log_interval == 0:					
					input_data = feature[0]
					single_data = prob[0]
					_, predict_index = torch.max(single_data, 1)
					input_sentence = util.transform_idx2word(input_data.detach().cpu().numpy(), idx2word=word_idx[0])
					predict_sentence = util.transform_idx2word(predict_index.detach().cpu().numpy(), idx2word=word_idx[0])	
					print("Epoch: {} at {}".format(epoch, str(datetime.datetime.now())))
					print("Steps: {}".format(steps))
					print("Loss: {}".format(loss.detach().item()))
					print("Input Sentence:")
					print(input_sentence)
					print("Output Sentence:")
					print(predict_sentence)
					del input_data, single_data, _, predict_index
				del feature, prob, loss
			
			exp.log("\nEpoch: {} at {}".format(epoch, str(datetime.datetime.now())))
			_avg_loss, _rouge_1, _rouge_2 = eval_reconstruction_with_rouge(text_autoencoder, word_idx[0], criterion, val_loader, device)
			exp.log("\nEvaluation - loss: {}  Rouge1: {}    Rouge2: {}".format(_avg_loss, _rouge_1, _rouge_2))

			if best_loss > _avg_loss:
				best_loss = _avg_loss
				util.save_models({
					'epoch': epoch + 1,
					'text_encoder': text_encoder.state_dict(),
					'text_decoder': text_decoder.state_dict(),
					'best_loss': best_loss,
					'optimizer' : optimizer.state_dict(),
				}, CONFIG.CHECKPOINT_PATH, "text_autoencoder")
		print("Finish!!!")

	finally:
		exp.end()

def eval_reconstruction(autoencoder, criterion, data_iter, device):
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
		prob = autoencoder(feature)
		loss = criterion(prob.transpose(1, 2), feature)	
		avg_loss += loss.detach().item()
		step = step + 1
		del feature, prob, loss
	avg_loss = avg_loss / step
	print("===============================================================")
	autoencoder.train()

	return avg_loss

def eval_reconstruction_with_rouge(autoencoder, idx2word, criterion, data_iter, device):
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
		prob = autoencoder(feature)
		_, predict_index = torch.max(prob, 2)
		original_sentences = [util.transform_idx2word(sentence, idx2word=idx2word) for sentence in feature.detach().cpu().numpy()]		
		predict_sentences = [util.transform_idx2word(sentence, idx2word=idx2word) for sentence in predict_index.detach().cpu().numpy()]	
		r1, r2 = calc_rouge(original_sentences, predict_sentences)		
		rouge_1 += r1 / len(batch)
		rouge_2 += r2 / len(batch)
		loss = criterion(prob.transpose(1, 2), feature)	
		avg_loss += loss.detach().item()
		step = step + 1
		del feature, prob, loss, _, predict_index
	avg_loss = avg_loss / step
	rouge_1 = rouge_1 / step
	rouge_2 = rouge_2 / step
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