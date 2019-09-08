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
from model import text_model, imgseq_model, multimodal_model
from model.util import load_multimodal_data
from model.component import AdamW, cyclical_lr



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
	parser.add_argument('-arch', type=str, default='resnext101_32x8d', help='image embedding model')
	parser.add_argument('-image_embedding_dim', type=int, default=2048, help='embedding dimension of the model')
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
	with open(os.path.join(CONFIG.DATASET_PATH, args.target_dataset, 'word_embedding.p'), "rb") as f:
		text_embedding_model = cPickle.load(f)
	with open(os.path.join(CONFIG.DATASET_PATH, args.target_dataset, 'word_idx.json'), "r", encoding='utf-8') as f:
		word_idx = json.load(f)
	print("Loading embedding model completed")
	print("Loading dataset...")
	train_dataset, val_dataset = load_multimodal_data(args, CONFIG, word2idx=word_idx[1])
	print("Loading dataset completed")
	train_loader, val_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=args.shuffle),\
								  DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

	# t1 = max_sentence_len + 2 * (args.filter_shape - 1)
	t1 = CONFIG.MAX_SENTENCE_LEN
	t2 = int(math.floor((t1 - args.filter_shape) / 2) + 1) # "2" means stride size
	t3 = int(math.floor((t2 - args.filter_shape) / 2) + 1)
	args.t3 = t3
	text_embedding = nn.Embedding.from_pretrained(torch.FloatTensor(text_embedding_model))
	text_encoder = text_model.ConvolutionEncoder(text_embedding, t3, args.filter_size, args.filter_shape, args.latent_size)
	text_decoder = text_model.DeconvolutionDecoder(text_embedding, args.tau, t3, args.filter_size, args.filter_shape, args.latent_size, device)
	if args.text_pt:
		text_checkpoint = torch.load(os.path.join(CONFIG.CHECKPOINT_PATH, args.text_pt), map_location=lambda storage, loc: storage)
		text_encoder.load_state_dict(text_checkpoint['text_encoder'])
		text_decoder.load_state_dict(text_checkpoint['text_decoder'])
	imgseq_encoder = imgseq_model.RNNEncoder(args.image_embedding_dim, args.num_layer, args.latent_size, bidirectional=True)
	imgseq_decoder = imgseq_model.RNNDecoder(CONFIG.MAX_SEQUENCE_LEN, args.image_embedding_dim, args.num_layer, args.latent_size, bidirectional=True)
	if args.imgseq_pt:
		imgseq_checkpoint = torch.load(os.path.join(CONFIG.CHECKPOINT_PATH, args.imgseq_pt), map_location=lambda storage, loc: storage)
		imgseq_encoder.load_state_dict(imgseq_checkpoint['imgseq_encoder'])
		imgseq_decoder.load_state_dict(imgseq_checkpoint['imgseq_decoder'])
	multimodal_encoder = multimodal_model.MultimodalEncoder(text_encoder, imgseq_encoder, args.latent_size)
	multimodal_decoder = multimodal_model.MultimodalDecoder(text_decoder, imgseq_decoder, args.latent_size, CONFIG.MAX_SEQUENCE_LEN)

	if args.resume:
		print("Restart from checkpoint")
		checkpoint = torch.load(os.path.join(CONFIG.CHECKPOINT_PATH, args.resume), map_location=lambda storage, loc: storage)
		start_epoch = checkpoint['epoch']
		multimodal_encoder.load_state_dict(checkpoint['multimodal_encoder'])
		multimodal_decoder.load_state_dict(checkpoint['multimodal_decoder'])
	else:		
		print("Start from initial")
		start_epoch = 0
	
	multimodal_autoencoder = multimodal_model.MultimodalAutoEncoder(multimodal_encoder, multimodal_decoder)
	text_criterion = nn.NLLLoss().to(device)
	imgseq_criterion = nn.MSELoss().to(device)
	multimodal_autoencoder.to(device)

	optimizer = AdamW(multimodal_autoencoder.parameters(), lr=1., weight_decay=args.weight_decay, amsgrad=True)
	step_size = args.half_cycle_interval*len(train_loader)
	clr = cyclical_lr(step_size, min_lr=args.lr, max_lr=args.lr*args.lr_factor)
	scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, [clr])
	if args.resume:
		optimizer.load_state_dict(checkpoint['optimizer'])
		scheduler.load_state_dict(checkpoint['scheduler'])
	exp = Experiment("Multimodal autoencoder", capture_io=False)

	for arg, value in vars(args).items():
		exp.param(arg, value) 
	try:
		multimodal_autoencoder.train() 

		for epoch in range(start_epoch, args.epochs):
			print("Epoch: {}".format(epoch))
			for steps, (text_batch, imgseq_batch) in enumerate(train_loader):
				torch.cuda.empty_cache()
				text_feature = Variable(text_batch).to(device)
				imgseq_feature = Variable(imgseq_batch).to(device)
				optimizer.zero_grad()
				text_prob, imgseq_feature_hat = multimodal_autoencoder(text_feature, imgseq_feature)
				text_loss = text_criterion(text_prob.transpose(1, 2), text_feature)
				imgseq_loss = imgseq_criterion(imgseq_feature_hat, imgseq_feature)
				loss = text_loss + imgseq_loss
				del text_loss, imgseq_loss
				loss.backward()
				optimizer.step()
				scheduler.step()

				if (steps * args.batch_size) % args.log_interval == 0:					
					input_data = text_feature[0]
					single_data = text_prob[0]
					_, predict_index = torch.max(single_data, 1)
					input_sentence = util.transform_idx2word(input_data.detach().cpu().numpy(), idx2word=word_idx[0])
					predict_sentence = util.transform_idx2word(predict_index.detach().cpu().numpy(), idx2word=word_idx[0])	
					print("Epoch: {} at {} lr: {}".format(epoch, str(datetime.datetime.now()), str(scheduler.get_lr())))
					print("Steps: {}".format(steps))
					print("Loss: {}".format(loss.detach().item()))
					print("Input Sentence:")
					print(input_sentence)
					print("Output Sentence:")
					print(predict_sentence)
					del input_data, single_data, _, predict_index
				del text_feature, text_prob, imgseq_feature, imgseq_feature_hat, loss
			
			exp.log("\nEpoch: {} at {} lr: {}".format(epoch, str(datetime.datetime.now()), str(scheduler.get_lr())))
			_avg_loss, _rouge_1, _rouge_2 = eval_reconstruction_with_rouge(multimodal_autoencoder, word_idx[0], text_criterion, imgseq_criterion, val_loader, device)
			exp.log("\nEvaluation - loss: {}  Rouge1: {} Rouge2: {}".format(_avg_loss, _rouge_1, _rouge_2))

			util.save_models({
				'epoch': epoch + 1,
				'text_encoder': text_encoder.state_dict(),
				'text_decoder': text_decoder.state_dict(),
				'avg_loss': _avg_loss,
				'Rouge1:': _rouge_1,
				'Rouge2': _rouge_2,
				'optimizer' : optimizer.state_dict(),
				'scheduler' : scheduler.state_dict()
			}, CONFIG.CHECKPOINT_PATH, "text_autoencoder")

		print("Finish!!!")

	finally:
		exp.end()

def eval_reconstruction_with_rouge(autoencoder, idx2word, text_criterion, imgseq_criterion, data_iter, device):
	print("=================Eval======================")
	autoencoder.eval()
	step = 0
	avg_loss = 0.
	rouge_1 = 0.
	rouge_2 = 0.
	for batch in tqdm(data_iter):
		torch.cuda.empty_cache()
		with torch.no_grad():
			text_feature = Variable(text_batch).to(device)
			imgseq_feature = Variable(imgseq_batch).to(device)
		text_prob, imgseq_feature_hat = autoencoder(text_feature, imgseq_feature)
		_, predict_index = torch.max(text_prob, 2)
		original_sentences = [util.transform_idx2word(sentence, idx2word=idx2word) for sentence in feature.detach().cpu().numpy()]		
		predict_sentences = [util.transform_idx2word(sentence, idx2word=idx2word) for sentence in predict_index.detach().cpu().numpy()]	
		r1, r2 = calc_rouge(original_sentences, predict_sentences)		
		rouge_1 += r1 / len(batch)
		rouge_2 += r2 / len(batch)
		text_loss = text_criterion(text_prob.transpose(1, 2), text_feature)
		imgseq_loss = imgseq_criterion(imgseq_feature_hat, imgseq_feature)
		loss = text_loss + imgseq_loss
		del text_loss, imgseq_loss
		avg_loss += loss.detach().item()
		step = step + 1
		del text_feature, text_prob, imgseq_feature, imgseq_feature_hat, loss, _, predict_index
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