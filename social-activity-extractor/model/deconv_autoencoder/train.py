import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import StepLR, CyclicLR

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

from model.deconv_autoencoder import util
from model.deconv_autoencoder import net
from model.deconv_autoencoder.util import load_hotel_review_data, load_corpus_data
from model.deconv_autoencoder.parallel import DataParallelModel, DataParallelCriterion
from model.deconv_autoencoder.adamW import AdamW


def train_reconstruction(args, CONFIG):
	device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")
	print("Loading embedding model...")
	model_name = 'FASTTEXT_' + args.embedding_model + '.model'
	embedding_model = FastTextKeyedVectors.load(os.path.join(CONFIG.EMBEDDING_PATH, model_name))
	# pad_value = np.finfo(np.float32).eps
	pad_value = 1.
	args.pad_value = pad_value
	embedding_model.add("<PAD>", np.full(embedding_model.vector_size, pad_value), replace=True)
	embedding_model.init_sims(replace=True)	
	print("Building index...")
	indexer = AnnoyIndexer(embedding_model, 10)
	print("Loading embedding model completed")
	print("Loading dataset...")
	train_dataset, val_dataset, max_sentence_len = load_corpus_data(args, CONFIG, embedding_model)
	print("Loading dataset completed")
	train_loader, val_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=args.shuffle),\
								  DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

	# t1 = max_sentence_len + 2 * (args.filter_shape - 1)
	t1 = max_sentence_len
	t2 = int(math.floor((t1 - args.filter_shape) / 2) + 1) # "2" means stride size
	t3 = int(math.floor((t2 - args.filter_shape) / 2) + 1)
	if args.snapshot is None:
		print("Start from initial")
		if args.RNN:
			print("Set autoencoder to RNN")
			autoencoder = net.RNNAutoEncoder(train_dataset.embedding_dim(), max_sentence_len, args.num_layer, args.latent_size)
		else:
			print("Set autoencoder to CNN")
			autoencoder = net.CNNAutoEncoder(train_dataset.embedding_dim(), t3, args.filter_size, args.filter_shape, args.latent_size)
	else:
		print("Restart from snapshot")
		autoencoder = torch.load(os.path.join(CONFIG.SNAPSHOT_PATH, args.snapshot))

	criterion = nn.MSELoss()
	autoencoder.to(device)
	criterion.to(device)
	if args.distributed:
		autoencoder = DataParallelModel(autoencoder)
		criterion = DataParallelCriterion(criterion)
	exp = Experiment("Reconstruction Training")
	try:
		optimizer = AdamW(autoencoder.parameters(), lr=args.lr, weight_decay=args.weight_decay, amsgrad=True)
		avg_loss = []
		rouge_1 = []
		rouge_2 = []

		autoencoder.train() 

		steps = 0
		for epoch in range(args.epochs):
			print("Epoch: {}".format(epoch))
			for batch in train_loader:
				torch.cuda.empty_cache()
				feature = Variable(batch)
				if args.use_cuda:
					feature = feature.to(device)
				optimizer.zero_grad()
				feature_hat = autoencoder(feature)
				loss = criterion(feature_hat, feature)
				if args.distributed:
					loss = loss.mean()
				loss.backward()
				optimizer.step()

				steps += 1

				if steps % 100 == 0:
					print("Epoch: {} at {}".format(epoch, str(datetime.datetime.now())))
					print("Steps: {}".format(steps))
					print("Loss: {}".format(loss.detach().item()))
					exp.metric("Loss", loss.detach().item())
				# check reconstructed sentence
				if steps % args.log_interval == 0:
					print("Test!!")
					input_data = feature[0]
					if args.distributed:
						single_data = feature_hat[0][0]
					else:
						single_data = feature_hat[0]
					input_sentence = util.transform_vec2sentence(input_data.detach().cpu().numpy(), embedding_model, indexer)
					predict_sentence = util.transform_vec2sentence(single_data.detach().cpu().numpy(), embedding_model, indexer)
					print("Input Sentence:")
					print(input_sentence)
					print("Output Sentence:")
					print(predict_sentence)
					del input_data, single_data
				del feature, feature_hat, loss

			if epoch % args.test_interval == 0:
				_avg_loss, _rouge_1, _rouge_2 = eval_reconstruction(autoencoder, embedding_model, indexer, criterion, val_loader, args, device)
				avg_loss.append(_avg_loss)
				rouge_1.append(_rouge_1)
				rouge_2.append(_rouge_2)

			if epoch % args.save_interval == 0:
				util.save_models(autoencoder, CONFIG.SNAPSHOT_PATH, "autoencoder", epoch)

		# finalization

		# save models
		util.save_models(autoencoder, CONFIG.SNAPSHOT_PATH, "autoencoder", "final")
		table = []
		table.append(avg_loss)
		table.append(rouge_1)
		table.append(rouge_2)
		df = pd.DataFrame(table)
		df = df.transpose()
		df.columns = ['avg_loss', 'rouge_1', 'rouge_2']
		df.to_csv(os.path.join(CONFIG.CSV_PATH, 'Evaluation_result.csv'), encoding='utf-8-sig')

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
	for batch in tqdm(data_iter):
		torch.cuda.empty_cache()
		with torch.no_grad():
			feature = Variable(batch)
			if args.use_cuda:
				feature = feature.to(device)
		feature_hat = autoencoder(feature)	
		original_sentences = [util.transform_vec2sentence(sentence, embedding_model, indexer) for sentence in feature.detach().cpu().numpy()]		
		predict_sentences = [util.transform_vec2sentence(sentence, embedding_model, indexer) for sentence in feature_hat.detach().cpu().numpy()]	
		r1, r2 = calc_rouge(original_sentences, predict_sentences)		
		rouge_1 += r1 / len(batch)
		rouge_2 += r2 / len(batch)
		loss = criterion(feature_hat, feature)	
		if args.distributed:
			loss = loss.mean()
		avg_loss += loss.detach().item()
		step = step + 1
		del feature, feature_hat, loss
	avg_loss = avg_loss / step
	# avg_loss = avg_loss / args.max_sentence_len
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
