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
from hyperdash import Experiment

from model.deconv_autoencoder import util
from model.deconv_autoencoder import net
from model.deconv_autoencoder.datasets import load_hotel_review_data
from model.deconv_autoencoder.parallel import DataParallelModel, DataParallelCriterion


def train_reconstruction(args, CONFIG):
	device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")
	train_data, test_data = load_hotel_review_data(args.sentence_len)
	train_loader, test_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=args.shuffle),\
								  DataLoader(test_data, batch_size=int(args.batch_size/2), shuffle=False)

	k = args.embed_dim
	v = train_data.vocab_lennght()
	# t1 = args.sentence_len + 2 * (args.filter_shape - 1)
	t1 = args.sentence_len
	t2 = int(math.floor((t1 - args.filter_shape) / 2) + 1) # "2" means stride size
	t3 = int(math.floor((t2 - args.filter_shape) / 2) + 1)

	if args.snapshot is None:
		print("Start from initial")
		embedding = nn.Embedding(v, k, max_norm=1, norm_type=2.0)
		embedding.weight.data = embedding.weight.data.to(device)
		autoencoder = net.AutoEncoder(embedding, args.tau, t3, args.filter_size, args.filter_shape, args.latent_size)
	else:
		print("Restart from snapshot")
		autoencoder = torch.load(os.path.join(CONFIG.DECONV_SNAPSHOT_PATH, args.snapshot))

	criterion = nn.NLLLoss()
	autoencoder.to(device)
	criterion.to(device)
	if args.distributed:
		autoencoder = DataParallelModel(autoencoder)
		criterion = DataParallelCriterion(criterion)
	exp = Experiment("Reconstruction Training")
	try:
		lr = args.lr
		optimizer = Adam(autoencoder.parameters(), lr=lr)

		avg_loss = []
		rouge_1 = []
		rouge_2 = []

		autoencoder.train() 

		steps = 0
		for epoch in range(args.epochs):
			print('Epoch:', epoch)
			for batch in train_loader:
				torch.cuda.empty_cache()
				feature = Variable(batch)
				if args.use_cuda:
					feature = feature.to(device)

				optimizer.zero_grad()

				prob = autoencoder(feature)
				loss = criterion(prob, feature)
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
						single_data = prob[0][0]
					else:
						single_data = prob[0]
					_, predict_index = torch.max(single_data, 0)
					input_sentence = util.transform_id2word(input_data.detach(), train_loader.dataset.index2word, lang="en")
					predict_sentence = util.transform_id2word(predict_index.detach(), train_loader.dataset.index2word, lang="en")
					print("Input Sentence:")
					print(input_sentence)
					print("Output Sentence:")
					print(predict_sentence)
					del input_data, single_data, _, predict_index
				del feature, prob, loss

			if epoch % args.test_interval == 0:
				_avg_loss, _rouge_1, _rouge_2 = eval_reconstruction(autoencoder, criterion, test_loader, args, device)
				avg_loss.append(_avg_loss)
				rouge_1.append(_rouge_1)
				rouge_2.append(_rouge_2)

			if epoch % args.save_interval == 0:
				util.save_models(optimizer, CONFIG.DECONV_SNAPSHOT_PATH, "autoencoder", epoch)

		# finalization
		# save vocabulary
		with open(os.path.join(CONFIG.DECONV_VOCAB_PATH, 'word2index.p'), "wb") as w2i, \
		open(os.path.join(CONFIG.DECONV_VOCAB_PATH, 'index2word.p'), "wb") as i2w:
			pickle.dump(train_loader.dataset.word2index, w2i)
			pickle.dump(train_loader.dataset.index2word, i2w)

		# save models
		util.save_models(autoencoder, CONFIG.DECONV_SNAPSHOT_PATH, "autoencoder", "final")
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

def eval_reconstruction(autoencoder, criterion, data_iter, args, device):
	print("=================Eval======================")
	autoencoder.eval()
	avg_loss = 0
	rouge_1 = 0.0
	rouge_2 = 0.0
	index2word = data_iter.dataset.index2word
	for batch in data_iter:
		torch.cuda.empty_cache()
		with torch.no_grad():
			feature = Variable(batch)
			if args.use_cuda:
				feature = feature.to(device)
		prob = autoencoder(feature)		
		if args.distributed:
			concat_prob = torch.cat(prob, 0)
			_, predict_index = torch.max(concat_prob, 1)
			del concat_prob
		else:
			_, predict_index = torch.max(prob, 1)
		original_sentences = [util.transform_id2word(sentence, index2word, "en") for sentence in batch.detach()]
		predict_sentences = [util.transform_id2word(sentence, index2word, "en") for sentence in predict_index.detach()]
		r1, r2 = calc_rouge(original_sentences, predict_sentences)
		rouge_1 += r1
		rouge_2 += r2
		loss = criterion(prob, feature)
		if args.distributed:
			loss = loss.mean()
		avg_loss += loss.detach().item()
		del feature, prob, _, predict_index, loss
	avg_loss = avg_loss / len(data_iter.dataset)
	# avg_loss = avg_loss / args.sentence_len
	rouge_1 = rouge_1 / len(data_iter.dataset)
	rouge_2 = rouge_2 / len(data_iter.dataset)
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
