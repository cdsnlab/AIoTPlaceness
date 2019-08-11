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

	if args.enc_snapshot is None or args.dec_snapshot is None :
		print("Start from initial")
		embedding = nn.Embedding(v, k, max_norm=1, norm_type=2.0)
		embedding.weight.data = embedding.weight.data.to(device)
		encoder = net.ConvolutionEncoder(embedding, t3, args.filter_size, args.filter_shape, args.latent_size)
		decoder = net.DeconvolutionDecoder(embedding, args.tau, t3, args.filter_size, args.filter_shape, args.latent_size)
	else:
		print("Restart from snapshot")
		encoder = torch.load(os.path.join(CONFIG.DECONV_SNAPSHOT_PATH, args.enc_snapshot))
		decoder = torch.load(os.path.join(CONFIG.DECONV_SNAPSHOT_PATH, args.dec_snapshot))

	criterion = nn.NLLLoss()
	if torch.cuda.device_count() > 1 and args.use_cuda:
		encoder = DataParallelModel(encoder)
		decoder = DataParallelModel(decoder)
		criterion = DataParallelCriterion(criterion) 
	encoder.to(device)
	decoder.to(device)
	exp = Experiment("Reconstruction Training")
	try:
		lr = args.lr
		optimizer_enc = Adam(encoder.parameters(), lr=lr)
		optimizer_dec = Adam(decoder.parameters(), lr=lr)

		avg_loss = []
		rouge_1 = []
		rouge_2 = []

		encoder.train() 
		decoder.train() 

		steps = 0
		for epoch in range(args.epochs):
			print('Epoch:', epoch)
			for batch in train_loader:
				torch.cuda.empty_cache()
				feature = Variable(batch)
				if args.use_cuda:
					feature = feature.to(device)

				optimizer_enc.zero_grad()
				optimizer_dec.zero_grad()

				h = encoder(feature)
				prob = decoder(h)
				prob_t = torch.transpose(prob, 1, 2)
				reconstruction_loss = criterion(prob_t, feature)
				reconstruction_loss.backward()
				optimizer_enc.step()
				optimizer_dec.step()

				steps += 1

				if steps % 100 == 0:
					print("Epoch: {} at {}".format(epoch, str(datetime.datetime.now())))
					print("Steps: {}".format(steps))
					print("Loss: {}".format(reconstruction_loss.detach().item()))
					exp.metric("Loss", reconstruction_loss.detach().item())
				# check reconstructed sentence
				if steps % args.log_interval == 0:
					print("Test!!")
					input_data = feature[0]
					single_data = prob[0]
					_, predict_index = torch.max(single_data, 1)
					input_sentence = util.transform_id2word(input_data.detach(), train_loader.dataset.index2word, lang="en")
					predict_sentence = util.transform_id2word(predict_index.detach(), train_loader.dataset.index2word, lang="en")
					print("Input Sentence:")
					print(input_sentence)
					print("Output Sentence:")
					print(predict_sentence)
				del feature, prob

			if epoch % args.test_interval == 0:
				_avg_loss, _rouge_1, _rouge_2 = eval_reconstruction(encoder, decoder, test_loader, args)
				avg_loss.append(_avg_loss)
				rouge_1.append(_rouge_1)
				rouge_2.append(_rouge_2)

			if epoch % args.save_interval == 0:
				util.save_models(optimizer_enc, CONFIG.DECONV_SNAPSHOT_PATH, "encoder", epoch)
				util.save_models(optimizer_dec, CONFIG.DECONV_SNAPSHOT_PATH, "decoder", epoch)

		# finalization
		# save vocabulary
		with open(os.path.join(CONFIG.DECONV_VOCAB_PATH, 'word2index.p'), "wb") as w2i, \
		open(os.path.join(DECONV_VOCAB_PATH, 'index2word.p'), "wb") as i2w:
			pickle.dump(train_loader.dataset.word2index, w2i)
			pickle.dump(train_loader.dataset.index2word, i2w)

		# save models
		util.save_models(encoder, CONFIG.DECONV_SNAPSHOT_PATH, "encoder", "final")
		util.save_models(decoder, CONFIG.DECONV_SNAPSHOT_PATH, "decoder", "final")
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

def eval_reconstruction(encoder, decoder, data_iter, args):
	print("=================Eval======================")
	encoder.eval()
	decoder.eval()
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
		h = encoder(feature)
		prob = decoder(h)
		_, predict_index = torch.max(prob, 2)
		original_sentences = [util.transform_id2word(sentence, index2word, "en") for sentence in batch.detach()]
		predict_sentences = [util.transform_id2word(sentence, index2word, "en") for sentence in predict_index.detach()]
		r1, r2 = calc_rouge(original_sentences, predict_sentences)
		rouge_1 += r1
		rouge_2 += r2
		reconstruction_loss = compute_cross_entropy(prob, feature)
		
		avg_loss += reconstruction_loss.detach().item()
		del feature, prob
	avg_loss = avg_loss / len(data_iter.dataset)
	# avg_loss = avg_loss / args.sentence_len
	rouge_1 = rouge_1 / len(data_iter.dataset)
	rouge_2 = rouge_2 / len(data_iter.dataset)
	print("Evaluation - loss: {}  Rouge1: {}    Rouge2: {}".format(avg_loss, rouge_1, rouge_2))
	print("===============================================================")
	encoder.train()
	decoder.train()

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
