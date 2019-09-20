import argparse
import config
import requests
import json
import pickle
import datetime
import os
import csv
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
from model.util import load_fullmultimodal_data


CONFIG = config.Config

def slacknoti(contentstr):
	webhook_url = "https://hooks.slack.com/services/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"
	payload = {"text": contentstr}
	requests.post(webhook_url, data=json.dumps(payload), headers={'Content-Type': 'application/json'})

def main():
	parser = argparse.ArgumentParser(description='text convolution-deconvolution auto-encoder model')
	parser.add_argument('-tau', type=float, default=0.01, help='temperature parameter')
	# data
	parser.add_argument('-target_dataset', type=str, default=None, help='folder name of target dataset')
	parser.add_argument('-shuffle', default=True, help='shuffle data every epoch')
	parser.add_argument('-split_rate', type=float, default=0.9, help='split rate between train and validation')
	parser.add_argument('-batch_size', type=int, default=16, help='batch size for training')

	# model
	parser.add_argument('-latent_size', type=int, default=1000, help='size of latent variable')
	parser.add_argument('-filter_size', type=int, default=300, help='filter size of convolution')
	parser.add_argument('-filter_shape', type=int, default=5,
						help='filter shape to use for convolution')
	parser.add_argument('-arch', type=str, default='resnext101_32x8d', help='image embedding model')
	parser.add_argument('-image_embedding_dim', type=int, default=2048, help='embedding dimension of the model')
	parser.add_argument('-num_layer', type=int, default=4, help='layer number')
	parser.add_argument('-encode_latent', type=int, default=1000, help='latent vector size of encoder')
	parser.add_argument('-decode_latent', type=int, default=1000, help='latent vector size of decoder')
	parser.add_argument('-normalize', action='store_true', default=False, help='normalize before fusion')
	parser.add_argument('-add_latent', action='store_true', default=False, help='add latent or concat')
	parser.add_argument('-no_decode', action='store_true', default=False, help='no decode')

	# train
	parser.add_argument('-noti', action='store_true', default=False, help='whether using gpu server')
	parser.add_argument('-gpu', type=str, default='cuda', help='gpu number')
	# option
	parser.add_argument('-checkpoint', type=str, default=None, help='filename of checkpoint to resume ')

	args = parser.parse_args()

	if args.noti:
		slacknoti("underkoo start using")
	get_latent(args)
	if args.noti:
		slacknoti("underkoo end using")



def get_latent(args):
	device = torch.device(args.gpu)
	print("Loading embedding model...")
	with open(os.path.join(CONFIG.DATASET_PATH, args.target_dataset, 'word_embedding.p'), "rb") as f:
		text_embedding_model = cPickle.load(f)
	with open(os.path.join(CONFIG.DATASET_PATH, args.target_dataset, 'word_idx.json'), "r", encoding='utf-8') as f:
		word_idx = json.load(f)
	print("Loading embedding model completed")
	print("Loading dataset...")
	full_dataset = load_fullmultimodal_data(args, CONFIG, word2idx=word_idx[1])
	print("Loading dataset completed")
	full_loader = DataLoader(full_dataset, batch_size=args.batch_size, shuffle=False)
	
	# t1 = max_sentence_len + 2 * (args.filter_shape - 1)
	t1 = CONFIG.MAX_SENTENCE_LEN
	t2 = int(math.floor((t1 - args.filter_shape) / 2) + 1) # "2" means stride size
	t3 = int(math.floor((t2 - args.filter_shape) / 2) + 1)
	args.t3 = t3

	text_embedding = nn.Embedding.from_pretrained(torch.FloatTensor(text_embedding_model))
	text_encoder = text_model.ConvolutionEncoder(text_embedding, t3, args.filter_size, args.filter_shape, args.encode_latent)
	imgseq_encoder = imgseq_model.RNNEncoder(args.image_embedding_dim, args.num_layer, args.encode_latent, bidirectional=True)
	multimodal_encoder = multimodal_model.MultimodalEncoder(text_encoder, imgseq_encoder, args.latent_size, args.normalize, args.add_latent)
	checkpoint = torch.load(os.path.join(CONFIG.CHECKPOINT_PATH, args.checkpoint), map_location=lambda storage, loc: storage)
	multimodal_encoder.load_state_dict(checkpoint['multimodal_encoder'])
	multimodal_encoder.to(device)
	multimodal_encoder.eval() 


	csv_name = 'latent_' + args.target_dataset
	if args.normalize:
		csv_name = csv_name + "_normalize"
	if args.add_latent:
		csv_name = csv_name + "_add_latent"
	if args.no_decode:
		csv_name = csv_name + "_no_decode"
	csv_name = csv_name + '.csv'
	#f_csv = open(os.path.join(CONFIG.CSV_PATH, 'latent_' + args.target_dataset + '.csv'), 'w', encoding='utf-8-sig')
	#wr = csv.writer(f_csv)
	short_code_list = []
	row_list = []
	for text_batch, imgseq_batch, short_code in tqdm(full_loader):
		torch.cuda.empty_cache()
		with torch.no_grad():	
			text_feature = Variable(text_batch).to(device)
			imgseq_feature = Variable(imgseq_batch).to(device)
		h = multimodal_encoder(text_feature, imgseq_feature)

		for _short_code, _h in zip(short_code, h):
			short_code_list.append(_short_code)
			row_list.append(_h.detach().cpu().numpy().tolist())
			# row = [_short_code] + _h.detach().cpu().numpy().tolist()
			# wr.writerow(row)
		del text_feature, imgseq_feature
	#f_csv.close()
	result_df = pd.DataFrame(data=row_list, index=short_code_list, columns=[i for i in range(args.latent_size)])
	result_df.index.name = short_code
	result_df.sort_index(inplace=True)
	result_df.to_csv(os.path.join(CONFIG.CSV_PATH, csv_name), encoding='utf-8-sig')
	print("Finish!!!")



if __name__ == '__main__':
	main()