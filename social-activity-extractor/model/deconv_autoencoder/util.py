import torch
from torch.utils.data import Dataset, random_split
import math
import os
import sys
import numpy as np
from tqdm import tqdm
from collections import Counter
from copy import deepcopy
from scipy.spatial.distance import cdist


class ToTensor(object):
	"""Convert ndarrays in sample to Tensors."""
	def __call__(self, data):
		return torch.from_numpy(data).type(torch.FloatTensor)


def load_corpus_data(args, CONFIG, embedding_model):
	corpus_name = args.target_corpus + '.txt'	
	with open(os.path.join(CONFIG.DATA_PATH, 'corpus', corpus_name), "r", encoding="utf-8") as f:
		max_sentence_len = 0
		full_data = []
		while True:
			line = f.readline()
			if not line: break;
			full_data.append(line)
			word_list = line.split()
			if len(word_list) > max_sentence_len:
				max_sentence_len = len(word_list)
	if max_sentence_len % 2 == 0:
		max_sentence_len = max_sentence_len + 1
	train_size = int(args.split_rate * len(full_data))
	val_size = len(full_data) - train_size
	train_data, val_data = torch.utils.data.random_split(full_data, [train_size, val_size])
	train_dataset, val_dataset = CorpusDataset(train_data, embedding_model, max_sentence_len, args.pad_value, transform=ToTensor()), \
							 CorpusDataset(val_data, embedding_model, max_sentence_len, args.pad_value, transform=ToTensor())
	return train_dataset, val_dataset, max_sentence_len

class CorpusDataset(Dataset):
	"""
	Hotel Reviews Dataset
	"""
	def __init__(self, data_list, embedding_model, max_sentence_len, pad_value, transform=None):
		self.data = data_list
		self.transform = transform
		self.embedding_model = embedding_model
		self.max_sentence_len = max_sentence_len
		self.pad_value = pad_value

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		sentence = self.data[idx]
		word_list = sentence.split()
		vector_list = []
		if len(word_list) > self.max_sentence_len:
			# truncate sentence if sentence length is longer than `max_sentence_len`
			word_list = word_list[:self.max_sentence_len]
		else:
			word_list = word_list + ['<PAD>'] * (self.max_sentence_len - len(word_list))
		for word in word_list:
			vector = self.embedding_model.get_vector(word)
			vector_list.append(vector)
		vector_array = np.array(vector_list, dtype=np.float32)
		# if len(word_list) > self.max_sentence_len:
		# 	# truncate sentence if sentence length is longer than `max_sentence_len`
		# 	vector_array = np.array(vector_list[:self.max_sentence_len], dtype=np.float32)
		# else:
		# 	# pad sentence with 0 if sentence length is shorter than `max_sentence_len`
		# 	vector_array = np.lib.pad(np.array(vector_list, dtype=np.float32),
		# 							((0, self.max_sentence_len - len(word_list)), (0,0)),
		# 							"constant",
		# 							constant_values=(self.pad_value))
		if self.transform:
			vector_array = self.transform(vector_array)
		del vector_list
		return vector_array

	def embedding_dim(self):
		return self.embedding_model.vector_size

def transform_vec2sentence(vector_list, embedding_model, indexer):
	return " ".join([embedding_model.most_similar(positive=[vector], topn=1, indexer=indexer)[0][0] for vector in vector_list])

def save_models(model, path, prefix, epoch):
	if not os.path.isdir(path):
		os.makedirs(path)
	model_save_path = '{}/{}_epoch_{}.pt'.format(path, prefix, epoch)
	torch.save(model, model_save_path)
