import torch
import torchvision.transforms as transforms
from torchvision.datasets.folder import pil_loader
from torch.utils.data import Dataset, random_split
import math
import os
import sys
import numpy as np
import pandas as pd
import _pickle as cPickle
from tqdm import tqdm


class ToTensor(object):
	"""Convert ndarrays in sample to Tensors."""
	def __call__(self, data):
		return torch.from_numpy(data).type(torch.FloatTensor)


def load_text_data(args, CONFIG, embedding_model):	
	full_data = []
	df_data = pd.read_csv(os.path.join(CONFIG.DATASET_PATH, args.target_dataset, 'posts.csv'), header=None, encoding='utf-8')
	pbar = tqdm(total=df_data.shape[0])
	for index, row in df_data.iterrows():
		pbar.update(1)
		full_data.append(row.iloc[1])
	pbar.close()

	train_size = int(args.split_rate * len(full_data))
	train_data, val_data = full_data[:train_size], full_data[train_size:]
	train_dataset, val_dataset = TextDataset(train_data, embedding_model, CONFIG, transform=ToTensor()), \
							 TextDataset(val_data, embedding_model, CONFIG, transform=ToTensor())
	return train_dataset, val_dataset

class TextDataset(Dataset):
	def __init__(self, data_list, embedding_model, CONFIG, transform=None):
		self.data = data_list
		self.embedding_model = embedding_model
		self.CONFIG = CONFIG
		self.transform = transform

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		word_list = self.data[idx].split()
		vector_list = []
		if len(word_list) > self.CONFIG.MAX_SENTENCE_LEN:
			# truncate sentence if sentence length is longer than `max_sentence_len`
			word_list = word_list[:self.CONFIG.MAX_SENTENCE_LEN]
			word_list[-1] = '<EOS>'
		else:
			word_list = word_list + ['<PAD>'] * (self.CONFIG.MAX_SENTENCE_LEN - len(word_list))
		for word in word_list:
			vector = self.embedding_model.get_vector(word)
			vector_list.append(vector)
		vector_array = np.array(vector_list, dtype=np.float32)
		del word_list, vector_list

		if self.transform:
			vector_array = self.transform(vector_array)
		return vector_array

def load_imgseq_data(args, CONFIG):
	full_data = []

	image_dir = os.path.join(CONFIG.DATASET_PATH, args.target_dataset, 'resnet50')
	for image_path in tqdm(os.listdir(image_dir)):
		with open(os.path.join(image_dir, image_path), "rb") as f:
			full_data.append(cPickle.load(f))
		f.close()
	full_data = np.array(full_data, dtype=np.float32)
	train_size = int(args.split_rate * len(full_data))
	train_data, val_data = full_data[:train_size], full_data[train_size:]
	train_dataset, val_dataset = ImgseqDataset(train_data, embedding_model, CONFIG, transform=ToTensor()), \
							 ImgseqDataset(val_data, embedding_model, CONFIG, transform=ToTensor())
	return train_dataset, val_dataset


class ImgseqDataset(Dataset):
	def __init__(self, data_list, embedding_model, CONFIG, transform):
		self.data = data_list
		self.embedding_model = embedding_model
		self.CONFIG = CONFIG
		self.transform = transform

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		vector_array = self.data[idx]
		if self.transform:
			vector_array = self.transform(vector_array)
		return vector_array


def transform_vec2sentence(vector_list, embedding_model, indexer):
	return " ".join([embedding_model.most_similar(positive=[vector], topn=1, indexer=indexer)[0][0] for vector in vector_list])

def save_models(checkpoint, path, prefix):
	if not os.path.isdir(path):
		os.makedirs(path)
	checkpoint_save_path = '{}/{}_epoch_{}.pt'.format(path, prefix, checkpoint['epoch'])
	torch.save(checkpoint, checkpoint_save_path)
