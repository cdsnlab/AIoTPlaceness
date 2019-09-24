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

torch.manual_seed(42)
def load_text_data(args, CONFIG, word2idx):	
	full_data = []
	df_data = pd.read_csv(os.path.join(CONFIG.DATASET_PATH, args.target_dataset, 'posts.csv'), header=None, encoding='utf-8-sig')
	pbar = tqdm(total=df_data.shape[0])
	for index, row in df_data.iterrows():
		pbar.update(1)
		text_data = row.iloc[1]
		full_data.append(text_data)
		del text_data
	pbar.close()
	train_size = int(args.split_rate * len(full_data))
	val_size = len(full_data) - train_size
	train_data, val_data = torch.utils.data.random_split(full_data, [train_size, val_size])
	train_dataset, val_dataset = TextDataset(train_data, CONFIG, word2idx), \
							 TextDataset(val_data, CONFIG, word2idx)
	return train_dataset, val_dataset

class TextDataset(Dataset):
	def __init__(self, data_list, CONFIG, word2idx):
		self.data = data_list
		self.word2idx = word2idx
		self.CONFIG = CONFIG

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		word_list = self.data[idx].split()
		index_list = []
		if len(word_list) > self.CONFIG.MAX_SENTENCE_LEN:
			# truncate sentence if sentence length is longer than `max_sentence_len`
			word_list = word_list[:self.CONFIG.MAX_SENTENCE_LEN]
			word_list[-1] = '<EOS>'
		else:
			word_list = word_list + ['<PAD>'] * (self.CONFIG.MAX_SENTENCE_LEN - len(word_list))
		for word in word_list:
			index_list.append(self.word2idx[word])
		text_array = np.array(index_list)
		text_tensor = torch.from_numpy(text_array).type(torch.LongTensor)
		return text_tensor

def load_imgseq_pretrain_data(args, CONFIG):
	full_data = []
	dataset_path = os.path.join(CONFIG.DATA_PATH, 'dataset', args.target_dataset)
	image_dir = os.path.join(dataset_path, 'resize224')
	for image_path in tqdm(os.listdir(image_dir)):
		with open(os.path.join(image_dir, image_path), "rb") as f:
			image_data = cPickle.load(f)
		if len(full_data) == 0:
			full_data = image_data
		else:
			full_data = np.concatenate([full_data, image_data], axis=0)
		f.close()
		del image_data
	#full_data = np.concatenate(full_data, axis=0)
	train_size = int(args.split_rate * len(full_data))
	val_size = len(full_data) - train_size
	train_data, val_data = torch.utils.data.random_split(full_data, [train_size, val_size])
	train_dataset, val_dataset = Imgseq_pretrain_Dataset(train_data, CONFIG), \
							 Imgseq_pretrain_Dataset(val_data, CONFIG)
	return train_dataset, val_dataset


class Imgseq_pretrain_Dataset(Dataset):
	def __init__(self, data_list, CONFIG):
		self.data = data_list
		self.CONFIG = CONFIG

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		image_tensor = torch.from_numpy(self.data[idx]).type(torch.FloatTensor)
		return image_tensor


def load_imgseq_data(args, CONFIG):
	full_data = []
	print("Using embedding model: ", args.arch)
	image_dir = os.path.join(CONFIG.DATASET_PATH, args.target_dataset, args.arch)
	for image_path in tqdm(os.listdir(image_dir)):
		with open(os.path.join(image_dir, image_path), "rb") as f:
			image_data = cPickle.load(f)
		full_data.append(image_data)
		f.close()
		del image_data
	full_data = np.array(full_data, dtype=np.float32)
	train_size = int(args.split_rate * len(full_data))
	val_size = len(full_data) - train_size
	train_data, val_data = torch.utils.data.random_split(full_data, [train_size, val_size])
	train_dataset, val_dataset = ImgseqDataset(train_data, CONFIG), \
							 ImgseqDataset(val_data, CONFIG)
	return train_dataset, val_dataset


class ImgseqDataset(Dataset):
	def __init__(self, data_list, CONFIG):
		self.data = data_list
		self.CONFIG = CONFIG

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		imgseq_tensor = torch.from_numpy(self.data[idx]).type(torch.FloatTensor)
		return imgseq_tensor

def load_multimodal_data(args, CONFIG, word2idx):	
	full_data = []
	df_data = pd.read_csv(os.path.join(CONFIG.DATASET_PATH, args.target_dataset, 'posts.csv'), header=None, encoding='utf-8-sig')
	image_dir = os.path.join(CONFIG.DATASET_PATH, args.target_dataset, args.arch)
	pbar = tqdm(total=df_data.shape[0])
	for index, row in df_data.iterrows():
		pbar.update(1)
		text_data = row.iloc[1]
		image_path = os.path.join(image_dir, row.iloc[0]) + '.p'
		if os.path.exists(image_path):
			with open(image_path, "rb") as f:
				image_data = cPickle.load(f)
			full_data.append([text_data, image_data])
			del text_data, image_data
		else:
			del text_data
			continue
	pbar.close()
	train_size = int(args.split_rate * len(full_data))
	val_size = len(full_data) - train_size
	train_data, val_data = torch.utils.data.random_split(full_data, [train_size, val_size])
	train_dataset, val_dataset = MultimodalDataset(train_data, CONFIG, word2idx), \
							 MultimodalDataset(val_data, CONFIG, word2idx)
	return train_dataset, val_dataset

class MultimodalDataset(Dataset):
	def __init__(self, data_list, CONFIG, word2idx):
		self.data = data_list
		self.word2idx = word2idx
		self.CONFIG = CONFIG

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		word_list = self.data[idx][0].split()
		index_list = []
		if len(word_list) > self.CONFIG.MAX_SENTENCE_LEN:
			# truncate sentence if sentence length is longer than `max_sentence_len`
			word_list = word_list[:self.CONFIG.MAX_SENTENCE_LEN]
			word_list[-1] = '<EOS>'
		else:
			word_list = word_list + ['<PAD>'] * (self.CONFIG.MAX_SENTENCE_LEN - len(word_list))
		for word in word_list:
			index_list.append(self.word2idx[word])
		text_array = np.array(index_list)
		text_tensor = torch.from_numpy(text_array).type(torch.LongTensor)
		imgseq_tensor = torch.from_numpy(self.data[idx][1]).type(torch.FloatTensor)

		return text_tensor, imgseq_tensor

def load_fullmultimodal_data(args, CONFIG, word2idx):	
	full_data = []
	df_data = pd.read_csv(os.path.join(CONFIG.DATASET_PATH, args.target_dataset, 'posts.csv'), header=None, encoding='utf-8-sig')
	image_dir = os.path.join(CONFIG.DATASET_PATH, args.target_dataset, args.arch)
	pbar = tqdm(total=df_data.shape[0])
	for index, row in df_data.iterrows():
		pbar.update(1)
		short_code = row.iloc[0]
		text_data = row.iloc[1]
		image_path = os.path.join(image_dir, row.iloc[0]) + '.p'
		if os.path.exists(image_path):
			with open(image_path, "rb") as f:
				image_data = cPickle.load(f)
			full_data.append([text_data, image_data, short_code])
			del text_data, image_data
		else:
			del text_data
			continue
	pbar.close()
	full_dataset = FullMultimodalDataset(full_data, CONFIG, word2idx)
	return full_dataset

class FullMultimodalDataset(Dataset):
	def __init__(self, data_list, CONFIG, word2idx):
		self.data = data_list
		self.word2idx = word2idx
		self.CONFIG = CONFIG

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		word_list = self.data[idx][0].split()
		index_list = []
		if len(word_list) > self.CONFIG.MAX_SENTENCE_LEN:
			# truncate sentence if sentence length is longer than `max_sentence_len`
			word_list = word_list[:self.CONFIG.MAX_SENTENCE_LEN]
			word_list[-1] = '<EOS>'
		else:
			word_list = word_list + ['<PAD>'] * (self.CONFIG.MAX_SENTENCE_LEN - len(word_list))
		for word in word_list:
			index_list.append(self.word2idx[word])
		text_array = np.array(index_list)
		text_tensor = torch.from_numpy(text_array).type(torch.LongTensor)
		imgseq_tensor = torch.from_numpy(self.data[idx][1]).type(torch.FloatTensor)

		return text_tensor, imgseq_tensor, self.data[idx][2]

def transform_idx2word(index, idx2word):
	return " ".join([idx2word[str(idx)] for idx in index])


def save_models(checkpoint, path, prefix):
	if not os.path.isdir(path):
		os.makedirs(path)
	checkpoint_save_path = '{}/{}_epoch_{}.pt'.format(path, prefix, checkpoint['epoch'])
	torch.save(checkpoint, checkpoint_save_path)
