import torch
import torchvision.transforms as transforms
from torchvision.datasets.folder import pil_loader
from torch.utils.data import Dataset, random_split
import math
import os
import sys
import numpy as np
import _pickle as cPickle
from tqdm import tqdm


class ToTensor(object):
	"""Convert ndarrays in sample to Tensors."""
	def __call__(self, data):
		return torch.from_numpy(data).type(torch.FloatTensor)


def load_text_data(args, CONFIG, embedding_model):
	dataset_path = os.path.join(CONFIG.DATASET_PATH, args.target_dataset)
	
	full_data = []
	for loc_id in tqdm(os.listdir(dataset_path)):
		path_dir = os.path.join(dataset_path, loc_id)
		for post in os.listdir(path_dir):
			with open(os.path.join(path_dir, post, "text.txt"), 'r', encoding='utf-8', newline='\n') as f:
				text_data = f.read()
				full_data.append(text_data)
			f.close()

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

def load_imgseq_data(args, CONFIG, embedding_model, device):
	dataset_path = os.path.join(CONFIG.DATASET_PATH, args.target_dataset)
	transform_func = transforms.Compose([
			transforms.Resize(256),
			transforms.CenterCrop(224),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406],
									 std=[0.229, 0.224, 0.225])
		])
	full_data = []
	for loc_id in tqdm(os.listdir(dataset_path)):
		path_dir = os.path.join(dataset_path, loc_id)
		for post in os.listdir(path_dir):
			img_dir = os.path.join(path_dir, post, "images")
			img_list = []
			for image in os.listdir(img_dir):
				img_path = os.path.join(img_dir,image)
				img_list.append(transform_func(pil_loader(img_path)))
			img_tensor = torch.stack(img_list)
			full_data.append(img_tensor)

	train_size = int(args.split_rate * len(full_data))
	train_data, val_data = full_data[:train_size], full_data[train_size:]
	train_dataset, val_dataset = ImgseqDataset(train_data, embedding_model, CONFIG, device=device, transform=ToTensor()), \
							 ImgseqDataset(val_data, embedding_model, CONFIG, device=device, transform=ToTensor())
	return train_dataset, val_dataset


class ImgseqDataset(Dataset):
	def __init__(self, data_list, embedding_model, CONFIG, device, transform):
		self.data = data_list
		self.embedding_model = embedding_model
		self.CONFIG = CONFIG
		self.device = device
		self.transform = transform

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		with torch.no_grad():
			image_tensor = self.data[idx].to(self.device)
		vector_array = self.embedding_model(image_tensor).detach().cpu().numpy()
		if len(vector_array) < self.CONFIG.MAX_SEQUENCE_LEN:
			vector_array = np.lib.pad(vector_array,
										((0, self.CONFIG.MAX_SEQUENCE_LEN - len(vector_array)), (0, 0)),
										"constant",
										constant_values=(1.))
		del image_tensor
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
