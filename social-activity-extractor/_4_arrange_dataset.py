# -*- coding: utf-8 -*-
import os
import shutil
import config
import re
import sys
import csv
import random
import _pickle as cPickle
import numpy as np
import pandas as pd

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision.datasets.folder import pil_loader
from gensim.models.keyedvectors import FastTextKeyedVectors
from gensim.similarities.index import AnnoyIndexer

from util import process_text

CONFIG = config.Config

def copy_selected_post(target_folder):

	path_to_posts = {}
	data_path = os.path.join(CONFIG.DATA_PATH, target_folder)

	for directory in os.listdir(data_path):
		path_dir = os.path.join(data_path, directory)
		path_to_posts[directory] = []
		for file in os.listdir(path_dir):
			if file.endswith('UTC.txt'):
				path_to_posts[directory].append(file)

	print("Total # of locations: ", len(path_to_posts))

	data_path = os.path.join(CONFIG.DATA_PATH, target_folder)
	dataset_path = os.path.join(CONFIG.DATASET_PATH, target_folder)
	if not os.path.exists(dataset_path):
		os.mkdir(dataset_path)
	count = 0
	for directory, posts in path_to_posts.items():
		print(str(count), "th Location directory: ", directory)
		path_dir = os.path.join(data_path, directory)


		for file in os.listdir(path_dir):
			if file.endswith('location.txt'):
				os.remove(os.path.join(path_dir, file))
				continue
			if not file.endswith('.jpg') and not file.endswith('.txt') and not file.endswith('.json'):
				os.remove(os.path.join(path_dir, file))
				continue

		for post in tqdm(posts):
			post_name = post.replace(".txt", "")
			post_dic = {"img":[], "text":"", "json":""}
			for file in os.listdir(path_dir):
				if file.startswith(post_name):
					if file.endswith('.jpg'):
						post_dic['img'].append(file)
					elif file.endswith('.json'):
						post_dic['json'] = file
					elif file.endswith('.txt') and not file.endswith('location.txt'):
						post_dic['text'] = file
					else:
						pass

			if len(post_dic["img"]) > 0 and post_dic["text"] != "" and post_dic["json"] != "":

				with open(os.path.join(path_dir, post_dic["text"]), 'r', encoding='utf-8', newline='\n') as f:
					# print("file: ", text_file)
					data = f.read()
					line = process_text(data)
					if len(line) > 0:						
						path_to_location = os.path.join(dataset_path, directory)
						if not os.path.exists(path_to_location):
							os.mkdir(path_to_location)
						path_to_post = os.path.join(dataset_path, directory, post_name)
						if not os.path.exists(path_to_post):
							os.mkdir(path_to_post)
						shutil.move(os.path.join(path_dir, post_dic["json"]), os.path.join(path_to_post, "meta.json"))
						os.mkdir(os.path.join(path_to_post, "images"))
						for idx, img in enumerate(post_dic["img"]):
							img_name = "image_" + str(idx) + ".jpg"
							shutil.move(os.path.join(path_dir, img), os.path.join(path_to_post, "images", img_name))
						f_wr = open(os.path.join(path_to_post, "text.txt"), 'w', encoding='utf-8')
						f_wr.write(line + ' <EOS>\n')
						f_wr.close()
				f.close()
		shutil.rmtree(path_dir)
		count = count + 1

	print("Copy completed")

class last_layer(nn.Module):
	def __init__(self):
		super(last_layer, self).__init__()
		
	def forward(self, x):
		normalized_x = F.normalize(x, p=2, dim=1)
		return normalized_x

def embedding_images(target_dataset, arch, gpu):
	dataset_path = os.path.join(CONFIG.DATASET_PATH, target_dataset)
	device = torch.device(gpu)
	print("Loading embedding model...")
	embedding_model = models.__dict__[arch](pretrained=True)
	embedding_model.fc = last_layer()
	embedding_model.eval()
	embedding_model.to(device)
	print("Loading embedding model completed")
	pad_value = 0.
	embedding_path = os.path.join(dataset_path, arch)
	if not os.path.exists(embedding_path):
		os.mkdir(embedding_path)
	original_path = os.path.join(CONFIG.DATA_PATH, 'dataset', target_dataset, 'original')
	for image_path in tqdm(os.listdir(original_path)):
		with open(os.path.join(original_path, image_path), 'rb') as f:
			image_data = cPickle.load(f)
		f.close()

		image_data = torch.from_numpy(image_data).type(torch.FloatTensor).to(device)
		embedded_image = embedding_model(image_data).detach().cpu().numpy()

		if len(embedded_image) < CONFIG.MAX_SEQUENCE_LEN:
			# pad sentence with 0 if sentence length is shorter than `max_sentence_len`
			vector_array = np.lib.pad(embedded_image,
									((0, CONFIG.MAX_SEQUENCE_LEN - len(embedded_image)), (0,0)),
									"constant",
									constant_values=(pad_value))
		else:
			vector_array = embedded_image
		#vector_array = vector_array / np.linalg.norm(vector_array, axis=1, ord=2, keepdims=True)
		with open(os.path.join(embedding_path, image_path), 'wb') as f:
			cPickle.dump(vector_array, f)
		f.close()
		del image_data, embedded_image, vector_array

def embedding_text(target_dataset):
	print("Loading embedding model...")
	model_name = 'FASTTEXT_' + target_dataset + '.model'
	embedding_model = FastTextKeyedVectors.load(os.path.join(CONFIG.EMBEDDING_PATH, model_name))
	print("Loading embedding model completed")
	dataset_path = os.path.join(CONFIG.DATASET_PATH, target_dataset)
	for loc_id in tqdm(os.listdir(dataset_path)):
		path_dir = os.path.join(dataset_path, loc_id)
		for post in tqdm(os.listdir(path_dir), leave=False):
			pickle_path = os.path.join(path_dir, post, "text.p")
			with open(os.path.join(path_dir, post, "text.txt"), 'r', encoding='utf-8', newline='\n') as f:
				text_data = f.read()
				word_list = text_data.split()
				vector_list = []
				if len(word_list) > CONFIG.MAX_SENTENCE_LEN:
					# truncate sentence if sentence length is longer than `max_sentence_len`
					word_list = word_list[:CONFIG.MAX_SENTENCE_LEN]
					word_list[-1] = '<EOS>'
				else:
					word_list = word_list + ['<PAD>'] * (CONFIG.MAX_SENTENCE_LEN - len(word_list))
				for word in word_list:
					vector = embedding_model.get_vector(word)
					vector_list.append(vector)
				vector_array = np.array(vector_list, dtype=np.float32)
			f.close()
			with open(pickle_path, 'wb') as f:
				cPickle.dump(vector_array, f, protocol=-1)
			f.close()
			del text_data, word_list, vector_array

def process_dataset_images(target_dataset):

	img_transform = transforms.Compose([
					transforms.Resize(256),
					transforms.CenterCrop(224),
					transforms.ToTensor(),
					transforms.Normalize(mean=[0.485, 0.456, 0.406],
									 std=[0.229, 0.224, 0.225])
				])	
	dataset_path = os.path.join(CONFIG.DATA_PATH, 'dataset', target_dataset)
	if not os.path.exists(dataset_path):
		os.mkdir(dataset_path)
	if not os.path.exists(os.path.join(dataset_path, 'original')):
		os.mkdir(os.path.join(dataset_path, 'original'))
	df_data = pd.read_csv(os.path.join(CONFIG.TARGET_PATH, 'posts.csv'), encoding='utf-8')
	pbar = tqdm(total=df_data.shape[0])
	for index, in_row in df_data.iterrows():
		pbar.update(1)
		images = []
		for image in in_row.iloc[7:]:
			if not pd.isna(image):
				image_path = os.path.join(CONFIG.TARGET_PATH, 'original', image)
				try:
					images.append(img_transform(pil_loader(image_path)))
				except OSError as e:
					print(e)
					print(image_path)
		if len(images) > 0:
			image_data = torch.stack(images).detach().numpy()
			with open(os.path.join(dataset_path, 'original', in_row.iloc[1]+'.p'), 'wb') as f:
				cPickle.dump(image_data, f)
			f.close()
			del image_data
	pbar.close()

def process_dataset_text(target_dataset):
	dataset_path = os.path.join(CONFIG.DATASET_PATH, target_dataset)
	if not os.path.exists(dataset_path):
		os.mkdir(dataset_path)
	df_data = pd.read_csv(os.path.join(CONFIG.TARGET_PATH, 'SEOUL_SUBWAY_DATA-3.csv'), index_col=1, encoding='utf-8-sig')
	df_data.index.name = "short_code"
	df_data.sort_index(inplace=True)
	print("tokenizing sentences...")
	pbar = tqdm(total=df_data.shape[0])
	shortcode_list = []
	word_list_list = []
	image_list = []
	for index, in_row in df_data.iterrows():
		pbar.update(1)
		if pd.isna(in_row.iloc[1]):
			continue
		word_list = process_text(in_row.iloc[1])
		if len(word_list) > 0:
			shortcode_list.append(index)
			word_list_list.append(word_list)
			image_list.append(in_row.iloc[6])
	pbar.close()
	print("counting frequencies...")
	frequency = {}
	pbar = tqdm(total=len(word_list_list))
	for word_list in word_list_list:
		pbar.update(1)
		for word in word_list:
			count = frequency.get(word, 0)
			frequency[word] = count + 1
	pbar.close()
	count = 0
	for word in frequency:
		if frequency[word] >= CONFIG.MIN_WORD_COUNT:
			count = count + 1
	print("words more then min_count: " + str(count))
	print("convert too few words to UNK token...")
	pbar = tqdm(total=len(word_list_list))
	processed_word_list_list = []
	for word_list in word_list_list:
		pbar.update(1)
		processed_word_list = []
		for word in word_list:
			if frequency[word] < CONFIG.MIN_WORD_COUNT:
				processed_word_list.append('UNK')
			else:
				processed_word_list.append(word)
		processed_word_list_list.append(processed_word_list)
	pbar.close()
	print("making corpus and csv files...")
	f_csv = open(os.path.join(dataset_path, 'posts.csv'), 'w', encoding='utf-8-sig')
	f_corpus = open(os.path.join(dataset_path, 'corpus.txt'), 'w', encoding='utf-8')
	wr = csv.writer(f_csv)
	pbar = tqdm(total=len(processed_word_list_list))
	for index in range(len(processed_word_list_list)):
		pbar.update(1)
		sentence = ' '.join(processed_word_list_list[index])
		if len(sentence) > 0:
			out_row = []
			out_row.append(shortcode_list[index])
			out_row.append(sentence + ' <EOS>')
			out_row.append(image_list[index])
			wr.writerow(out_row)
			f_corpus.write(sentence + ' <EOS>\n')
	pbar.close()
	f_csv.close()
	f_corpus.close()

def test(target_dataset):
	# toy_path = os.path.join(CONFIG.DATASET_PATH, 'instagram0830')
	# full_data = []
	# full_data_norm = []
	# for image_path in os.listdir(os.path.join(toy_path, 'resnext101_32x8d')):
	# 	with open(os.path.join(toy_path, 'resnext101_32x8d', image_path), "rb") as f:
	# 		image_data = cPickle.load(f)
	# 		# print(data)
	# 		# print(np.max(data))
	# 		# print(np.min(data))
	# 		# print(np.mean(data))
	# 		# print(data.shape)
	# 	full_data.append(image_data)
	# 	image_data_norm = np.linalg.norm(image_data, axis=1, ord=2)
	# 	full_data_norm.append(image_data_norm)
	# #df_data = pd.read_csv(os.path.join(CONFIG.DATASET_PATH, target_dataset, 'posts.csv'), header=None, encoding='utf-8')
	# #print(df_data)

	# full_data = np.array(full_data, dtype=np.float32)
	# full_data_norm = np.array(full_data_norm, dtype=np.float32)
	# temp = np.mean(np.mean(full_data, axis=2), axis=1)
	# print(temp.shape)
	# print("mean: ", np.mean(np.mean(full_data, axis=2), axis=1))
	# print("std: ", np.mean(np.std(full_data, axis=2), axis=1))
	# print("max: ", np.mean(np.max(full_data, axis=2), axis=1))
	# print("min: ", np.mean(np.min(full_data, axis=2), axis=1))
	# print("norm: ", full_data_norm)


	df_data = pd.read_csv(os.path.join(CONFIG.TARGET_PATH, 'SEOUL_SUBWAY_DATA-3.csv'), index_col=1, encoding='utf-8-sig')
	df_data.index.name = "short_code"
	df_data.sort_index(inplace=True)
	index_list = []
	before_list = []
	for index, in_row in df_data.iloc[:11].iterrows():
		if not pd.isna(in_row.iloc[1]):
			index_list.append(index)
			before_list.append(in_row.iloc[1])

	dataset_path = os.path.join(CONFIG.DATASET_PATH, target_dataset)
	toy_path = os.path.join(CONFIG.DATASET_PATH, 'toy')
	df_data = pd.read_csv(os.path.join(dataset_path, 'posts.csv'), index_col=0, header=None, encoding='utf-8')
	after_list = []
	for index, in_row in df_data.loc[index_list].iterrows():
		after_list.append(in_row.iloc[0])
	index_array = np.array(index_list)
	before_array = np.array(before_list)
	after_array = np.array(after_list)
	data_array = np.stack([before_list, after_list], axis=1)
	result_df = pd.DataFrame(data=data_array, index=index_array, columns=["before", "after"])
	result_df.index.name = "short_code"
	result_df.sort_index(inplace=True)
	result_df.to_csv('temp.csv', encoding='utf-8-sig')


	# dataset_path = os.path.join(CONFIG.DATA_PATH, 'dataset', target_dataset)
	# with open(os.path.join(dataset_path, 'resize224', 'BsFudrehNdL.p'), 'rb') as f:
	# 	image_data = cPickle.load(f)
	# f.close()
	# print(image_data)
	# imgseq_tensor = torch.from_numpy(image_data).type(torch.FloatTensor)
	# save_image(imgseq_tensor, './result/temp.png', nrow=5, padding=0)
	

def make_toy_dataset(target_dataset):
	dataset_path = os.path.join(CONFIG.DATASET_PATH, target_dataset)
	toy_path = os.path.join(CONFIG.DATASET_PATH, 'toy')
	df_data = pd.read_csv(os.path.join(dataset_path, 'posts.csv'), header=None, encoding='utf-8')
	short_codes = []
	for index, row in df_data.iterrows():
		short_codes.append(row)
	toy_codes = random.sample(short_codes, k=500)

	f_csv = open(os.path.join(toy_path, 'posts.csv'), 'w', encoding='utf-8')
	wr = csv.writer(f_csv)
	for row in toy_codes:
		wr.writerow(row)
		short_code = row.iloc[0] + '.p'
		shutil.copy2(os.path.join(dataset_path, 'resnet152', short_code), os.path.join(toy_path, 'resnet152', short_code))

def process_resize_imgseq(target_dataset):

	img_transform = transforms.Compose([
					transforms.Resize(224),
					transforms.CenterCrop(224),
					transforms.ToTensor()
					])	
	dataset_path = os.path.join(CONFIG.DATA_PATH, 'dataset', target_dataset)
	if not os.path.exists(dataset_path):
		os.mkdir(dataset_path)
	if not os.path.exists(os.path.join(dataset_path, 'resize224')):
		os.mkdir(os.path.join(dataset_path, 'resize224'))
	df_data = pd.read_csv(os.path.join(CONFIG.TARGET_PATH, 'posts.csv'), encoding='utf-8')
	pbar = tqdm(total=df_data.shape[0])
	for index, in_row in df_data.iterrows():
		pbar.update(1)
		images = []
		for image in in_row.iloc[7:]:
			if not pd.isna(image):
				image_path = os.path.join(CONFIG.TARGET_PATH, 'original', image)
				try:
					images.append(img_transform(pil_loader(image_path)))
				except OSError as e:
					print(e)
					print(image_path)
		if len(images) > 0:
			image_data = torch.stack(images).detach().numpy()
			with open(os.path.join(dataset_path, 'resize224', in_row.iloc[1]+'.p'), 'wb') as f:
				cPickle.dump(image_data, f)
			f.close()
			del image_data
	pbar.close()

def drop_non_korean_images(target_dataset):	
	dataset_path = os.path.join(CONFIG.DATASET_PATH, target_dataset)
	df_data = pd.read_csv(os.path.join(dataset_path, 'posts.csv'), header=None, encoding='utf-8')

	short_codes = []
	pbar = tqdm(total=df_data.shape[0])
	for index, row in df_data.iterrows():
		short_codes.append(row[0])
		pbar.update(1)
	pbar.close()
	image_nas_path = os.path.join(CONFIG.DATA_PATH, 'dataset', target_dataset)
	image_dir = os.path.join(image_nas_path, 'original')
	for image_path in tqdm(os.listdir(image_dir)):
		short_code = image_path.replace('.p', '')
		if short_code not in short_codes:
			os.remove(os.path.join(image_dir, image_path))

def process_dataset_image(target_dataset):

	img_transform = transforms.Compose([
					transforms.Resize(256),
					transforms.CenterCrop(224),
					transforms.ToTensor(),
					transforms.Normalize(mean=[0.485, 0.456, 0.406],
									 std=[0.229, 0.224, 0.225])
				])	
	dataset_path = os.path.join(CONFIG.DATASET_PATH, target_dataset)
	df_data = pd.read_csv(os.path.join(dataset_path, 'posts.csv'), header=None, encoding='utf-8')
	print(df_data[:5])

	# if not os.path.exists(dataset_path):
	# 	os.mkdir(dataset_path)
	# if not os.path.exists(os.path.join(dataset_path, 'original')):
	# 	os.mkdir(os.path.join(dataset_path, 'original'))
	# df_data = pd.read_csv(os.path.join(CONFIG.TARGET_PATH, 'posts.csv'), encoding='utf-8')
	# pbar = tqdm(total=df_data.shape[0])
	# for index, in_row in df_data.iterrows():
	# 	pbar.update(1)
	# 	images = []
	# 	for image in in_row.iloc[7:]:
	# 		if not pd.isna(image):
	# 			image_path = os.path.join(CONFIG.TARGET_PATH, 'original', image)
	# 			try:
	# 				images.append(img_transform(pil_loader(image_path)))
	# 			except OSError as e:
	# 				print(e)
	# 				print(image_path)
	# 	if len(images) > 0:
	# 		image_data = torch.stack(images).detach().numpy()
	# 		with open(os.path.join(dataset_path, 'original', in_row.iloc[1]+'.p'), 'wb') as f:
	# 			cPickle.dump(image_data, f)
	# 		f.close()
	# 		del image_data
	# pbar.close()

def normalized_and_pca(target_csv):

	df_data = pd.read_csv(os.path.join(CONFIG.CSV_PATH, target_csv), encoding='utf-8')
	print(df_data[:5])
	df_normalized = df_data.div((np.sqrt(np.sum(np.square(df_data), axis=1))), axis=0)
	pca_normalized = PCA(n_components=300, random_state=42)
	df_pca_normalized = pd.DataFrame(pca_normalized.fit_transform(df_normalized))
	df_pca_normalized.columns = ['PC' + str(i) for i in range(df_pca_normalized.shape[1])]
	df_pca_normalized.index = df_normalized.index
	print(df_pca_normalized[:5])
	df_pca_normalized.to_csv(os.path.join(CONFIG.CSV_PATH, 'pca_normalized_' + target_csv), encoding='utf-8-sig')


def run(option):
	if option == 0:
		copy_selected_post(target_folder=sys.argv[2])
	elif option == 1:
		embedding_images(target_dataset=sys.argv[2], arch=sys.argv[3], gpu=sys.argv[4])
	elif option == 2:
		embedding_text(target_dataset=sys.argv[2])
	elif option == 3:
		process_dataset_imgseq(target_dataset=sys.argv[2])
	elif option == 4:
		process_dataset_text(target_dataset=sys.argv[2])
	elif option == 5:
		test(target_dataset=sys.argv[2])
	elif option == 6:
		make_toy_dataset(target_dataset=sys.argv[2])
	elif option == 7:
		process_resize_images(target_dataset=sys.argv[2])
	elif option == 8:
		drop_non_korean_images(target_dataset=sys.argv[2])
	elif option == 9:
		process_dataset_image(target_dataset=sys.argv[2])
	elif option == 10:
		normalized_and_pca(target_csv=sys.argv[2])
	else:
		print("This option does not exist!\n")


if __name__ == '__main__':
	run(int(sys.argv[1]))