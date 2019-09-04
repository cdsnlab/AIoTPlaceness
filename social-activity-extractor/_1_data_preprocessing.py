# -*- coding: UTF-8 -*-
import sys
import os
import config
import csv
import pickle
import regex
import re
import numpy as np
import pandas as pd
from tqdm import tqdm


from nltk import word_tokenize
#from konlpy.tag import Okt
#import nagisa
#import jieba

from gensim.models import word2vec, Word2Vec, FastText 
from gensim.models.fasttext import load_facebook_model
from gensim.models.keyedvectors import Word2VecKeyedVectors, FastTextKeyedVectors
from gensim.test.utils import datapath
#from polyglot.detect import Detector


from util import process_text
CONFIG = config.Config
# okt=Okt()

def make_corpus(target_folder):
	print(target_folder)
	corpus_name = target_folder + '.txt'
	f_wr = open(os.path.join(CONFIG.DATA_PATH, 'corpus', corpus_name), 'w', encoding='utf-8')
	text_path = os.path.join(CONFIG.DATA_PATH, target_folder)
	text_folder_list = os.listdir(text_path)
	count = 0
	# languages_dic = dict()
	for text_folder in text_folder_list:
		# print("folder: ", text_folder)
		text_files = os.listdir(os.path.join(text_path, text_folder))
		for text_file in text_files:
			if text_file.endswith('.txt') and not text_file.endswith('_location.txt'):
				if count % 100 == 0: 
					print(count)
				with open(os.path.join(text_path, text_folder, text_file), 'r', encoding='utf-8', newline='\n') as f:
					# print("file: ", text_file)
					data = f.read()
					line = process_text(data)
					if len(line) > 0:
						f_wr.write(line + ' <EOS> <PAD>\n')
				count = count + 1
	f_wr.close()
	# csv_name = target_folder + '_meta.csv'
	# with open(os.path.join(CONFIG.CSV_PATH, csv_name), 'w', encoding='utf-8-sig', newline='') as f:
	# 	w = csv.writer(f)
	# 	for k,v in languages_dic.items():
	# 		w.writerow((k, v))
	print("completed to make corpus")

# def multi_language_tokenizer(input_line):
# 	language = Detector(input_line, quiet=True).languages[0]

# 	output_line = []
# 	if language.code == 'en':
# 		tokens = word_tokenize(input_line)
# 		for token in tokens:
# 			if token != '#':
# 				output_line.append(token)
# 	elif language.code == 'ko':
# 		tokens = okt.pos(input_line)
# 		for token in tokens:
# 			if token[1] == 'Hashtag':
# 				output_line.append(token[0][1:])
# 			elif token[1] == 'Punctuation':
# 				pass
# 			else:
# 				output_line.append(token[0])
# 	elif language.code == 'ja':
# 		tokens = nagisa.tagging(input_line)
# 		for token in tokens.words:
# 			if token != '#':
# 				output_line.append(token)
# 	elif language.code == 'zh_Hant':
# 		tokens = jieba.cut(input_line)
# 		for token in tokens:
# 			if token != '#':
# 				output_line.append(token)
# 	else:
# 		return ("", language.name)
# 	return (input_line, language.code)

def make_fasttext(target_dataset):

	corpus_path = os.path.join(CONFIG.DATASET_PATH, target_dataset, "corpus.txt")
	sentences = word2vec.LineSentence(corpus_path) 
	dimension_size = 300
	print("embedding started")
	embedding_model = FastText(sentences=sentences, size=dimension_size, window=6, min_count=5, workers=4, sg = 1) #skip-gram
	embedding_model = FastText(size=dimension_size, window=6, min_count=5, workers=4, sg = 1) #skip-gram
	embedding_model.build_vocab(sentences=sentences)
	embedding_model.train(sentences=sentences, total_examples=embedding_model.corpus_count, epochs=10)
	model_name = "FASTTEXT_"+ target_dataset + ".model"
	#pad_value = np.finfo(np.float32).eps
	pad_value = 1.
	embedding_model.wv.add("<PAD>", np.full(embedding_model.vector_size, pad_value), replace=True)
	embedding_model.wv.init_sims(replace=True)
	embedding_model.wv.save(os.path.join(CONFIG.EMBEDDING_PATH, model_name))
	print("embedding completed")

def model_to_csv(target_model):
	model_name = 'FASTTEXT_' + target_model + '.model'
	model = FastTextKeyedVectors.load(os.path.join(CONFIG.EMBEDDING_PATH,model_name))
	vocab = list(model.vocab)
	vocab_list = [x for x in vocab]
	print("vocab length: ", len(vocab_list))

	# f_csv = open(DF_PATH+'Word2VecBlog300_5_min10_mecab.csv', 'w', encoding='utf-8-sig', newline='')
	print("started to write csv")
	csv_name = target_model + '.csv'
	f_csv = open(os.path.join(CONFIG.CSV_PATH, csv_name), 'w', encoding='utf-8-sig', newline='')
	wr = csv.writer(f_csv)

	for voca in vocab_list:
		wr.writerow([voca]+model[voca].tolist())

	f_csv.close()
	print("completed to write csv")

def test_fasttext(target_model):
	model_name = 'FASTTEXT_' + target_model + '.model'
	model = FastTextKeyedVectors.load(os.path.join(CONFIG.EMBEDDING_PATH, model_name))
	pad_vector = np.full(300, np.finfo(np.float32).eps)
	# pad_vector = np.random.randn(300)
	# pad_vector = np.ones(300)
	# pad_vector = np.full(300, 100)
	# print(pad_vector)
	print(model.similar_by_word("<EOS>"))
	print(model.similar_by_vector(vector=pad_vector, topn=5))
	model.add("<PAD>", pad_vector)
	model.init_sims(replace=True)
	print(model.similar_by_vector(vector=pad_vector, topn=5))
	print(model.get_vector("<EOS>"))
	print(model.get_vector("<PAD>"))

def pickle_to_corpus(target_pickle):
	pickle_name = target_pickle + '.p'
	text_name = target_pickle + '.txt'
	f_wr = open(os.path.join(CONFIG.DATA_PATH, 'corpus', text_name), 'w', encoding='utf-8')
	import _pickle as cPickle
	count = 0
	with open(os.path.join(CONFIG.DATA_PATH, 'pickle', pickle_name), "rb") as f:
		dataset = cPickle.load(f, encoding="latin1")
		for pg in dataset[0]:
			if count % 100 == 0: 
				print(count)
			data = " ".join([dataset[3][idx] for idx in pg])
			data = data.replace("END_TOKEN", "")
			expression = re.compile('[ㄱ-ㅣ가-힣|a-zA-Z|\s]+') 
			data = [re.findall(expression, x) for x in data if x.isprintable()]
			word_list = []
			word = []
			for character in data:
				if len(character) is 0:
					continue
				if character[0] == ' ' or character[0] == 'ㅤ':
					if len(word) != 0:
						word_list.append(''.join(word))
						word = []
					else:
						continue
				else:
					if character[0].isalpha():
						character[0] = character[0].lower()
					word.append(character[0])
			line = ' '.join(word_list)
			if len(line) > 0:
				f_wr.write(line + ' <EOS>\n')
			count = count + 1
		for pg in dataset[1]:
			if count % 100 == 0: 
				print(count)
			data = " ".join([dataset[3][idx] for idx in pg])
			data = data.replace("END_TOKEN", "")
			expression = re.compile('[ㄱ-ㅣ가-힣|a-zA-Z|\s]+') 
			data = [re.findall(expression, x) for x in data if x.isprintable()]
			word_list = []
			word = []
			for character in data:
				if len(character) is 0:
					continue
				if character[0] == ' ' or character[0] == 'ㅤ':
					if len(word) != 0:
						word_list.append(''.join(word))
						word = []
					else:
						continue
				else:
					if character[0].isalpha():
						character[0] = character[0].lower()
					word.append(character[0])
			line = ' '.join(word_list)
			if len(line) > 0:
				f_wr.write(line + ' <EOS>\n')
			count = count + 1
	f_wr.close()

def make_word2vec(target_corpus):
	target_corpus_name = target_corpus + '.txt'
	corpus_path = os.path.join(CONFIG.DATA_PATH, "corpus", target_corpus_name)
	sentences = word2vec.LineSentence(corpus_path) 
	
	print("embedding started")
	embedding_model = Word2Vec(sentences, size=300, window=5, min_count=1, workers=4, sg = 1, hs=0, negative=5, sample = 0.00001, iter = 100)
	model_name = "WORD2VEC_"+ target_corpus + ".model"
	embedding_model.wv.save(os.path.join(CONFIG.EMBEDDING_PATH, model_name))
	print("embedding completed")

def test():
	# target_corpus_name = target_corpus + '.txt'
	# length_list = []
	# sentence_len = 0
	# with open(os.path.join(CONFIG.DATA_PATH, 'corpus', target_corpus_name), 'r', encoding='utf-8-sig', newline='\n') as f:
	# 	while True:
	# 		line = f.readline()
	# 		if not line: break;
	# 		length_list.append(len(line.split()))
	# 		if len(line.split()) > sentence_len:
	# 			sentence_len = len(line.split())
	# 			print(line)
	# length_array = np.array(length_list)
	# print("mean: ", np.mean(length_array))
	# print("max: ", np.max(length_array))

	full_data = []
	full_data_norm = []
	df_data = pd.read_csv(os.path.join(CONFIG.CSV_PATH, "hongdae.csv"), header=None, encoding='utf-8')
	pbar = tqdm(total=df_data.shape[0])
	for index, row in df_data.iterrows():
		pbar.update(1)
		text_data = row.iloc[1:]
		text_data = np.array(text_data, dtype=np.float32)
		text_data_norm = np.linalg.norm(text_data, axis=0, ord=2)
		full_data.append(text_data)
		full_data_norm.append(text_data_norm)
		del text_data
	pbar.close()
	full_data = np.array(full_data, dtype=np.float32)
	full_data_norm = np.array(full_data_norm, dtype=np.float32)
	print("mean: ", np.mean(full_data, axis=1))
	print("std: ", np.std(full_data, axis=1))
	print("max: ", np.max(full_data, axis=1))
	print("min: ", np.min(full_data, axis=1))
	print("norm: ", full_data_norm)

def make_fasttext_pretrained(target_corpus, pretrined_model):

	target_corpus_name = target_corpus + '.txt'
	corpus_path = os.path.join(CONFIG.DATA_PATH, "corpus", target_corpus_name)
	fb_path = os.path.join(CONFIG.EMBEDDING_PATH, "facebook", pretrined_model)
	sentences = word2vec.LineSentence(corpus_path) 
	embedding_model = load_facebook_model(fb_path) # load pretrained model
	print("embedding started")	
	embedding_model.build_vocab(sentences=sentences, update=True)
	print(embedding_model.epochs)
	print("train started")	
	embedding_model.train(sentences=sentences, total_examples=embedding_model.corpus_count, epochs=10)
	print("train completed")	
	model_name = "FASTTEXT_"+ target_corpus + "_pretrained.model"
	embedding_model.wv.save(os.path.join(CONFIG.EMBEDDING_PATH, model_name))
	print("embedding completed")

def run(option):
	if option == 0:
		make_corpus(target_folder=sys.argv[2])
	elif option == 1:
		make_fasttext(target_dataset=sys.argv[2])
	elif option == 2:
		model_to_csv(target_model=sys.argv[2])
	elif option == 3:
		test_fasttext(target_model=sys.argv[2])
	elif option == 4:
		pickle_to_corpus(target_pickle=sys.argv[2])
	elif option == 5:
		make_word2vec(target_corpus=sys.argv[2])
	elif option == 6:
		test()
	elif option == 7:
		make_fasttext_pretrained(target_corpus=sys.argv[2], pretrined_model=sys.argv[3])
	else:
		print("This option does not exist!\n")


if __name__ == '__main__':
	run(int(sys.argv[1]))



