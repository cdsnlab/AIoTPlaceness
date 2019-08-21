# -*- coding: UTF-8 -*-
import sys
import os
import config
import csv
import pickle
import regex
import numpy as np


from nltk import word_tokenize
from konlpy.tag import Okt
import nagisa
import jieba

from gensim.models import word2vec, Word2Vec, FastText 
from gensim.models.keyedvectors import Word2VecKeyedVectors, FastTextKeyedVectors
from gensim.test.utils import datapath
from polyglot.detect import Detector

CONFIG = config.Config
okt=Okt()

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
					data = data.replace("#", " ")
					data = data.replace("\n", " ")
					data = [regex.findall('[\p{Hangul}|\p{Latin}|\s]+', x) for x in data if x.isprintable()]
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
						f_wr.write(line + '\n')
				count = count + 1
	f_wr.close()
	# csv_name = target_folder + '_meta.csv'
	# with open(os.path.join(CONFIG.CSV_PATH, csv_name), 'w', encoding='utf-8-sig', newline='') as f:
	# 	w = csv.writer(f)
	# 	for k,v in languages_dic.items():
	# 		w.writerow((k, v))
	print("completed to make corpus")

def multi_language_tokenizer(input_line):
	language = Detector(input_line, quiet=True).languages[0]

	output_line = []
	if language.code == 'en':
		tokens = word_tokenize(input_line)
		for token in tokens:
			if token != '#':
				output_line.append(token)
	elif language.code == 'ko':
		tokens = okt.pos(input_line)
		for token in tokens:
			if token[1] == 'Hashtag':
				output_line.append(token[0][1:])
			elif token[1] == 'Punctuation':
				pass
			else:
				output_line.append(token[0])
	elif language.code == 'ja':
		tokens = nagisa.tagging(input_line)
		for token in tokens.words:
			if token != '#':
				output_line.append(token)
	elif language.code == 'zh_Hant':
		tokens = jieba.cut(input_line)
		for token in tokens:
			if token != '#':
				output_line.append(token)
	else:
		return ("", language.name)
	return (input_line, language.code)

def make_fasttext(target_corpus):
	target_corpus_name = target_corpus + '.txt'
	corpus_path = os.path.join(CONFIG.DATA_PATH, "corpus", target_corpus_name)
	sentences = word2vec.LineSentence(corpus_path) 
	
	print("embedding started")
	embedding_model = FastText(sentences, size=300, window=6, min_count=5, workers=4, sg = 1, iter=100) #skip-gram
	model_name = "FASTTEXT_"+ target_corpus + ".model"
	embedding_model.wv.save(os.path.join(CONFIG.EMBEDDING_PATH, model_name))
	print("embedding completed")

def model_to_csv(target_model, model_type):
	if model_type is 0:
		model_name = 'FASTTEXT_' + target_model + '.model'
		model = FastTextKeyedVectors.load(os.path.join(CONFIG.EMBEDDING_PATH,model_name))
	else:
		model_name = 'WORD2VEC_' + target_model + '.model'
		model = Word2VecKeyedVectors.load(os.path.join(CONFIG.EMBEDDING_PATH,model_name))
	vocab = list(model.vocab)
	vocab_list = [x for x in vocab]

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
	print(model.get_vector('hotelreview'))
	print(model.most_similar('hotel'))

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
			data = [regex.findall('[\p{Hangul}|\p{Latin}|\s]+', x) for x in data if x.isprintable()]
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
				f_wr.write(line + '\n')
			count = count + 1
		for pg in dataset[1]:
			if count % 100 == 0: 
				print(count)
			data = " ".join([dataset[3][idx] for idx in pg])
			data = data.replace("END_TOKEN", "")
			data = [regex.findall('[\p{Hangul}|\p{Latin}|\s]+', x) for x in data if x.isprintable()]
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
				f_wr.write(line + '\n')
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

def test(target_corpus):
	target_corpus_name = target_corpus + '.txt'
	length_list = []
	with open(os.path.join(CONFIG.DATA_PATH, 'corpus', target_corpus_name), 'r', encoding='utf-8-sig', newline='\n') as f:
		while True:
			line = f.readline()
			if not line: break;
			length_list.append(len(line.split()))
	length_array = np.array(length_list)
	print("mean: ", np.mean(length_array))
	print("max: ", np.max(length_array))

def run(option):
	if option == 0:
		make_corpus(target_folder=sys.argv[2])
	elif option == 1:
		make_fasttext(target_corpus=sys.argv[2])
	elif option == 2:
		model_to_csv(target_model=sys.argv[2], model_type=sys.argv[3])
	elif option == 3:
		test_fasttext(target_model=sys.argv[2])
	elif option == 4:
		pickle_to_corpus(target_pickle=sys.argv[2])
	elif option == 5:
		make_word2vec(target_corpus=sys.argv[2])
	elif option == 6:
		test(target_corpus=sys.argv[2])
	else:
		print("This option does not exist!\n")


if __name__ == '__main__':
	run(int(sys.argv[1]))



