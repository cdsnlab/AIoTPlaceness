import argparse

from sklearn import svm
from sklearn.metrics import accuracy_score, normalized_mutual_info_score, f1_score
from sklearn.model_selection import StratifiedKFold

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
from sklearn.feature_extraction.text import TfidfVectorizer

CONFIG = config.Config


def main():
	parser = argparse.ArgumentParser(description='text convolution-deconvolution auto-encoder model')
	# data
	parser.add_argument('-target_dataset', type=str, default='seoul_subway', help='folder name of target dataset')
	parser.add_argument('-label_csv', type=str, default='category_label.csv', help='folder name of target dataset')
	parser.add_argument('-sampled_n', type=int, default=None, help='number of fold')
    parser.add_argument('-start_fold', type=int, default=0, help='fold for start')
	parser.add_argument('-fold', type=int, default=5, help='number of fold')
	args = parser.parse_args()

	get_latent(args)

def get_latent(args):

	df_data = pd.read_csv(os.path.join(CONFIG.DATASET_PATH, args.target_dataset, 'posts.csv'), index_col=0, header=None, encoding='utf-8-sig')
	#df_label = pd.read_csv(os.path.join(CONFIG.CSV_PATH, args.label_csv), index_col=0, encoding='utf-8-sig')
	#df_data = df_data.loc[df_label.index]

	acc_list = []
	nmi_list = []
	f_1_list = []
	kf_count = 0
	for fold_idx in range(args.start_fold, args.fold):
		print("Current fold: ", kf_count)
		df_train = pd.read_csv(os.path.join(CONFIG.CSV_PATH, "train_" + str(fold_idx) + "_" + args.label_csv),
							  index_col=0,
							  encoding='utf-8-sig')
		if args.sampled_n is not None:
			df_train = df_train.sample(n=args.sampled_n, random_state=42)
		df_test = pd.read_csv(os.path.join(CONFIG.CSV_PATH, "test_" + str(fold_idx) + "_" + args.label_csv),
							  index_col=0,
							  encoding='utf-8-sig')
		X_train, X_test = np.array(df_data.loc[df_train.index][1]), np.array(df_data.loc[df_test.index][1])
		tf_vectorizer = TfidfVectorizer(ngram_range=(1, 2)).fit(X_train)
		X_train_tfidf = tf_vectorizer.transform(X_train)
		X_test_tfidf = tf_vectorizer.transform(X_test)
		#Y_train, Y_test = np.array(df_label.loc[df_train.index]['category']), np.array(df_label.loc[df_test.index]['category'])
		Y_train, Y_test = np.array(df_train['category']), np.array(df_test['category'])
		clf = svm.LinearSVC()
		clf.fit(X_train_tfidf, Y_train)
		test_pred = clf.predict(X_test_tfidf)
		test_acc = accuracy_score(Y_test, test_pred)
		test_nmi = normalized_mutual_info_score(Y_test, test_pred, average_method='geometric')
		test_f_1 = f1_score(Y_test, test_pred, average='macro')
		print("#Test acc: %.4f, Test nmi: %.4f, Test f_1: %.4f" % (
			test_acc, test_nmi, test_f_1))
		acc_list.append(test_acc)
		nmi_list.append(test_nmi)
		f_1_list.append(test_f_1)
		kf_count = kf_count + 1
	print("#Average acc: %.4f, Average nmi: %.4f, Average f_1: %.4f" % (
		np.mean(acc_list), np.mean(nmi_list), np.mean(f_1_list)))

	print("Finish!!!")



if __name__ == '__main__':
	main()