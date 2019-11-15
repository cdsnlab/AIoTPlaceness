import argparse

from sklearn import svm
from sklearn.metrics import accuracy_score, normalized_mutual_info_score, f1_score
from sklearn.model_selection import KFold

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
	args = parser.parse_args()

	get_latent(args)

def get_latent(args):

	df_data = pd.read_csv(os.path.join(CONFIG.DATASET_PATH, args.target_dataset, 'posts.csv'), index_col=0, header=None, encoding='utf-8-sig')
	df_label = pd.read_csv(os.path.join(CONFIG.CSV_PATH, args.label_csv), index_col=0, encoding='utf-8-sig')
	df_data = df_data.loc[df_label.index]

	short_code_array = np.array(df_data.index)
	row_array = np.array(df_data[1])
	category_array = np.array(df_label['category'])

	tf_vectorizer = TfidfVectorizer(ngram_range=(1, 2)).fit(row_array)

	kf = KFold(n_splits=5, shuffle=True, random_state=42)
	acc_list = []
	nmi_list = []
	f_1_list = []
	kf_count = 0
	for train_index, test_index in kf.split(row_array):
		print("Current fold: ", kf_count)
		X_train, X_test = row_array[train_index], row_array[test_index]
		X_train_tfidf = tf_vectorizer.transform(X_train)
		X_test_tfidf = tf_vectorizer.transform(X_test)
		Y_train, Y_test = category_array[train_index], category_array[test_index]
		clf = svm.LinearSVC()
		clf.fit(X_train_tfidf, Y_train)
		test_pred = clf.predict(X_test_tfidf)
		test_acc = accuracy_score(Y_test, test_pred)
		test_nmi = normalized_mutual_info_score(Y_test, test_pred, average_method='geometric')
		test_f_1 = f1_score(Y_test, test_pred, average='micro')
		print("#Test acc: %.4f, Test nmi: %.4f, Test f_1: %.4f" % (
			test_acc, test_nmi, test_f_1))
		acc_list.append(test_acc)
		nmi_list.append(test_nmi)
		f_1_list.append(test_f_1)
		kf_count = kf_count + 1
	print("#Average acc: %.4f, Average nmi: %.4f, Average f_1: %.4f" % (
		np.mean(acc_list), np.mean(nmi_list), np.mean(f_1_list)))


	# print(row_array[:5])
	# print(row_array.shape)
	# csv_name = 'text_tfidf_' + args.target_dataset + '.csv'



	# result_df = pd.DataFrame(data=row_array, index=short_code_array, columns=[i for i in range(embedding_model.vector_size)])
	# result_df.index.name = "short_code"
	# result_df.sort_index(inplace=True)
	# result_df.to_csv(os.path.join(CONFIG.CSV_PATH, csv_name), encoding='utf-8-sig')
	print("Finish!!!")



if __name__ == '__main__':
	main()