# -*- coding: utf-8 -*-
import os
import collections
import shutil
import config
import re
import sys
import csv
import time
import numpy as np
import pandas as pd
from sklearn.cluster import SpectralClustering, AffinityPropagation, AgglomerativeClustering, KMeans, DBSCAN, OPTICS

CONFIG = config.Config

def clustering_dbscan(target_dataset):
	df_data = pd.read_csv(os.path.join(CONFIG.CSV_PATH, 'latent_' + target_dataset + '.csv'), index_col=0, header=None, encoding='utf-8-sig')
	df_data.index.name = 'short_code'
	print(df_data.iloc[:100])

	tsne_pca = pd.read_csv(os.path.join(CONFIG.CSV_PATH, 'tsne_' + target_dataset + '.csv'), index_col=0, header=0, encoding='utf-8-sig')
	tsne_pca = tsne_pca.iloc[1:]
	tsne_pca.index.name = 'short_code'
	print(tsne_pca.iloc[:100])

	start_time = time.time()
	clustering = DBSCAN(eps=0.3, min_samples=100).fit(df_data)
	print("time elapsed: " + str(time.time()-start_time))
	print(clustering.labels_)
	count_percentage(clustering.labels_)
	cluster_list = np.array(clustering.labels_).tolist()
	tsne_pca['cluster'] = clustering.labels_
	tsne_pca.to_csv(os.path.join(CONFIG.CSV_PATH, 'clustered_dbscan_' + target_dataset + '.csv'), encoding='utf-8-sig')

def clustering_optics(target_dataset):
	df_data = pd.read_csv(os.path.join(CONFIG.CSV_PATH, 'latent_' + target_dataset + '.csv'), index_col=0, header=None, encoding='utf-8-sig')
	df_data.index.name = 'short_code'
	print(df_data.iloc[:100])

	tsne_pca = pd.read_csv(os.path.join(CONFIG.CSV_PATH, 'tsne_' + target_dataset + '.csv'), index_col=0, header=0, encoding='utf-8-sig')
	tsne_pca = tsne_pca.iloc[1:]
	tsne_pca.index.name = 'short_code'
	print(tsne_pca.iloc[:100])

	start_time = time.time()
	clustering = OPTICS(min_samples=5).fit(df_data)
	print("time elapsed: " + str(time.time()-start_time))
	print(clustering.labels_)
	count_percentage(clustering.labels_)
	cluster_list = np.array(clustering.labels_).tolist()
	tsne_pca['cluster'] = clustering.labels_
	tsne_pca.to_csv(os.path.join(CONFIG.CSV_PATH, 'clustered_optics_' + target_dataset + '.csv'), encoding='utf-8-sig')

def clustering_spectral(target_dataset, num_clusters):
	df_data = pd.read_csv(os.path.join(CONFIG.CSV_PATH, 'latent_' + target_dataset + '.csv'), index_col=0, header=None, encoding='utf-8-sig')
	df_data.index.name = 'short_code'
	print(df_data.iloc[:100])

	tsne_pca = pd.read_csv(os.path.join(CONFIG.CSV_PATH, 'tsne_' + target_dataset + '.csv'), index_col=0, header=0, encoding='utf-8-sig')
	tsne_pca = tsne_pca.iloc[1:]
	tsne_pca.index.name = 'short_code'
	print(tsne_pca.iloc[:100])

	start_time = time.time()
	clustering = SpectralClustering(n_clusters=int(num_clusters), random_state=42, n_jobs=4).fit(df_data)
	print("time elapsed: " + str(time.time()-start_time))
	print(clustering.labels_)
	count_percentage(clustering.labels_)
	cluster_list = np.array(clustering.labels_).tolist()
	tsne_pca['cluster'] = clustering.labels_
	tsne_pca.to_csv(os.path.join(CONFIG.CSV_PATH, 'clustered_spectral_' + target_dataset + '.csv'), encoding='utf-8-sig')

def count_percentage(cluster_labels):
	count = collections.Counter(cluster_labels)
	for k in count:
		print("cluster {} : {:.2%}".format(str(k), count[k]/len(cluster_labels)))


def run(option): 
	if option == 0:
		clustering_dbscan(target_dataset=sys.argv[2])
	if option == 1:
		clustering_optics(target_dataset=sys.argv[2])
	if option == 2:
		clustering_spectral(target_dataset=sys.argv[2], num_clusters=sys.argv[3])
	else:
		print("This option does not exist!\n")


if __name__ == '__main__':
	run(int(sys.argv[1]))