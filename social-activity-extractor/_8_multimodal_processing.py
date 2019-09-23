# -*- coding: utf-8 -*-
import os
import collections
import shutil
import config
import re
import sys
import csv
import time
import random
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.manifold import TSNE
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.cluster import Birch, SpectralClustering, AffinityPropagation, AgglomerativeClustering, KMeans, DBSCAN, OPTICS

CONFIG = config.Config

num_cluster = 12
def do_clustering(target_csv, cluster_method):
	df_data = pd.read_csv(os.path.join(CONFIG.CSV_PATH, target_csv + '.csv'), index_col=0, header=0, encoding='utf-8-sig')
	df_data.index.name = 'short_code'
	print(df_data.iloc[:100])
	print(df_data.shape)

	start_time = time.time()
	if cluster_method == 0:
		clustering = DBSCAN(eps=0.3, min_samples=20)
		clustering.fit(df_data)
		csv_name = 'clustered_dbscan_' + target_dataset + '.csv'
	elif cluster_method == 1:
		clustering = OPTICS(min_samples=20)
		clustering.fit(df_data)
		csv_name = 'clustered_optics_' + target_dataset + '.csv'
	elif cluster_method == 2:
		clustering = AgglomerativeClustering(n_clusters=num_cluster)
		clustering.fit(df_data)
		csv_name = 'clustered_ward_' + target_dataset + '.csv'
	elif cluster_method == 3:
		clustering = AgglomerativeClustering(affinity='cosine', linkage='complete', n_clusters=num_cluster)
		clustering.fit(df_data)
		csv_name = 'clustered_agglo_complete_' + target_dataset + '.csv'
	elif cluster_method == 4:
		clustering = Birch(n_clusters=num_cluster)
		clustering.fit(df_data)
		csv_name = 'clustered_birch_' + target_dataset + '.csv'
	elif cluster_method == 5:
		clustering = KMeans(n_clusters=num_cluster)
		clustering.fit(df_data)
		csv_name = 'clustered_kmeans_' + target_dataset + '.csv'
	print("time elapsed for clustering: " + str(time.time()-start_time))
	print(clustering.get_params())
	print(clustering.labels_)
	count_percentage(clustering.labels_)
	result_df = pd.DataFrame(data=clustering.labels_, index=df_data.index, columns=['cluster'])


	start_time = time.time()
	print("calinski_harabasz_score: ", calinski_harabasz_score(df_data, result_df['cluster'].squeeze()))
	print("silhouette_score: ", silhouette_score(df_data, result_df['cluster'].squeeze()))
	print("davies_bouldin_score: ", davies_bouldin_score(df_data, result_df['cluster'].squeeze()))
	print("time elapsed for scoring: " + str(time.time()-start_time))
	result_df.to_csv(os.path.join(CONFIG.CSV_PATH, csv_name), encoding='utf-8-sig')

def count_percentage(cluster_labels):
	count = collections.Counter(cluster_labels)
	for k in count:
		print("cluster {} : {:.2%}".format(str(k), count[k]/len(cluster_labels)))

def do_spectral_clustering(target_csv):
	df_data = pd.read_csv(os.path.join(CONFIG.CSV_PATH, target_csv + '.csv'), index_col=0, header=0, encoding='utf-8-sig')
	df_data.index.name = 'short_code'
	print(df_data.iloc[:100])
	print(df_data.shape)

	start_time = time.time()
	ds_data = df_data.sample(frac=0.5)
	ns_data = df_data.loc[set(df_data.index) - set(ds_data.index)]
	clustering = SpectralClustering(n_clusters=num_cluster, random_state=42, assign_labels='discretize')
	clustering.fit(ds_data)

	print("time elapsed for clustering: " + str(time.time()-start_time))
	print(clustering.get_params())
	print(clustering.labels_)
	count_percentage(clustering.labels_)
	start_time = time.time()
	result_ds = pd.DataFrame(data=clustering.labels_, index=ds_data.index, columns=['cluster'])
	ns_label = clustering.fit_predict(ns_data)
	result_ns = pd.DataFrame(data=ns_label, index=ns_data.index, columns=['cluster'])
	result_df = pd.concat([result_ds, result_ns])
	result_df.sort_index(inplace=True)
	print("time elapsed for prediction: " + str(time.time()-start_time))


	start_time = time.time()
	print("calinski_harabasz_score: ", calinski_harabasz_score(df_data, result_df['cluster'].squeeze()))
	print("silhouette_score: ", silhouette_score(df_data, result_df['cluster'].squeeze()))
	print("davies_bouldin_score: ", davies_bouldin_score(df_data, result_df['cluster'].squeeze()))
	print("time elapsed for scoring: " + str(time.time()-start_time))
	result_df.to_csv(os.path.join(CONFIG.CSV_PATH, 'clustered_spectral_' + target_csv + '.csv'), encoding='utf-8-sig')

sample_length = 10
def sample_from_cluster(target_dataset, target_clustering):

	df_clustered = pd.read_csv(os.path.join(CONFIG.CSV_PATH, 'clustered_' + target_clustering + '_' + target_dataset + '.csv'), index_col=0, header=0, encoding='utf-8-sig')
	df_clustered.index.name = 'short_code'
	print(df_clustered.iloc[:100])
	print(df_clustered.shape)
	num_cluster = np.max(df_clustered, axis=0)[0] + 1

	result_path = os.path.join(CONFIG.RESULT_PATH, target_dataset)
	if not os.path.exists(result_path):
		os.mkdir(result_path)
	result_path = os.path.join(result_path, target_clustering)
	if not os.path.exists(result_path):
		os.mkdir(result_path)
	for cluster_id in range(num_cluster):
		cluster_path = os.path.join(result_path, str(cluster_id))
		if not os.path.exists(cluster_path):
			os.mkdir(cluster_path)

	print("making cluster dict...")
	cluster_dict = {i: [] for i in range(num_cluster)}
	pbar = tqdm(total=df_clustered.shape[0])
	for index, row in df_clustered.iterrows():
		cluster_dict[row[0]].append(index)
		pbar.update(1)
	pbar.close()

	print("making sampled short_code dict...")
	short_code_dict = {}
	pbar = tqdm(total=len(cluster_dict))
	for key, value in cluster_dict.items():
		if len(value) > sample_length:
			sampled = random.sample(value, sample_length)
		else:
			sampled = value
		for short_code in sampled:
			short_code_dict[short_code] = key
		pbar.update(1)
	pbar.close()

	print("copying sampled posts...")
	df_original = pd.read_csv(os.path.join(CONFIG.TARGET_PATH, 'posts.csv'), encoding='utf-8-sig')
	pbar = tqdm(total=df_original.shape[0])
	for index, row in df_original.iterrows():
		if row[1] in short_code_dict:
			cluster_id = short_code_dict[row[1]]
			post_path = os.path.join(result_path, str(cluster_id), row[1])
			if not os.path.exists(post_path):
				os.mkdir(post_path)
			f_wr = open(os.path.join(post_path, 'caption.txt'), 'w', encoding='utf-8')
			f_wr.write(row[2])
			f_wr.close()
			for image in row[7:]:
				if not pd.isna(image):
					shutil.copy2(os.path.join(CONFIG.TARGET_PATH, 'original', image), os.path.join(post_path, image))
		pbar.update(1)
	pbar.close()


def apply_tsne(target_csv=sys.argv[2]):
	df_pca_data = pd.read_csv(os.path.join(CONFIG.CSV_PATH, 'pca_' + target_csv + '.csv'), index_col=0, header=0, encoding='utf-8-sig')
	lr_list = [20, 100, 200]
	perp_list = [50, 200, 500, 1000, 2000]
	for lr in lr_list:
		for perp in perp_list:     
			start_time = time.time()
			#tsne_pca = do_tsne(TSNE(n_components=2, perplexity=50, early_exaggeration=12.0, learning_rate=100, n_iter=5000, random_state=42, verbose=1), df_pca_data)
			tsne_pca = do_tsne(TSNE(n_components=2, perplexity=perp, learning_rate=lr, n_iter=2000, random_state=42), df_pca_data)
			print("time elapsed: " + str(time.time()-start_time))
			scatterplot_pointlabels(tsne_pca, 0.2)
			plt.title('t-SNE on PCA data lr: ' + str(lr) + ' perp: ' + str(perp) )
			plt.savefig(os.path.join(CONFIG.SVG_PATH, 'tsne_pca_' + target_csv + '_' + str(lr) + '_' + str(perp) + '.svg'))
			tsne_pca.to_csv(os.path.join(CONFIG.CSV_PATH, 'tsne_pca_' + target_csv + '_' + str(lr) + '_' + str(perp) + '.csv'), encoding='utf-8-sig')

def do_tsne(tsne_object, data_to_pass):
    data_tsne = pd.DataFrame(tsne_object.fit_transform(data_to_pass))
    data_tsne.index = data_to_pass.index
    data_tsne.columns = [['tsne1', 'tsne2']]
    return data_tsne

def scatterplot_pointlabels(df_twocols, markersize=None):
    # basic scatterplot
    fig = plt.figure()
    plt.plot(df_twocols.iloc[:, 0], df_twocols.iloc[:, 1], marker='.', linestyle='None', markersize=markersize)

def run(option): 
	if option == 0:
		do_clustering(target_csv=sys.argv[2], cluster_method=int(sys.argv[3]))
	elif option == 1:
		do_spectral_clustering(target_csv=sys.argv[2])
	elif option == 2:
		sample_from_cluster(target_dataset=sys.argv[2], target_clustering=sys.argv[3])
	elif option == 3:
		apply_tsne(target_csv=sys.argv[2])
	else:
		print("This option does not exist!\n")


if __name__ == '__main__':
	run(int(sys.argv[1]))