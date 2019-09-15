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
from sklearn.cluster import Birch, SpectralClustering, AffinityPropagation, AgglomerativeClustering, KMeans, DBSCAN, OPTICS

CONFIG = config.Config

def do_clustering(target_dataset, cluster_method):
	df_data = pd.read_csv(os.path.join(CONFIG.CSV_PATH, 'normalized_' + target_dataset + '.csv'), index_col=0, header=None, encoding='utf-8-sig')
	df_data.index.name = 'short_code'
	print(df_data.iloc[:100])
	print(df_data.shape)

	start_time = time.time()
	if cluster_method == 0:
		clustering = DBSCAN(eps=0.3, min_samples=20).fit(df_data)
		csv_name = 'clustered_dbscan_' + target_dataset + '.csv'
	elif cluster_method == 1:
		clustering = OPTICS(min_samples=20).fit(df_data)
		csv_name = 'clustered_optics_' + target_dataset + '.csv'
	elif cluster_method == 2:
		ds_data = df_data.sample(frac=0.5)
		print(ds_data.iloc[:100])
		print(ds_data.shape)
		clustering = SpectralClustering(n_clusters=21, random_state=42).fit(ds_data)
		csv_name = 'clustered_spectral_' + target_dataset + '.csv'
	elif cluster_method == 3:
		clustering = AgglomerativeClustering(n_clusters=21).fit(df_data)
		csv_name = 'clustered_agglomerative_' + target_dataset + '.csv'
	elif cluster_method == 4:
		clustering = Birch(n_clusters=21).fit(df_data)
		csv_name = 'clustered_birch_' + target_dataset + '.csv'
	print("time elapsed: " + str(time.time()-start_time))
	print(clustering.get_params())
	print(clustering.labels_)
	count_percentage(clustering.labels_)
	result_df = pd.DataFrame(data=clustering.labels_, index=df_data.index, columns=['cluster'])
	result_df.to_csv(os.path.join(CONFIG.CSV_PATH, csv_name), encoding='utf-8-sig')

def count_percentage(cluster_labels):
	count = collections.Counter(cluster_labels)
	for k in count:
		print("cluster {} : {:.2%}".format(str(k), count[k]/len(cluster_labels)))


def run(option): 
	if option == 0:
		do_clustering(target_dataset=sys.argv[2], cluster_method=int(sys.argv[3]))
	else:
		print("This option does not exist!\n")


if __name__ == '__main__':
	run(int(sys.argv[1]))