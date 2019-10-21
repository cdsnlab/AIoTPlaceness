# -*- coding: utf-8 -*-
import os
import collections
import shutil

from torchvision import transforms
from torchvision.datasets.folder import pil_loader
from torchvision.utils import save_image

import config
import re
import sys
import csv
import time
import random
import math
from tqdm import tqdm
from wordcloud import WordCloud
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.manifold import TSNE
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.cluster import Birch, SpectralClustering, AffinityPropagation, AgglomerativeClustering, KMeans, DBSCAN, \
    OPTICS

CONFIG = config.Config


def do_clustering(target_csv, cluster_method):
    num_cluster = 24
    df_data = pd.read_csv(os.path.join(CONFIG.CSV_PATH, target_csv + '.csv'), index_col=0, header=0,
                          encoding='utf-8-sig')
    df_data.index.name = 'short_code'
    print(df_data.iloc[:100])
    print(df_data.shape)

    start_time = time.time()
    if cluster_method == 0:
        clustering = DBSCAN(eps=0.3, min_samples=1000)
        clustering.fit(df_data)
        csv_name = 'clustered_dbscan_' + target_csv + '.csv'
    elif cluster_method == 1:
        clustering = OPTICS(min_samples=1000, metric='cosine')
        clustering.fit(df_data)
        csv_name = 'clustered_optics_' + target_csv + '.csv'
    elif cluster_method == 2:
        clustering = AgglomerativeClustering(n_clusters=num_cluster)
        clustering.fit(df_data)
        csv_name = 'clustered_ward_' + target_csv + '.csv'
    elif cluster_method == 3:
        clustering = AgglomerativeClustering(affinity='cosine', linkage='complete', n_clusters=num_cluster)
        clustering.fit(df_data)
        csv_name = 'clustered_agglo_complete_' + target_csv + '.csv'
    elif cluster_method == 4:
        clustering = AgglomerativeClustering(affinity='cosine', linkage='single', n_clusters=num_cluster)
        clustering.fit(df_data)
        csv_name = 'clustered_agglo_single_' + target_csv + '.csv'
    elif cluster_method == 5:
        clustering = Birch(n_clusters=num_cluster)
        clustering.fit(df_data)
        csv_name = 'clustered_birch_' + target_csv + '.csv'
    elif cluster_method == 6:
        clustering = KMeans(n_clusters=num_cluster)
        clustering.fit(df_data)
        csv_name = 'clustered_kmeans_' + target_csv + '.csv'
    elif cluster_method == 7:
        clustering = SpectralClustering(n_clusters=num_cluster, random_state=42, assign_labels='discretize')
        clustering.fit(df_data)
        csv_name = 'clustered_spectral_' + target_csv + '.csv'
    print("time elapsed for clustering: " + str(time.time() - start_time))
    print(clustering.get_params())
    print(clustering.labels_)
    count_percentage(clustering.labels_)
    result_df = pd.DataFrame(data=clustering.labels_, index=df_data.index, columns=['cluster'])

    start_time = time.time()
    print("calinski_harabasz_score: ", calinski_harabasz_score(df_data, result_df['cluster'].squeeze()))
    print("silhouette_score: ", silhouette_score(df_data, result_df['cluster'].squeeze()))
    print("davies_bouldin_score: ", davies_bouldin_score(df_data, result_df['cluster'].squeeze()))
    print("time elapsed for scoring: " + str(time.time() - start_time))
    result_df.to_csv(os.path.join(CONFIG.CSV_PATH, csv_name), encoding='utf-8-sig')


def count_percentage(cluster_labels):
    count = collections.Counter(cluster_labels)
    for k in count:
        print("cluster {} : {:.2%}".format(str(k), count[k] / len(cluster_labels)))


def do_spectral_clustering(target_csv):
    num_cluster = 24
    df_data = pd.read_csv(os.path.join(CONFIG.CSV_PATH, target_csv + '.csv'), index_col=0, header=0,
                          encoding='utf-8-sig')
    df_data.index.name = 'short_code'
    print(df_data.iloc[:100])
    print(df_data.shape)

    start_time = time.time()
    ds_data = df_data.sample(frac=0.5)
    ns_data = df_data.loc[set(df_data.index) - set(ds_data.index)]
    clustering = SpectralClustering(n_clusters=num_cluster, random_state=42, assign_labels='discretize')
    clustering.fit(ds_data)

    print("time elapsed for clustering: " + str(time.time() - start_time))
    print(clustering.get_params())
    print(clustering.labels_)
    count_percentage(clustering.labels_)
    start_time = time.time()
    result_ds = pd.DataFrame(data=clustering.labels_, index=ds_data.index, columns=['cluster'])
    ns_label = clustering.fit_predict(ns_data)
    result_ns = pd.DataFrame(data=ns_label, index=ns_data.index, columns=['cluster'])
    result_df = pd.concat([result_ds, result_ns])
    result_df.sort_index(inplace=True)
    print("time elapsed for prediction: " + str(time.time() - start_time))

    start_time = time.time()
    print("calinski_harabasz_score: ", calinski_harabasz_score(df_data, result_df['cluster'].squeeze()))
    print("silhouette_score: ", silhouette_score(df_data, result_df['cluster'].squeeze()))
    print("davies_bouldin_score: ", davies_bouldin_score(df_data, result_df['cluster'].squeeze()))
    print("time elapsed for scoring: " + str(time.time() - start_time))
    result_df.to_csv(os.path.join(CONFIG.CSV_PATH, 'clustered_spectral_' + target_csv + '.csv'), encoding='utf-8-sig')


def sample_from_cluster(target_csv, target_dataset, target_clustering):
    sample_length = 10
    df_clustered = pd.read_csv(os.path.join(CONFIG.CSV_PATH, target_csv + '.csv'), index_col=0, header=0,
                               encoding='utf-8-sig')
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
    df_pca_data = pd.read_csv(os.path.join(CONFIG.CSV_PATH, 'pca_' + target_csv + '.csv'), index_col=0, header=0,
                              encoding='utf-8-sig')
    lr_list = [20, 100, 200]
    perp_list = [50, 200, 500, 1000, 2000]
    for lr in lr_list:
        for perp in perp_list:
            start_time = time.time()
            # tsne_pca = do_tsne(TSNE(n_components=2, perplexity=50, early_exaggeration=12.0, learning_rate=100, n_iter=5000, random_state=42, verbose=1), df_pca_data)
            tsne_pca = do_tsne(TSNE(n_components=2, perplexity=perp, learning_rate=lr, n_iter=2000, random_state=42),
                               df_pca_data)
            print("time elapsed: " + str(time.time() - start_time))
            scatterplot_pointlabels(tsne_pca, 0.2)
            plt.title('t-SNE on PCA data lr: ' + str(lr) + ' perp: ' + str(perp))
            plt.savefig(
                os.path.join(CONFIG.SVG_PATH, 'tsne_pca_' + target_csv + '_' + str(lr) + '_' + str(perp) + '.svg'))
            tsne_pca.to_csv(
                os.path.join(CONFIG.CSV_PATH, 'tsne_pca_' + target_csv + '_' + str(lr) + '_' + str(perp) + '.csv'),
                encoding='utf-8-sig')


def do_tsne(tsne_object, data_to_pass):
    data_tsne = pd.DataFrame(tsne_object.fit_transform(data_to_pass))
    data_tsne.index = data_to_pass.index
    data_tsne.columns = [['tsne1', 'tsne2']]
    return data_tsne


def scatterplot_pointlabels(df_twocols, markersize=None):
    # basic scatterplot
    fig = plt.figure()
    plt.plot(df_twocols.iloc[:, 0], df_twocols.iloc[:, 1], marker='.', linestyle='None', markersize=markersize)


def test(target_dataset, target_clustering):
    result_path = os.path.join(CONFIG.RESULT_PATH, target_dataset)
    result_path = os.path.join(result_path, target_clustering)
    cluster_len = {i: [] for i in range(12)}
    for clustered_directory in os.listdir(result_path):
        clustered_path = os.path.join(result_path, clustered_directory)
        for short_code in os.listdir(clustered_path):
            short_code_path = os.path.join(clustered_path, short_code)
            for content in os.listdir(short_code_path):
                if content.endswith('.txt'):
                    with open(os.path.join(short_code_path, content), 'r', encoding='utf-8') as f:
                        data = f.read()
                    cluster_len[int(clustered_directory)].append(len(data.split()))
    print(cluster_len)
    for cluster in cluster_len:
        print(np.mean(cluster_len[cluster]))


def test2(target_dataset):
    df_data = pd.read_csv(os.path.join(CONFIG.TARGET_PATH, 'SEOUL_SUBWAY_DATA-3.csv'), encoding='utf-8-sig')

    pbar = tqdm(total=df_data.shape[0])
    for index, row in df_data.iterrows():
        if not pd.isna(row[2]):
            word_list = row[2].split()
            if 'asd' in word_list:
                print(row)
                print(row[7])
        pbar.update(1)
    pbar.close()


def sample_from_cluster_text_and_image(target_csv, target_dataset, confidence):
    sample_length = 10
    df_clustered = pd.read_csv(os.path.join(CONFIG.CSV_PATH, target_csv), index_col=0, header=0, encoding='utf-8-sig')
    df_clustered.index.name = 'short_code'

    confidence = float(confidence)
    if confidence != 0.:
        print("length of full data is " + str(len(df_clustered)))
        df_clustered = df_clustered[df_clustered['confidence'] > confidence]
        print("length of data above confidence is " + str(len(df_clustered)))
    num_cluster = int(np.max(df_clustered, axis=0)[0] + 1)

    result_path = os.path.join(CONFIG.RESULT_PATH, target_dataset)
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    target_clustering = target_csv.replace('.csv', '') + '_' + str(confidence)
    result_path = os.path.join(result_path, target_clustering)
    if not os.path.exists(result_path):
        os.mkdir(result_path)

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
        print("number of items in cluster " + str(key) + " is " + str(len(sampled)))
        if len(sampled) != 0:
            for short_code in sampled:
                short_code_dict[short_code] = key
        pbar.update(1)
    pbar.close()

    img_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
    image_dict = {k: [] for k in range(num_cluster)}
    text_dict = {k: "" for k in range(num_cluster)}
    print("sampling posts...")
    df_original = pd.read_csv(os.path.join(CONFIG.TARGET_PATH, 'SEOUL_SUBWAY_DATA-3.csv'), encoding='utf-8-sig')
    pbar = tqdm(total=df_original.shape[0])
    for index, row in df_original.iterrows():
        if row[1] in short_code_dict:
            cluster_id = short_code_dict[row[1]]
            image_path = row[7]
            image_tensor = img_transform(pil_loader(image_path))
            image_dict[cluster_id].append(image_tensor)
            text_dict[cluster_id] = text_dict[cluster_id] + row[2] + "\n"
        pbar.update(1)
    pbar.close()

    print("copying sampled posts...")
    pbar = tqdm(total=num_cluster)
    for cluster_id in range(num_cluster):
        caption_path = os.path.join(result_path, 'caption_' + str(cluster_id) + '.txt')
        image_path = os.path.join(result_path, 'images_' + str(cluster_id) + '.png')
        f_wr = open(caption_path, 'w', encoding='utf-8')
        f_wr.write(text_dict[cluster_id])
        f_wr.close()
        save_image(image_dict[cluster_id], image_path, nrow=2)
        pbar.update(1)
    pbar.close()


def make_word_cloud(target_csv, target_dataset, confidence):
    df_clustered = pd.read_csv(os.path.join(CONFIG.CSV_PATH, target_csv), index_col=0, header=0, encoding='utf-8-sig')
    df_clustered.index.name = 'short_code'

    confidence = float(confidence)
    if confidence != 0.:
        print("length of full data is " + str(len(df_clustered)))
        df_clustered = df_clustered[df_clustered['confidence'] > confidence]
        print("length of data above confidence is " + str(len(df_clustered)))

    cluster_dict = df_clustered['cluster_id'].to_dict()
    num_cluster = int(np.max(df_clustered, axis=0)[0] + 1)

    model_name = 'DOC2VEC_' + target_dataset + '.model'
    model = Doc2Vec.load(os.path.join(CONFIG.EMBEDDING_PATH, model_name))
    stop_vocab = ['UNK', '<EOS>']
    vocab2index = {}
    index2vocab = {}
    index = 0
    for vocab in model.wv.vocab:
        if vocab not in stop_vocab:
            vocab2index[vocab] = index
            index2vocab[index] = vocab
            index = index + 1

    vocab_array = np.zeros((len(vocab2index), num_cluster))

    result_path = os.path.join(CONFIG.RESULT_PATH, target_dataset)
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    target_clustering = target_csv.replace('.csv', '') + '_' + str(confidence)
    result_path = os.path.join(result_path, target_clustering)
    if not os.path.exists(result_path):
        os.mkdir(result_path)

    df_data = pd.read_csv(os.path.join(CONFIG.DATASET_PATH, target_dataset, 'posts.csv'), index_col=0, header=None,
                          encoding='utf-8-sig')
    pbar = tqdm(total=df_data.shape[0])
    for index, row in df_data.iterrows():
        word_list = row[1].split()
        if index not in cluster_dict:
            pbar.update(1)
            continue
        cluster_label = cluster_dict[index]
        for word in word_list:
            if word not in stop_vocab:
                vocab_index = vocab2index[word]
                vocab_array[vocab_index][cluster_label] = vocab_array[vocab_index][cluster_label] + 1
        pbar.update(1)
    pbar.close()
    salient_array = np.nan_to_num(vocab_array / np.sum(vocab_array, axis=1, keepdims=True))

    font_path = './fonts/NanumBarunGothic.ttf'
    for cluster_id in tqdm(range(num_cluster)):
        vocab_max = vocab_array[:, cluster_id].argsort()
        vocab_max = vocab_max[-100:]
        keywords = {}
        for vocab_index in vocab_max:
            salient_value = salient_array[vocab_index][cluster_id]
            vocab = index2vocab[vocab_index]
            if salient_value != 0:
                keywords[vocab] = salient_value

        wordcloud = WordCloud(
            font_path=font_path,
            width=600,
            height=600,
            background_color="white"
        )
        wordcloud = wordcloud.generate_from_frequencies(keywords)
        array = wordcloud.to_array()
        plt.figure(figsize=(7, 7))
        plt.imshow(array, interpolation="bilinear")
        plt.axis("off")
        plt.title("Word cloud of cluster " + str(cluster_id))
        plt.savefig(os.path.join(result_path, 'wordcloud_' + str(cluster_id) + '.png'))
        plt.close()


def run(option):
    if option == 0:
        do_clustering(target_csv=sys.argv[2], cluster_method=int(sys.argv[3]))
    elif option == 1:
        do_spectral_clustering(target_csv=sys.argv[2])
    elif option == 2:
        sample_from_cluster(target_csv=sys.argv[2], target_dataset=sys.argv[3], target_clustering=sys.argv[4])
    elif option == 3:
        apply_tsne(target_csv=sys.argv[2])
    elif option == 4:
        test(target_dataset=sys.argv[2], target_clustering=sys.argv[3])
    elif option == 5:
        test2(target_dataset=sys.argv[2])
    elif option == 6:
        sample_from_cluster_text_and_image(target_csv=sys.argv[2], target_dataset=sys.argv[3], confidence=sys.argv[4])
    elif option == 7:
        make_word_cloud(target_csv=sys.argv[2], target_dataset=sys.argv[3], confidence=sys.argv[4])
    else:
        print("This option does not exist!\n")


if __name__ == '__main__':
    run(int(sys.argv[1]))
