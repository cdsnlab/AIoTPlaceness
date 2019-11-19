import argparse

import numpy as np
import pandas as pd

import config
import requests
import json
import _pickle as cPickle
from hyperdash import Experiment

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import util
from sklearn.model_selection import train_test_split, StratifiedKFold

from model.text_lstm import LSTMClassifier, TextModel
from model.util import load_multi_csv_data, load_semi_supervised_csv_data, load_text_data
from model.multidec import MDEC_encoder, MultiDEC

CONFIG = config.Config


def slacknoti(contentstr):
    webhook_url = "https://hooks.slack.com/services/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"
    payload = {"text": contentstr}
    requests.post(webhook_url, data=json.dumps(payload), headers={'Content-Type': 'application/json'})


def main():
    parser = argparse.ArgumentParser(description='text convolution-deconvolution auto-encoder model')
    # learning
    parser.add_argument('-lr', type=float, default=1e-03, help='initial learning rate')
    parser.add_argument('-epochs', type=int, default=500, help='number of epochs for train')
    parser.add_argument('-update_time', type=int, default=1, help='update time within epoch')
    parser.add_argument('-batch_size', type=int, default=256, help='batch size for training')
    # data
    parser.add_argument('-target_dataset', type=str, default='seoul_subway', help='folder name of target dataset')
    parser.add_argument('-label_csv', type=str, default='category_label.csv', help='file name of target label')
    # model
    parser.add_argument('-input_dim', type=int, default=300, help='size of input dimension')
    parser.add_argument('-hidden_size', type=int, default=256, help='size of latent variable')
    parser.add_argument('-dropout', type=float, default=0.5, help='dropout rate')
    # train
    parser.add_argument('-fold', type=int, default=5, help='number of fold')
    parser.add_argument('-noti', action='store_true', default=False, help='whether using gpu server')
    parser.add_argument('-gpu', type=str, default='cuda', help='gpu number')
    # option
    parser.add_argument('-eval', action='store_true', default=False, help='whether evaluate or train it')

    args = parser.parse_args()

    if args.noti:
        slacknoti("underkoo start using")
    if args.eval:
        eval_multidec(args)
    else:
        train_multidec(args)
    if args.noti:
        slacknoti("underkoo end using")


def train_multidec(args):
    print("Training test lstm")
    device = torch.device(args.gpu)

    with open(os.path.join(CONFIG.DATASET_PATH, args.target_dataset, 'word_embedding.p'), "rb") as f:
        embedding_model = cPickle.load(f)
    with open(os.path.join(CONFIG.DATASET_PATH, args.target_dataset, 'word_idx.json'), "r", encoding='utf-8') as f:
        word_idx = json.load(f)
    df_text_data = pd.read_csv(os.path.join(CONFIG.DATASET_PATH, args.target_dataset, 'posts.csv'), index_col=0, header=None,
                          encoding='utf-8-sig')
    print(df_text_data[:5])
    df_label = pd.read_csv(os.path.join(CONFIG.CSV_PATH, args.label_csv), index_col=0, encoding='utf-8-sig')
    df_text_data = df_text_data.loc[df_label.index]
    label_array = np.array(df_label['category'])
    n_clusters = np.max(label_array) + 1

    exp = Experiment("Text lstm", capture_io=True)

    for arg, value in vars(args).items():
        exp.param(arg, value)
    try:
        acc_list = []
        nmi_list = []
        f_1_list = []
        kf_count = 0
        for fold_idx in range(args.fold):
            print("Current fold: ", kf_count)
            df_train = pd.read_csv(os.path.join(CONFIG.CSV_PATH, "train_" + str(fold_idx) + "_category_label.csv"),
                                   index_col=0,
                                   encoding='utf-8-sig')
            df_test = pd.read_csv(os.path.join(CONFIG.CSV_PATH, "test_" + str(fold_idx) + "_category_label.csv"),
                                  index_col=0,
                                  encoding='utf-8-sig')
            print("Loading dataset...")
            train_dataset, test_dataset = load_text_data(df_text_data, df_train, df_test, CONFIG, word2idx=word_idx[1])
            print("\nLoading dataset completed")
            embedding = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_model))
            text_encoder = LSTMClassifier(device=device, batch_size=args.batch_size, output_size=n_clusters, hidden_size=[128, 256, 512],
                                          embedding=embedding, dropout=args.dropout)
            text_model = TextModel(device=device, text_encoder=text_encoder)
            text_model.fit(train_dataset, lr=args.lr, batch_size=args.batch_size, num_epochs=args.epochs,
                     save_path=CONFIG.CHECKPOINT_PATH)
            text_model.predict(test_dataset, batch_size=args.batch_size)
            acc_list.append(text_model.acc)
            nmi_list.append(text_model.nmi)
            f_1_list.append(text_model.f_1)
            kf_count = kf_count + 1
        print("#Average acc: %.4f, Average nmi: %.4f, Average f_1: %.4f" % (
            np.mean(acc_list), np.mean(nmi_list), np.mean(f_1_list)))

    finally:
        exp.end()


def eval_multidec(args):
    print("Evaluate multidec")
    device = torch.device(args.gpu)
    print("Loading dataset...")
    full_dataset = load_multi_csv_data(args, CONFIG)
    print("Loading dataset completed")
    # full_loader = DataLoader(full_dataset, batch_size=args.batch_size, shuffle=False)

    image_encoder = MDEC_encoder(input_dim=args.input_dim, z_dim=args.latent_dim, n_clusters=n_clusters,
                                 encodeLayer=[500, 500, 2000], activation="relu", dropout=0)
    image_encoder.load_model(os.path.join(CONFIG.CHECKPOINT_PATH, "image_sdae_" + str(args.latent_dim)) + ".pt")
    text_encoder = MDEC_encoder(input_dim=args.input_dim, z_dim=args.latent_dim, n_clusters=n_clusters,
                                encodeLayer=[500, 500, 2000], activation="relu", dropout=0)
    text_encoder.load_model(os.path.join(CONFIG.CHECKPOINT_PATH, "text_sdae_" + str(args.latent_dim)) + ".pt")
    mdec = MultiDEC(device=device, image_encoder=image_encoder, text_encoder=text_encoder)
    mdec.load_model(
        os.path.join(CONFIG.CHECKPOINT_PATH, "mdec_" + str(args.latent_dim)) + '_' + ".pt")
    short_codes, y_pred, y_confidence, pvalue = mdec.fit_predict(full_dataset, args.batch_size)

    result_df = pd.DataFrame(data={'cluster_id': y_pred, 'confidence': y_confidence}, index=short_codes)
    result_df.index.name = "short_code"
    result_df.sort_index(inplace=True)
    result_df.to_csv(
        os.path.join(CONFIG.CSV_PATH, 'multidec_result_' + str(args.latent_dim) + '_' + '.csv'),
        encoding='utf-8-sig')

    pvalue_df = pd.DataFrame(data=pvalue, index=short_codes, columns=[str(i) for i in range(args.n_clusters)])
    pvalue_df.index.name = "short_code"
    pvalue_df.sort_index(inplace=True)
    pvalue_df.to_csv(
        os.path.join(CONFIG.CSV_PATH, 'multidec_pvalue_' + str(args.latent_dim) + '_' + '.csv'),
        encoding='utf-8-sig')


if __name__ == '__main__':
    main()
