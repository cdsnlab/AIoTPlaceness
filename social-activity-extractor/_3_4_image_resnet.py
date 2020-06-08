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

import torchvision.models as models
from model.image_resnet import ImageModel
from model.util import load_multi_csv_data, load_semi_supervised_csv_data, load_text_data, load_image_data
from model.multidec import MDEC_encoder, MultiDEC

CONFIG = config.Config


def slacknoti(contentstr):
    webhook_url = "https://hooks.slack.com/services/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"
    payload = {"text": contentstr}
    requests.post(webhook_url, data=json.dumps(payload), headers={'Content-Type': 'application/json'})


def main():
    parser = argparse.ArgumentParser(description='text convolution-deconvolution auto-encoder model')
    # learning
    parser.add_argument('-lr', type=float, default=0.01, help='initial learning rate')
    parser.add_argument('-epochs', type=int, default=200, help='number of epochs for train')
    parser.add_argument('-update_time', type=int, default=1, help='update time within epoch')
    parser.add_argument('-batch_size', type=int, default=64, help='batch size for training')
    # data
    parser.add_argument('-target_dataset', type=str, default='seoul_subway', help='folder name of target dataset')
    parser.add_argument('-label_csv', type=str, default='category_label.csv', help='file name of target label')
    parser.add_argument('-sampled_n', type=int, default=None, help='number of fold')
    # model
    parser.add_argument('-dropout', type=float, default=0.5, help='dropout rate')
    parser.add_argument('-arch', type=str, default='resnet50', help='torchvision model')
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

class LastLayer(nn.Module):
    def __init__(self, in_features, n_clusters, dropout=0.):
        super(self.__class__, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(512, n_clusters),
            nn.Sigmoid(),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        out = self.fc1(x)
        log_prob = self.fc2(out)
        return log_prob

def train_multidec(args):
    print("Training test image resnet")
    device = torch.device(args.gpu)

    df_image_data = pd.read_csv(os.path.join(CONFIG.DATASET_PATH, args.target_dataset, 'posts.csv'), index_col=0, header=None,
                          encoding='utf-8-sig')
    print(df_image_data[:5])
    df_label = pd.read_csv(os.path.join(CONFIG.CSV_PATH, args.label_csv), index_col=0, encoding='utf-8-sig')
    df_image_data = df_image_data.loc[df_label.index]
    label_array = np.array(df_label['category'])
    n_clusters = np.max(label_array) + 1

    exp = Experiment("Image resnet", capture_io=True)

    for arg, value in vars(args).items():
        exp.param(arg, value)
    try:
        acc_list = []
        nmi_list = []
        f_1_list = []
        kf_count = 0
        for fold_idx in range(args.fold):
            print("Current fold: ", kf_count)
            df_train = pd.read_csv(os.path.join(CONFIG.CSV_PATH, "train_" + str(fold_idx) + "_" + args.label_csv),
                                  index_col=0,
                                  encoding='utf-8-sig')
            if args.sampled_n is not None:
                df_train = df_train.sample(n=args.sampled_n, random_state=42)
            df_test = pd.read_csv(os.path.join(CONFIG.CSV_PATH, "test_" + str(fold_idx) + "_" + args.label_csv),
                                  index_col=0,
                                  encoding='utf-8-sig')

            print("Loading dataset...")
            train_dataset, test_dataset = load_image_data(df_image_data, df_train, df_test, CONFIG)
            print("\nLoading dataset completed")
            if args.arch == 'placesCNN':
                print("Loading model trained in places")
                image_encoder = models.__dict__[args.arch](num_classes=365)
                checkpoint = torch.load(os.path.join(CONFIG.CHECKPOINT_PATH, 'resnet50_places365.pth.tar'),
                                        map_location=lambda storage, loc: storage)
                state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
                image_encoder.load_state_dict(state_dict)
            else:
                print("Loading model")
                image_encoder = models.__dict__[args.arch](pretrained=True)
            image_encoder.fc = LastLayer(image_encoder.fc.in_features, n_clusters, dropout=args.dropout)
            image_model = ImageModel(device=device, image_encoder=image_encoder)
            image_model.fit(train_dataset, lr=args.lr, batch_size=args.batch_size, num_epochs=args.epochs,
                     save_path=os.path.join(CONFIG.CHECKPOINT_PATH, "image_resnet.pt"))
            image_model.predict(test_dataset, batch_size=args.batch_size)
            acc_list.append(image_model.acc)
            nmi_list.append(image_model.nmi)
            f_1_list.append(image_model.f_1)
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


def make_de(df_text_data, df_train, dictionary_list, n_clusters=12):
    df_text_data = df_text_data.loc[df_train.index]
    dictionary_count = {word: np.zeros(n_clusters) for word in dictionary_list}
    for index, row in df_text_data.iterrows():
        word_list = row.iloc[0].split()
        label = df_train.loc[index][0]
        for word in word_list:
            if word in dictionary_count:
                dictionary_count[word][label] = dictionary_count[word][label] + 1
    dictionary_embedding = {}
    for index, value in dictionary_count.items():
        if np.sum(value) > 0:
            dictionary_embedding[index] = np.argmax(value)
    return dictionary_embedding

if __name__ == '__main__':
    main()
