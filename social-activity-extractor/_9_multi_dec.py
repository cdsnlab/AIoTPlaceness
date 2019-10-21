import argparse

import pandas as pd

import config
import requests
import json
from hyperdash import Experiment

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import util
from model.util import load_multi_csv_data
from model.multidec import MDEC_encoder, MultiDEC

CONFIG = config.Config


def slacknoti(contentstr):
    webhook_url = "https://hooks.slack.com/services/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"
    payload = {"text": contentstr}
    requests.post(webhook_url, data=json.dumps(payload), headers={'Content-Type': 'application/json'})


def main():
    parser = argparse.ArgumentParser(description='text convolution-deconvolution auto-encoder model')
    # learning
    parser.add_argument('-lr', type=float, default=1e-02, help='initial learning rate')
    parser.add_argument('-epochs', type=int, default=50, help='number of epochs for train')
    parser.add_argument('-batch_size', type=int, default=64, help='batch size for training')
    # data
    parser.add_argument('-image_csv', type=str, default=None, help='file name of target csv')
    parser.add_argument('-text_csv', type=str, default=None, help='file name of target csv')
    parser.add_argument('-split_rate', type=float, default=0.8, help='split rate between train and validation')
    # model
    parser.add_argument('-input_dim', type=int, default=300, help='size of input dimension')
    parser.add_argument('-latent_dim', type=int, default=10, help='size of latent variable')
    parser.add_argument('-n_clusters', type=int, default=10, help='desired num of cluster')
    # train
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
    print("Training multidec")
    device = torch.device(args.gpu)
    print("Loading dataset...")
    full_dataset = load_multi_csv_data(args, CONFIG)
    print("Loading dataset completed")
    # full_loader = DataLoader(full_dataset, batch_size=args.batch_size, shuffle=False)

    image_encoder = MDEC_encoder(input_dim=args.input_dim, z_dim=args.latent_dim, n_clusters=args.n_clusters,
                                 encodeLayer=[500, 500, 2000], activation="relu", dropout=0)
    image_encoder.load_model(os.path.join(CONFIG.CHECKPOINT_PATH, "image_sdae_" + str(args.latent_dim)) + ".pt")
    text_encoder = MDEC_encoder(input_dim=args.input_dim, z_dim=args.latent_dim, n_clusters=args.n_clusters,
                                encodeLayer=[500, 500, 2000], activation="relu", dropout=0)
    text_encoder.load_model(os.path.join(CONFIG.CHECKPOINT_PATH, "text_sdae_" + str(args.latent_dim)) + ".pt")
    mdec = MultiDEC(device=device, image_encoder=image_encoder, text_encoder=text_encoder, n_clusters=args.n_clusters)
    exp = Experiment("MDEC " + str(args.latent_dim) + '_' + str(args.n_clusters), capture_io=True)
    print(mdec)

    for arg, value in vars(args).items():
        exp.param(arg, value)
    try:
        mdec.fit(full_dataset, lr=args.lr, batch_size=args.batch_size, num_epochs=args.epochs)
        mdec.save_model(os.path.join(CONFIG.CHECKPOINT_PATH, "mdec_" + str(args.latent_dim)) + '_' + str(args.n_clusters) + ".pt")
        print("Finish!!!")

    finally:
        exp.end()


def eval_multidec(args):
    print("Evaluate multidec")
    device = torch.device(args.gpu)
    print("Loading dataset...")
    full_dataset = load_multi_csv_data(args, CONFIG)
    print("Loading dataset completed")
    # full_loader = DataLoader(full_dataset, batch_size=args.batch_size, shuffle=False)

    image_encoder = MDEC_encoder(input_dim=args.input_dim, z_dim=args.latent_dim, n_clusters=args.n_clusters,
                                 encodeLayer=[500, 500, 2000], activation="relu", dropout=0)
    image_encoder.load_model(os.path.join(CONFIG.CHECKPOINT_PATH, "image_sdae_" + str(args.latent_dim)) + ".pt")
    text_encoder = MDEC_encoder(input_dim=args.input_dim, z_dim=args.latent_dim, n_clusters=args.n_clusters,
                                encodeLayer=[500, 500, 2000], activation="relu", dropout=0)
    text_encoder.load_model(os.path.join(CONFIG.CHECKPOINT_PATH, "text_sdae_" + str(args.latent_dim)) + ".pt")
    mdec = MultiDEC(device=device, image_encoder=image_encoder, text_encoder=text_encoder)
    mdec.load_model(os.path.join(CONFIG.CHECKPOINT_PATH, "mdec_" + str(args.latent_dim)) + '_' + str(args.n_clusters) + ".pt")
    short_codes, y_pred = mdec.fit_predict(full_dataset, args.batch_size)

    result_df = pd.DataFrame(data=y_pred, index=short_codes, columns=['cluster_id', 'confidence'])
    result_df.index.name = "short_code"
    result_df.sort_index(inplace=True)
    result_df.to_csv(os.path.join(CONFIG.CSV_PATH, 'multidec_result_' + str(args.latent_dim) + '_' + str(args.n_clusters) + '.csv'), encoding='utf-8-sig')


if __name__ == '__main__':
    main()
