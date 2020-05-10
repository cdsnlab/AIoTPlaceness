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
from model.util import load_csv_data, load_autoencoder_data
from model.stackedDAE import StackedDAE

CONFIG = config.Config


def slacknoti(contentstr):
    webhook_url = "https://hooks.slack.com/services/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"
    payload = {"text": contentstr}
    requests.post(webhook_url, data=json.dumps(payload), headers={'Content-Type': 'application/json'})


def main():
    parser = argparse.ArgumentParser(description='text convolution-deconvolution auto-encoder model')
    # learning
    parser.add_argument('-lr', type=float, default=1e-01, help='initial learning rate')
    parser.add_argument('-pretrain_epochs', type=int, default=100, help='number of epochs for train')
    parser.add_argument('-epochs', type=int, default=200, help='number of epochs for train')
    parser.add_argument('-batch_size', type=int, default=64, help='batch size for training')
    # data
    parser.add_argument('-prefix', type=str, default=None, help='prefix of file name')
    parser.add_argument('-target_csv', type=str, default=None, help='file name of target csv')
    parser.add_argument('-target_modal', type=str, default=None, help='file name of target csv')
    parser.add_argument('-shuffle', default=True, help='shuffle data every epoch')
    # model
    parser.add_argument('-input_dim', type=int, default=300, help='size of input dimension')
    parser.add_argument('-latent_dim', type=int, default=10, help='size of latent variable')
    parser.add_argument('-dropout', type=float, default=0, help='dropout rate')
    # train
    parser.add_argument('-fold', type=int, default=5, help='number of fold')
    parser.add_argument('-noti', action='store_true', default=False, help='whether using gpu server')
    parser.add_argument('-gpu', type=str, default='cuda', help='gpu number')
    # option
    parser.add_argument('-resume', action='store_true', default=False, help='whether using gpu server')

    args = parser.parse_args()

    if args.noti:
        slacknoti("underkoo start using")
    train_reconstruction(args)
    if args.noti:
        slacknoti("underkoo end using")


def train_reconstruction(args):
    device = torch.device(args.gpu)

    df_input_data = pd.read_csv(os.path.join(CONFIG.CSV_PATH, args.prefix + "_" + args.target_csv), index_col=0,
                                encoding='utf-8-sig')
    exp = Experiment(args.target_modal + " SDAE " + str(args.latent_dim), capture_io=True)
    try:
        for arg, value in vars(args).items():
            exp.param(arg, value)
        for fold_idx in range(args.fold):
            print("Loading dataset...")

            df_test = pd.read_csv(os.path.join(CONFIG.CSV_PATH, "test_" + str(fold_idx) + "_category_label.csv"), index_col=0,
                                  encoding='utf-8-sig')
            df_input_without_test = df_input_data.loc[set(df_input_data.index) - set(df_test.index)]
            train_dataset, val_dataset = load_autoencoder_data(df_input_without_test, CONFIG)
            print("Loading dataset completed")
            train_loader, val_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=args.shuffle), \
                                       DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

            sdae = StackedDAE(input_dim=args.input_dim, z_dim=args.latent_dim, binary=False,
                              encodeLayer=[500, 500, 2000], decodeLayer=[2000, 500, 500], activation="relu",
                              dropout=args.dropout, device=device)
            if args.resume:
                print("resume from checkpoint")
                sdae.load_model(os.path.join(CONFIG.CHECKPOINT_PATH, args.prefix + "_" + args.target_modal + "_sdae_" + str(args.latent_dim) + '_' + str(fold_idx)) + ".pt")
            else:
                sdae.pretrain(train_loader, val_loader, lr=args.lr, batch_size=args.batch_size,
                              num_epochs=args.pretrain_epochs, corrupt=0.2, loss_type="mse")
            sdae.fit(train_loader, val_loader, lr=args.lr, num_epochs=args.epochs, corrupt=0.2, loss_type="mse",
                     save_path=os.path.join(CONFIG.CHECKPOINT_PATH, args.prefix + "_" + args.target_modal + "_sdae_" + str(args.latent_dim) + '_' + str(fold_idx)) + ".pt")
    finally:
        exp.end()


if __name__ == '__main__':
    main()
