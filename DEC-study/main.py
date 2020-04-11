import argparse

import pandas as pd
import _pickle as cPickle

import config
import requests
import json
from hyperdash import Experiment

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from util import load_pretrain_data
from model.ddec import DualNet, DDEC

CONFIG = config.Config


def slacknoti(contentstr):
    webhook_url = "https://hooks.slack.com/services/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"
    payload = {"text": contentstr}
    requests.post(webhook_url, data=json.dumps(payload), headers={'Content-Type': 'application/json'})


def main():
    parser = argparse.ArgumentParser(description='text convolution-deconvolution auto-encoder model')
    # learning
    parser.add_argument('-lr', type=float, default=5e-04, help='initial learning rate')
    parser.add_argument('-epochs', type=int, default=50, help='number of epochs for train')
    parser.add_argument('-batch_size', type=int, default=128, help='batch size for training')
    # data
    parser.add_argument('-image_dir', type=str, default='/4TBSSD/IMAGE_EMBEDDED_SUBWAY_DATA', help='directory of embedded images')
    parser.add_argument('-data_csv', type=str, default='/4TBSSD/posts.csv', help='file name of target data csv')
    parser.add_argument('-text_embedding_dir', type=str, default='/4TBSSD/TEXT_EMBEDDED_SUBWAY_DATA', help='directory to text embedding')
    parser.add_argument('-split_rate', type=float, default=0.8, help='split rate between train and validation')
    # model
    parser.add_argument('-n_classes', type=int, default=12, help='desired num of cluster')
    parser.add_argument('-text_features', type=int, default=1024, help='number of dimensions')
    parser.add_argument('-z_dim', type=int, default=1024, help='number of dimensions')
    # train
    parser.add_argument('-noti', action='store_true', default=False, help='whether using gpu server')
    parser.add_argument('-gpu', type=str, default='cuda', help='gpu number')
    # option
    parser.add_argument('-mode', type=str, default='pretrain', help='pretrain, train, eval')

    args = parser.parse_args()

    if args.noti:
        slacknoti("underkoo start using")
    if args.mode == 'pretrain':
        pretrain_multidec(args)
    elif args.mode == 'train':
        train_multidec(args)
    elif args.mode == 'eval':
        eval_multidec(args)
    else:
        print('print select correct mode')
    if args.noti:
        slacknoti("underkoo end using")

def pretrain_multidec(args):
    print("Training multidec")
    device = torch.device(args.gpu)

    print("Loading dataset...")
    df_data = pd.read_csv(args.data_csv, index_col=0, header=None, encoding='utf-8')
    df_data.columns = ["caption", "path_to_image"]
    df_data.index.name = "shortcode"
    with open(os.path.join(args.text_embedding_dir, 'word_embedding.p'), "rb") as f:
        embedding_model = cPickle.load(f)
    with open(os.path.join(args.text_embedding_dir, 'word_idx.json'), "r", encoding='utf-8') as f:
        word_idx = json.load(f)
    df_train = pd.read_csv("/4TBSSD/train_0_category_label.csv",
                           index_col=0,
                           encoding='utf-8-sig')
    df_test = pd.read_csv("/4TBSSD/test_0_category_label.csv",
                          index_col=0,
                          encoding='utf-8-sig')
    train_dataset, test_dataset = load_pretrain_data(args.image_dir, word_idx[1], df_data, df_train, df_test, CONFIG)
    print("Loading dataset completed")

    dualnet = DualNet(device=device, pretrained_embedding=embedding_model, text_features=args.text_features, z_dim=args.z_dim, n_classes=args.n_classes)
    exp = Experiment("Dualnet_pretrain_" + str(args.n_classes), capture_io=True)
    print(dualnet)

    for arg, value in vars(args).items():
        exp.param(arg, value)
    try:
        dualnet.fit(train_dataset,  test_dataset, lr=args.lr, batch_size=args.batch_size, num_epochs=args.epochs,
                 save_path="/4TBSSD/CHECKPOINT/pretrain_0.pt")
        print("Finish!!!")

    finally:
        exp.end()


def train_multidec(args):
    print("Training multidec")
    device = torch.device(args.gpu)
    print("Loading dataset...")
    full_dataset = load_data(args, CONFIG)
    print("Loading dataset completed")
    # full_loader = DataLoader(full_dataset, batch_size=args.batch_size, shuffle=False)

    ddec = DDEC(device=device, n_classes=args.n_classes)
    exp = Experiment("SocialDEC " + str(args.latent_dim) + '_' + str(args.n_clusters), capture_io=True)
    print(ddec)

    for arg, value in vars(args).items():
        exp.param(arg, value)
    try:
        ddec.fit(full_dataset, lr=args.lr, batch_size=args.batch_size, num_epochs=args.epochs,
                 save_path=CONFIG.CHECKPOINT_PATH)
        print("Finish!!!")

    finally:
        exp.end()


def eval_multidec(args):
    print("Evaluate socialdec")
    device = torch.device(args.gpu)
    print("Loading dataset...")


if __name__ == '__main__':
    main()
