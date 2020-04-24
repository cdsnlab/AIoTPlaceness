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
from util import load_pretrain_data, load_data
from model.ddec import DualNet, DDEC

CONFIG = config.Config


def slacknoti(contentstr):
    webhook_url = "https://hooks.slack.com/services/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"
    payload = {"text": contentstr}
    requests.post(webhook_url, data=json.dumps(payload), headers={'Content-Type': 'application/json'})


def main():
    parser = argparse.ArgumentParser(description='text convolution-deconvolution auto-encoder model')
    # learning
    parser.add_argument('-lr', type=float, default=1e-02, help='initial learning rate')
    parser.add_argument('-kappa', type=float, default=0.1, help='lr adjust rate')
    parser.add_argument('-tol', type=float, default=1e-03, help='tolerance for early stopping')
    parser.add_argument('-pretrain_epochs', type=int, default=100, help='number of epochs for train')
    parser.add_argument('-epochs', type=int, default=50, help='number of epochs for train')
    parser.add_argument('-batch_size', type=int, default=128, help='batch size for training')
    # data
    parser.add_argument('-image_dir', type=str, default='/4TBSSD/IMAGE_EMBEDDED_SUBWAY_DATA', help='directory of embedded images')
    parser.add_argument('-raw_text_csv', type=str, default='/4TBSSD/posts.csv', help='file name of target data csv')
    parser.add_argument('-image_csv', type=str, default='/4TBSSD/sampled_plus_labeled_scaled_pca_normalized_image_encoded_seoul_subway.csv', help='file name of target data csv')
    parser.add_argument('-text_csv', type=str, default='/4TBSSD/sampled_plus_labeled_scaled_text_doc2vec_seoul_subway.csv', help='file name of target data csv')
    parser.add_argument('-text_embedding_dir', type=str, default='/4TBSSD/TEXT_EMBEDDED_SUBWAY_DATA', help='directory to text embedding')
    parser.add_argument('-split_rate', type=float, default=0.8, help='split rate between train and validation')
    # model
    parser.add_argument('-n_classes', type=int, default=12, help='desired num of cluster')
    parser.add_argument('-text_features', type=int, default=1024, help='number of dimensions')
    parser.add_argument('-z_dim', type=int, default=10, help='number of dimensions')
    parser.add_argument('-use_prior', action='store_true', default=False, help='use prior knowledge')
    # train
    parser.add_argument('-noti', action='store_true', default=False, help='whether using gpu server')
    parser.add_argument('-gpu', type=str, default='cuda', help='gpu number')
    parser.add_argument('-sample', type=int, default=None, help='sample train data')
    parser.add_argument('-skip_semi', action='store_true', default=False, help='whether skip semi-supervised learning')
    parser.add_argument('-resume', action='store_true', default=False, help='whether resume training')
    parser.add_argument('-optimizer', type=str, default='sgd', help='optimizer to train')

    # option
    parser.add_argument('-mode', type=str, default='pretrain', help='pretrain, train, eval')

    args = parser.parse_args()

    if args.noti:
        slacknoti("underkoo start using")
    if args.mode == 'pretrain':
        pretrain_ddec(args)
    elif args.mode == 'train':
        train_ddec(args)
    elif args.mode == 'eval':
        eval_ddec(args)
    else:
        print('print select correct mode')
    if args.noti:
        slacknoti("underkoo end using")

def pretrain_ddec(args):
    print("Pretraining...")

    print("Loading dataset...")
    with open(os.path.join(args.text_embedding_dir, 'word_embedding.p'), "rb") as f:
        embedding_model = cPickle.load(f)
    with open(os.path.join(args.text_embedding_dir, 'word_idx.json'), "r", encoding='utf-8') as f:
        word_idx = json.load(f)
    train_dataset, test_dataset = load_pretrain_data(args.image_dir, word_idx[1], args, CONFIG)
    print("Loading dataset completed")

    dualnet = DualNet(pretrained_embedding=embedding_model, text_features=args.text_features, z_dim=args.z_dim, n_classes=args.n_classes)
    if args.resume:
        print("loading model...")
        dualnet.load_model("/4TBSSD/CHECKPOINT/pretrain_" + str(args.z_dim) + "_0.pt")
    exp = Experiment("Dualnet_pretrain_" + str(args.z_dim), capture_io=True)
    print(dualnet)

    for arg, value in vars(args).items():
        exp.param(arg, value)
    try:
        dualnet.fit(train_dataset,  test_dataset, args=args,
                 save_path="/4TBSSD/CHECKPOINT/pretrain_" + str(args.z_dim) + "_0.pt")
        print("Finish!!!")

    finally:
        exp.end()


def train_ddec(args):
    print("Training...")
    device = torch.device(args.gpu)

    print("Loading dataset...")
    with open(os.path.join(args.text_embedding_dir, 'word_embedding.p'), "rb") as f:
        embedding_model = cPickle.load(f)
    with open(os.path.join(args.text_embedding_dir, 'word_idx.json'), "r", encoding='utf-8') as f:
        word_idx = json.load(f)
    full_dataset, train_dataset, test_dataset = load_data(args.image_dir, word_idx[1], args, CONFIG)
    print("Loading dataset completed")
    dualnet = DualNet(pretrained_embedding=embedding_model, text_features=args.text_features,
                      z_dim=args.z_dim, n_classes=args.n_classes)
    dualnet.load_model("/4TBSSD/CHECKPOINT/pretrain_" + str(args.z_dim) + "_0.pt")
    ddec = DDEC(pretrained_model=dualnet, n_classes=args.n_classes, z_dim=args.z_dim, use_prior=args.use_prior)
    if args.resume:
        print("loading model...")
        ddec.load_model("/4TBSSD/CHECKPOINT/train_" + str(args.z_dim) + "_0.pt")
    exp = Experiment("Dualnet_train_" + str(args.z_dim), capture_io=True)
    print(ddec)

    for arg, value in vars(args).items():
        exp.param(arg, value)
    try:
        ddec.fit(full_dataset, train_dataset,  test_dataset, args=args,
                 save_path="/4TBSSD/CHECKPOINT/train_" + str(args.z_dim) + "_0.pt")
        print("Finish!!!")

    finally:
        exp.end()


def eval_ddec(args):
    print("Evaluate socialdec")
    device = torch.device(args.gpu)
    print("Loading dataset...")


if __name__ == '__main__':
    main()
