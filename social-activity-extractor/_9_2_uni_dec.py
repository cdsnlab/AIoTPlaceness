import argparse

import numpy as np
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
from sklearn.model_selection import train_test_split, StratifiedKFold
from model.util import load_semi_supervised_uni_csv_data
from model.unidec import UDEC_encoder, UniDEC

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
    parser.add_argument('-epochs', type=int, default=200, help='number of epochs for train')
    parser.add_argument('-update_time', type=int, default=1, help='update time within epoch')
    parser.add_argument('-batch_size', type=int, default=256, help='batch size for training')
    # data
    parser.add_argument('-prefix', type=str, default=None, help='prefix of file name')
    parser.add_argument('-input_csv', type=str, default='labeled_scaled_pca_normalized_image_encoded_seoul_subway.csv', help='file name of target csv')
    parser.add_argument('-target_modal', type=str, default='image', help='file name of target label')
    parser.add_argument('-label_csv', type=str, default='category_label.csv', help='file name of target label')
    # model
    parser.add_argument('-input_dim', type=int, default=300, help='size of input dimension')
    parser.add_argument('-latent_dim', type=int, default=10, help='size of latent variable')
    parser.add_argument('-use_prior', action='store_true', default=False, help='use prior knowledge')
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
    print("Training unidec")
    device = torch.device(args.gpu)
    df_input_data = pd.read_csv(os.path.join(CONFIG.CSV_PATH, args.prefix + "_" + args.input_csv), index_col=0,
                                encoding='utf-8-sig')

    df_label = pd.read_csv(os.path.join(CONFIG.CSV_PATH, args.label_csv), index_col=0, encoding='utf-8-sig')
    label_array = np.array(df_label['category'])
    n_clusters = np.max(label_array) + 1

    exp = Experiment(args.prefix + "_" + args.target_modal + "_UDEC", capture_io=True)

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
            full_dataset, train_dataset, val_dataset = load_semi_supervised_uni_csv_data(df_input_data, df_train,
                                                                                     df_test, CONFIG)
            print("\nLoading dataset completed")

            encoder = UDEC_encoder(input_dim=args.input_dim, z_dim=args.latent_dim, n_clusters=n_clusters,
                                         encodeLayer=[500, 500, 2000], activation="relu", dropout=0)
            encoder.load_model(os.path.join(CONFIG.CHECKPOINT_PATH, args.prefix + "_" + args.target_modal + "_sdae_" + str(fold_idx)) + ".pt")
            # encoder.load_model(os.path.join(CONFIG.CHECKPOINT_PATH, "sampled_plus_labeled_scaled_" + args.target_modal + "_sdae_" + str(fold_idx)) + ".pt")
            udec = UniDEC(device=device, encoder=encoder, use_prior=args.use_prior, n_clusters=n_clusters)
            udec.fit_predict(full_dataset, train_dataset, val_dataset, lr=args.lr, batch_size=args.batch_size, num_epochs=args.epochs,
                     save_path=os.path.join(CONFIG.CHECKPOINT_PATH, args.prefix + "_" + args.target_modal + "_udec_" + str(fold_idx)) + ".pt", tol=args.tol, kappa=args.kappa)
            acc_list.append(udec.acc)
            nmi_list.append(udec.nmi)
            f_1_list.append(udec.f_1)
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
