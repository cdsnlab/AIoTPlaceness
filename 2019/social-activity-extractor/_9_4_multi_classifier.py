import argparse
import datetime

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
from sklearn.model_selection import train_test_split, KFold

from model.Multi_Classifier import MultiClassifier
from model.Single_Classifier import SingleClassifier
from model.Weight_Calculator import WeightCalculator
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
    parser.add_argument('-lr', type=float, default=1e-04, help='initial learning rate')
    parser.add_argument('-pretrain_epochs', type=int, default=100, help='number of epochs for train')
    parser.add_argument('-epochs', type=int, default=50, help='number of epochs for train')
    parser.add_argument('-batch_size', type=int, default=256, help='batch size for training')
    # data
    parser.add_argument('-image_csv', type=str, default='labeled_scaled_pca_normalized_image_encoded_seoul_subway.csv', help='file name of target csv')
    parser.add_argument('-text_csv', type=str, default='labeled_scaled_text_doc2vec_seoul_subway.csv', help='file name of target csv')
    parser.add_argument('-label_csv', type=str, default='category_label.csv', help='file name of target label')
    parser.add_argument('-weight_csv', type=str, default='weight_label.csv', help='file name of weight label')
    parser.add_argument('-split_fold', type=int, default=5, help='number of split fold between train and validation')
    # model
    parser.add_argument('-input_dim', type=int, default=300, help='size of input dimension')
    parser.add_argument('-fixed_weight', type=float, default=None, help='if set, weight is fixed')
    # train
    parser.add_argument('-noti', action='store_true', default=False, help='whether using gpu server')
    parser.add_argument('-gpu', type=str, default='cuda', help='gpu number')

    args = parser.parse_args()

    if args.noti:
        slacknoti("underkoo start using")
    train_multidec(args)
    if args.noti:
        slacknoti("underkoo end using")


def train_multidec(args):
    print("Training started")
    device = torch.device(args.gpu)
    df_image_data = pd.read_csv(os.path.join(CONFIG.CSV_PATH, args.image_csv), index_col=0,
                                encoding='utf-8-sig')
    df_text_data = pd.read_csv(os.path.join(CONFIG.CSV_PATH, args.text_csv), index_col=0,
                               encoding='utf-8-sig')

    df_label = pd.read_csv(os.path.join(CONFIG.CSV_PATH, args.label_csv), index_col=0, encoding='utf-8-sig')
    df_weight = pd.read_csv(os.path.join(CONFIG.CSV_PATH, args.weight_csv), index_col=0, encoding='utf-8-sig')
    short_code_array = np.array(df_label.index)
    label_array = np.array(df_label['category'])
    n_classes = np.max(label_array) + 1


    exp = Experiment("multi_classifier", capture_io=True)
    for arg, value in vars(args).items():
        exp.param(arg, value)
    try:
        kf = KFold(n_splits=5, random_state=42)
        image_score_list = []
        text_score_list = []
        multi_score_list = []
        kf_count = 0
        for train_index, val_index in kf.split(short_code_array):
            print("Current fold: ", kf_count)
            short_code_train = short_code_array[train_index]
            short_code_val = short_code_array[val_index]
            label_train = label_array[train_index]
            label_val = label_array[val_index]
            df_train = pd.DataFrame(data=label_train, index=short_code_train, columns=df_label.columns)
            df_val = pd.DataFrame(data=label_val, index=short_code_val, columns=df_label.columns)
            print("Loading dataset...")
            train_dataset, val_dataset = load_multi_csv_data(df_image_data, df_text_data, df_weight, df_train, df_val,
                                                             CONFIG)
            print("\nLoading dataset completed")

            if args.fixed_weight is None:
                image_classifier = SingleClassifier(device=device, input_dim=args.input_dim, filter_num=64,
                                                    n_classes=n_classes)
                text_classifier = SingleClassifier(device=device, input_dim=args.input_dim, filter_num=64,
                                                   n_classes=n_classes)
                print("pretraining image classifier...")
                image_classifier.fit(train_dataset, val_dataset, input_modal=1, lr=args.lr, num_epochs=args.pretrain_epochs,
                                     save_path=os.path.join(CONFIG.CHECKPOINT_PATH, "image_classifier") + ".pt")
                image_classifier.load_model(os.path.join(CONFIG.CHECKPOINT_PATH, "image_classifier") + ".pt")
                print("pretraining text classifier...")
                text_classifier.fit(train_dataset, val_dataset, input_modal=2, lr=args.lr, num_epochs=args.pretrain_epochs,
                                                   save_path=os.path.join(CONFIG.CHECKPOINT_PATH, "text_classifier") + ".pt")
                text_classifier.load_model(os.path.join(CONFIG.CHECKPOINT_PATH, "text_classifier") + ".pt")
                print("pretraining weight classifier...")
                weight_calculator = WeightCalculator(device=device, input_dim=args.input_dim*2, n_classes=n_classes)
                weight_calculator.fit(train_dataset, val_dataset, lr=args.lr, num_epochs=args.pretrain_epochs,
                                      save_path=os.path.join(CONFIG.CHECKPOINT_PATH, "weight_calculator") + ".pt")
                weight_calculator.load_model(os.path.join(CONFIG.CHECKPOINT_PATH, "weight_calculator") + ".pt")
                multi_classifier = MultiClassifier(device=device, image_classifier=image_classifier,
                                                   text_classifier=text_classifier, weight_calculator=weight_calculator)
                print(multi_classifier)
                print("training multi classifier...")
                multi_classifier.fit(train_dataset, val_dataset, lr=args.lr, batch_size=args.batch_size, num_epochs=args.epochs,
                                     save_path=os.path.join(CONFIG.CHECKPOINT_PATH, "multi_classifier") + ".pt")
            else:
                image_classifier = SingleClassifier(device=device, input_dim=args.input_dim, filter_num=64,
                                                    n_classes=n_classes)
                text_classifier = SingleClassifier(device=device, input_dim=args.input_dim, filter_num=64,
                                                   n_classes=n_classes)
                print("pretraining image classifier...")
                image_classifier.fit(train_dataset, val_dataset, input_modal=1, lr=args.lr, num_epochs=args.pretrain_epochs,
                                     save_path=os.path.join(CONFIG.CHECKPOINT_PATH, "image_classifier_fw_" + str(args.fixed_weight)) + ".pt")
                image_classifier.load_model(os.path.join(CONFIG.CHECKPOINT_PATH, "image_classifier_fw_" + str(args.fixed_weight)) + ".pt")
                print("pretraining text classifier...")
                text_classifier.fit(train_dataset, val_dataset, input_modal=2, lr=args.lr, num_epochs=args.pretrain_epochs,
                                                   save_path=os.path.join(CONFIG.CHECKPOINT_PATH, "text_classifier_fw_" + str(args.fixed_weight)) + ".pt")
                text_classifier.load_model(os.path.join(CONFIG.CHECKPOINT_PATH, "text_classifier_fw_" + str(args.fixed_weight)) + ".pt")
                multi_classifier = MultiClassifier(device=device, image_classifier=image_classifier,
                                                   text_classifier=text_classifier, fixed_weight=args.fixed_weight)
                print(multi_classifier)
                print("training multi classifier with fixed weight...")
                multi_classifier.fit(train_dataset, val_dataset, lr=args.lr, batch_size=args.batch_size, num_epochs=args.epochs,
                                     save_path=os.path.join(CONFIG.CHECKPOINT_PATH, "multi_classifier_fw_" + str(args.fixed_weight)) + ".pt")

            print("Finish!!!")
            print("#current fold best image score: %.6f, text score: %.6f multi score: %.6f" %
                  (image_classifier.score, text_classifier.score, multi_classifier.score))
            image_score_list.append(image_classifier.score)
            text_score_list.append(text_classifier.score)
            multi_score_list.append(multi_classifier.score)
            kf_count = kf_count + 1

        print("#average image score: %.6f, text score: %.6f multi score: %.6f" % (
        np.mean(image_score_list), np.mean(text_score_list), np.mean(multi_score_list)))

    finally:
        exp.end()

if __name__ == '__main__':
    main()
