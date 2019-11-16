import argparse
import config
import requests
import json
import pickle
import datetime
import os
import csv
import math
import numpy as np
import pandas as pd
import _pickle as cPickle
from sumeval.metrics.rouge import RougeCalculator
from sumeval.metrics.bleu import BLEUCalculator
from gensim.models.keyedvectors import FastTextKeyedVectors
from gensim.similarities.index import AnnoyIndexer
from hyperdash import Experiment
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchfile
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import StepLR, CyclicLR
from model import util
from model.util import load_image_data_with_short_code

CONFIG = config.Config


def slacknoti(contentstr):
    webhook_url = "https://hooks.slack.com/services/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"
    payload = {"text": contentstr}
    requests.post(webhook_url, data=json.dumps(payload), headers={'Content-Type': 'application/json'})


def main():
    parser = argparse.ArgumentParser(description='text convolution-deconvolution auto-encoder model')
    parser.add_argument('-tau', type=float, default=0.01, help='temperature parameter')
    # data
    parser.add_argument('-target_dataset', type=str, default='seoul_subway', help='folder name of target dataset')
    parser.add_argument('-shuffle', default=True, help='shuffle data every epoch')
    parser.add_argument('-split_rate', type=float, default=0.9, help='split rate between train and validation')
    parser.add_argument('-batch_size', type=int, default=16, help='batch size for training')

    # model
    parser.add_argument('-arch', type=str, default='resnet50', help='image embedding model')
    parser.add_argument('-placesCNN', action='store_true', default=False, help='whether using gpu server')

    # train
    parser.add_argument('-noti', action='store_true', default=False, help='whether using gpu server')
    parser.add_argument('-gpu', type=str, default='cuda', help='gpu number')
    # option
    parser.add_argument('-checkpoint', type=str, default=None, help='filename of checkpoint to resume ')

    args = parser.parse_args()

    if args.noti:
        slacknoti("underkoo start using")
    get_latent(args)
    if args.noti:
        slacknoti("underkoo end using")


class last_layer(nn.Module):
    def __init__(self):
        super(last_layer, self).__init__()

    def forward(self, x):
        return x


def get_latent(args):
    device = torch.device(args.gpu)
    print("Loading embedding model...")
    if args.placesCNN:
        print("Loading model trained in places")
        image_encoder = models.__dict__[args.arch](num_classes=365)
        checkpoint = torch.load(os.path.join(CONFIG.CHECKPOINT_PATH, 'resnet50_places365.pth.tar'),
                                map_location=lambda storage, loc: storage)
        state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
        image_encoder.load_state_dict(state_dict)
    else:
        image_encoder = models.__dict__[args.arch](pretrained=True)
    args.latent_size = image_encoder.fc.in_features
    image_encoder.fc = last_layer()

    print("Loading embedding model completed")
    print("Loading dataset...")
    full_dataset = load_image_data_with_short_code(args, CONFIG)
    print("Loading dataset completed")
    full_loader = DataLoader(full_dataset, batch_size=args.batch_size, shuffle=False)
    image_encoder.to(device)
    image_encoder.eval()

    if args.placesCNN:
        csv_name = 'places_encoded_' + args.target_dataset + '.csv'
    else:
        csv_name = 'image_encoded_' + args.target_dataset + '.csv'

    short_code_list = []
    row_list = []
    for short_code, image_batch in tqdm(full_loader):
        # torch.cuda.empty_cache()
        with torch.no_grad():
            image_feature = Variable(image_batch).to(device)
        h = image_encoder(image_feature)

        for _short_code, _h in zip(short_code, h):
            short_code_list.append(_short_code)
            row_list.append(_h.detach().cpu().numpy().tolist())
        del image_feature

    result_df = pd.DataFrame(data=row_list, index=short_code_list, columns=[i for i in range(args.latent_size)])
    result_df.index.name = "short_code"
    result_df.sort_index(inplace=True)
    result_df.to_csv(os.path.join('/ssdmnt/placeness', csv_name), encoding='utf-8-sig')
    print("Finish!!!")


if __name__ == '__main__':
    main()
