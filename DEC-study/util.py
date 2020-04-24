import collections
import operator

import torch
import torchvision.transforms as transforms
from torchvision.datasets.folder import pil_loader
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

import torch.nn.functional as F
from torch.utils.data import Dataset, random_split
import math
import os
import sys
import numpy as np
import pandas as pd
import _pickle as cPickle
from tqdm import tqdm
import config

torch.manual_seed(42)

def load_pretrain_data(image_dir, token_to_index, args, CONFIG):

    df_raw_text = pd.read_csv(args.raw_text_csv, index_col=0, header=None, encoding='utf-8')
    df_raw_text.columns = ["caption", "path_to_image"]
    df_raw_text.index.name = "shortcode"

    df_image_data = pd.read_csv(args.image_csv, index_col=0, encoding='utf-8-sig')
    df_text_data = pd.read_csv(args.text_csv, index_col=0, encoding='utf-8-sig')

    df_raw_text = df_raw_text.loc[df_image_data.index]

    df_test = pd.read_csv("/4TBSSD/test_0_category_label.csv",
                          index_col=0,
                          encoding='utf-8-sig')
    train_short_codes = []
    train_raw_text = []
    train_image_data = []
    train_text_data = []
    test_index = set(df_test.index)
    test_short_codes = []
    test_raw_text = []
    test_image_data = []
    test_text_data = []

    pbar = tqdm(total=df_raw_text.shape[0])
    for index, row in df_raw_text.iterrows():
        if index in test_index:
            test_short_codes.append(index)
            word_list = df_raw_text.loc[index]['caption'].split()
            test_raw_text.append(encode_text(word_list, CONFIG.MAX_SENTENCE_LEN, token_to_index))
            test_image_data.append(np.array(df_image_data.loc[index]))
            test_text_data.append(np.array(df_text_data.loc[index]))
        else:
            train_short_codes.append(index)
            word_list = df_raw_text.loc[index]['caption'].split()
            train_raw_text.append(encode_text(word_list, CONFIG.MAX_SENTENCE_LEN, token_to_index))
            train_image_data.append(np.array(df_image_data.loc[index]))
            train_text_data.append(np.array(df_text_data.loc[index]))

        pbar.update(1)
    pbar.close()
    train_dataset = LabeledDataset(image_dir, token_to_index, train_short_codes, train_raw_text, np.array(train_image_data), np.array(train_text_data), CONFIG)
    test_dataset = LabeledDataset(image_dir, token_to_index, test_short_codes, test_raw_text, np.array(test_image_data), np.array(test_text_data), CONFIG)
    return train_dataset, test_dataset

def encode_text(word_list, sentence_len, token_to_index):
    vec = torch.zeros(sentence_len).long()
    for i, token in enumerate(word_list):
        if i == sentence_len:
            break
        index = token_to_index.get(token, 0)
        vec[i] = index
    return vec, min(len(word_list), sentence_len)

class LabeledDataset(Dataset):
    def __init__(self, image_dir, token_to_index, short_codes, raw_text, image_data, text_data, CONFIG):
        self.short_codes = short_codes
        self.raw_text = raw_text
        self.CONFIG = CONFIG
        self.image_dir = image_dir
        self.image_data = image_data
        self.text_data = text_data
        self.token_to_index = token_to_index

    def __len__(self):
        return len(self.short_codes)

    def __getitem__(self, idx):
        with open(os.path.join(self.image_dir, self.short_codes[idx]) + '.p', "rb") as f:
            image_data = cPickle.load(f)
        image_tensor = torch.from_numpy(image_data).type(torch.FloatTensor)
        text_tensor = self.raw_text[idx][0]
        text_length = self.raw_text[idx][1]
        target_image_tensor = torch.from_numpy(self.image_data[idx]).type(torch.FloatTensor)
        target_text_tensor = torch.from_numpy(self.text_data[idx]).type(torch.FloatTensor)
        return self.short_codes[idx], image_tensor, text_tensor, text_length, target_image_tensor, target_text_tensor

def load_data(image_dir, token_to_index, args, CONFIG):
    df_raw_text = pd.read_csv(args.raw_text_csv, index_col=0, header=None, encoding='utf-8')
    df_raw_text.columns = ["caption", "path_to_image"]
    df_raw_text.index.name = "shortcode"

    df_image_data = pd.read_csv(args.image_csv, index_col=0, encoding='utf-8-sig')
    df_raw_text = df_raw_text.loc[df_image_data.index]
    df_train = pd.read_csv("/4TBSSD/train_0_category_label.csv",
                          index_col=0,
                          encoding='utf-8-sig')
    df_test = pd.read_csv("/4TBSSD/test_0_category_label.csv",
                          index_col=0,
                          encoding='utf-8-sig')
    full_short_codes = []
    full_raw_text = []
    train_index = set(df_train.index)
    train_short_codes = []
    train_raw_text = []
    train_label_data = []
    test_index = set(df_test.index)
    test_short_codes = []
    test_raw_text = []
    test_label_data = []
    # if sample is not None:
    #     labeled_index = train_index.union(test_index)
    #     df_sampled_data = df_raw_text.loc[df_raw_text.index.difference(labeled_index)].sample(n=sample, random_state=42)
    #     df_labeled_data = df_raw_text.loc[labeled_index]
    #     df_raw_text = pd.concat([df_sampled_data, df_labeled_data])
    #     df_raw_text = df_raw_text.sample(frac=1, random_state=42)
    pbar = tqdm(total=df_raw_text.shape[0])
    for index, row in df_raw_text.iterrows():
        word_list = df_raw_text.loc[index]['caption'].split()
        temp = encode_text(word_list, CONFIG.MAX_SENTENCE_LEN, token_to_index)
        if index in train_index:
            full_short_codes.append(index)
            full_raw_text.append(temp)
            train_short_codes.append(index)
            train_label_data.append(df_train.loc[index][0])
            train_raw_text.append(temp)
        elif index in test_index:
            test_short_codes.append(index)
            test_label_data.append(df_test.loc[index][0])
            test_raw_text.append(temp)
        else:
            full_short_codes.append(index)
            full_raw_text.append(temp)

        pbar.update(1)
    pbar.close()
    full_dataset = UnlabeledDataset(image_dir, token_to_index, full_short_codes, full_raw_text, CONFIG)
    train_dataset = LabeledDataset(image_dir, token_to_index, train_short_codes, train_raw_text, train_label_data, CONFIG)
    test_dataset = LabeledDataset(image_dir, token_to_index, test_short_codes, test_raw_text, test_label_data, CONFIG)
    return full_dataset, train_dataset, test_dataset

class UnlabeledDataset(Dataset):
    def __init__(self, image_dir, token_to_index, short_codes, raw_text, CONFIG):
        self.short_codes = short_codes
        self.raw_text = raw_text
        self.CONFIG = CONFIG
        self.image_dir = image_dir
        self.token_to_index = token_to_index

    def __len__(self):
        return len(self.short_codes)

    def __getitem__(self, idx):
        with open(os.path.join(self.image_dir, self.short_codes[idx]) + '.p', "rb") as f:
            image_data = cPickle.load(f)
        image_tensor = torch.from_numpy(image_data).type(torch.FloatTensor)
        text_tensor = self.raw_text[idx][0]
        text_length = self.raw_text[idx][1]
        return self.short_codes[idx], image_tensor, text_tensor, text_length

# def load_pretrain_data(image_dir, token_to_index, df_text_data, df_train, df_test, CONFIG):
#     train_index = set(df_train.index)
#     train_short_codes = []
#     train_text_data = []
#     train_label_data = []
#     test_index = set(df_test.index)
#     test_short_codes = []
#     test_text_data = []
#     test_label_data = []
#
#     pbar = tqdm(total=df_text_data.shape[0])
#     for index, row in df_text_data.iterrows():
#         if index in train_index:
#             train_short_codes.append(index)
#             train_text_data.append(np.array(df_text_data.loc[index]['caption']))
#             train_label_data.append(df_train.loc[index][0])
#         elif index in test_index:
#             test_short_codes.append(index)
#             test_text_data.append(np.array(df_text_data.loc[index]['caption']))
#             test_label_data.append(df_test.loc[index][0])
#         pbar.update(1)
#     pbar.close()
#     train_dataset = PretrainDataset(image_dir, token_to_index, train_short_codes, np.array(train_text_data), train_label_data, CONFIG)
#     test_dataset = PretrainDataset(image_dir, token_to_index, test_short_codes, np.array(test_text_data), test_label_data, CONFIG)
#     return train_dataset, test_dataset
#
#
# class PretrainDataset(Dataset):
#     def __init__(self, image_dir, token_to_index, short_codes, text_data, label_data, CONFIG):
#         self.short_codes = short_codes
#         self.text_data = text_data
#         self.CONFIG = CONFIG
#         self.image_dir = image_dir
#         self.label_data = label_data
#         self.token_to_index = token_to_index
#
#     def __len__(self):
#         return len(self.short_codes)
#
#     def __getitem__(self, idx):
#         with open(os.path.join(self.image_dir, self.short_codes[idx]), "rb") as f:
#             image_data = cPickle.load(f)
#         image_tensor = torch.from_numpy(image_data).type(torch.FloatTensor)
#         text_tensor = self._encode_text(self.text_data[idx])
#         return self.short_codes[idx], image_tensor, text_tensor, self.label_data[idx]
#
#     def _encode_text(self, text):
#         """ Turn a question into a vector of indices and a question length """
#         vec = torch.zeros(self.CONFIG.MAX_SENTENCE_LEN).type(torch.LongTensor)
#         word_list = text.split()
#         for i, token in enumerate(word_list):
#             index = self.token_to_index.get(token, 0)
#             vec[i] = index
#         return vec, len(word_list)